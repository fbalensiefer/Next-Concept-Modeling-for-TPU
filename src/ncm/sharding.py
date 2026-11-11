"""NamedSharding contract for global-array jit on a 2D dp/fsdp mesh."""
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def create_mesh(dp: int, fsdp: int, devices=None) -> Mesh:
    """Call jax.distributed.initialize() first when using multiple hosts."""
    devices = list(jax.devices() if devices is None else devices)
    if dp <= 0 or fsdp <= 0 or dp * fsdp != len(devices):
        raise ValueError("dp * fsdp must equal the number of provided devices")
    # Callers may supply topology-aware ordering for their TPU slice.
    return Mesh(np.asarray(devices, dtype=object).reshape(dp, fsdp), ("dp", "fsdp"))


def replicated(mesh):
    return NamedSharding(mesh, P())


def parameter_shardings(params, mesh):
    """Shard the largest divisible axis of each matrix/tensor over fsdp.

    Biases/norm scales and indivisible leaves replicate. Optimizer moment
    leaves follow this same deterministic shape-based rule. Explicit specs
    can replace this policy for topology-specific optimization.
    """
    def choose(leaf):
        if leaf.ndim < 2 or mesh.shape["fsdp"] == 1:
            return replicated(mesh)
        candidates = [i for i, n in enumerate(leaf.shape)
                      if n > 0 and n % mesh.shape["fsdp"] == 0]
        if not candidates:
            return replicated(mesh)
        axis = max(candidates, key=lambda i: leaf.shape[i])
        spec = [None] * leaf.ndim
        spec[axis] = "fsdp"
        return NamedSharding(mesh, P(*spec))
    return jax.tree.map(choose, params)


def train_state_shardings(state, mesh):
    return state.replace(
        params=parameter_shardings(state.params, mesh),
        opt_state=parameter_shardings(state.opt_state, mesh),
        codebooks=jax.tree.map(lambda _: replicated(mesh), state.codebooks),
        step=replicated(mesh))


def batch_shardings(batch, mesh):
    def choose(leaf):
        if leaf.shape[0] % mesh.shape["dp"]:
            raise ValueError("global batch size must be divisible by dp")
        return NamedSharding(mesh, P("dp", *([None] * (leaf.ndim - 1))))
    return jax.tree.map(choose, batch)


def put_process_local_batch(local_batch, mesh):
    """Assemble global arrays from each host's local batch on multi-host TPU."""
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            NamedSharding(mesh, P("dp", *([None] * (x.ndim - 1)))), x),
        local_batch)


def compile_train_step(step, state, batch, mesh, *, donate=False):
    """Return (compiled step, placed state, placed global batch).

    `batch` here is already global (or is a complete host array on one host).
    Multi-host callers should use put_process_local_batch before this helper.
    Global reductions produce replicated EMA state, without an unbound psum.
    """
    state_specs = train_state_shardings(state, mesh)
    batch_specs = batch_shardings(batch, mesh)
    scalar = replicated(mesh)
    compiled = jax.jit(step,
                       in_shardings=(state_specs, batch_specs, scalar, scalar),
                       out_shardings=(state_specs, scalar),
                       donate_argnums=(0,) if donate else ())
    return compiled, jax.device_put(state, state_specs), jax.device_put(batch, batch_specs)
