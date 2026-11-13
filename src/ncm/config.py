"""Immutable configuration: changes deliberately create new executables."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CoarseConfig:
    # Opt-in preserves the state schema and numerics of historical experiments.
    enabled: bool = False
    warmup_batches: int = 2
    kmeans_iterations: int = 20
    initialization_noise: float = 1e-4
    usage_decay: float = 0.9
    dead_code_threshold: float = 1.0  # EMA assignments per global batch, not a fraction
    dead_code_grace_steps: int = 25
    dead_code_check_every: int = 1
    reset_count: float = 2.0
    replacement_noise: float = 0.01
    entropy_weight: float = 0.05
    temp_start: float = 2.0
    temp_end: float = 0.25
    temp_anneal_steps: int = 200
    beta_start: float = 0.0
    beta_end: float = 0.25
    beta_warmup_steps: int = 200
    coarse_codebook_lr_scale: float = 0.25

    def __post_init__(self):
        positive_ints = ('warmup_batches', 'kmeans_iterations', 'dead_code_check_every')
        nonnegative_ints = ('dead_code_grace_steps', 'temp_anneal_steps', 'beta_warmup_steps')
        for name in positive_ints + nonnegative_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or value < (1 if name in positive_ints else 0):
                raise ValueError(f'{name} has an invalid step count')
        for name in ('initialization_noise', 'replacement_noise', 'dead_code_threshold',
                     'entropy_weight', 'beta_start', 'beta_end'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 <= self.usage_decay < 1:
            raise ValueError('usage_decay must be in [0, 1)')
        if not all(math.isfinite(t) and t > 0 for t in (self.temp_start, self.temp_end, self.reset_count)):
            raise ValueError('temperatures and reset_count must be finite and positive')
        if not math.isfinite(self.coarse_codebook_lr_scale) or not 0 < self.coarse_codebook_lr_scale <= 1:
            raise ValueError('coarse_codebook_lr_scale must be in (0, 1]')


@dataclass(frozen=True)
class RVQConfig:
    lengths: tuple[int, ...]
    codebook_size: int
    dim: int
    decay: float = 0.99
    epsilon: float = 1e-5
    entropy_temperature: float = 1.0
    nested_dropout: float = 0.0
    coarse: CoarseConfig = CoarseConfig()

    def __post_init__(self):
        if not self.lengths or any(n <= 0 for n in self.lengths):
            raise ValueError("lengths must be nonempty and positive")
        if tuple(sorted(self.lengths)) != self.lengths:
            raise ValueError("lengths must be ordered coarse to fine")
        if self.codebook_size < 2 or self.dim <= 0:
            raise ValueError("codebook_size >= 2 and dim > 0 required")
        if not 0 <= self.decay < 1 or self.epsilon <= 0:
            raise ValueError("decay must be in [0, 1), epsilon > 0")
        if self.entropy_temperature <= 0 or not 0 <= self.nested_dropout <= 1:
            raise ValueError("invalid temperature or nested_dropout")

    @property
    def levels(self):
        return len(self.lengths)

    @property
    def max_length(self):
        return self.lengths[-1]


@dataclass(frozen=True)
class TransformerConfig:
    width: int = 256
    heads: int = 8
    layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.width, self.heads, self.layers, self.mlp_ratio) <= 0:
            raise ValueError("Transformer dimensions must be positive")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("width must divide heads; dropout must be in [0, 1)")


@dataclass(frozen=True)
class LossConfig:
    beta: float = 0.25
    entropy: float = 0.01
    concept: float = 0.0
    prior: float = 1.0
    # Empty means linearly decreasing K, ..., 1 (coarse levels get more weight).
    entropy_weights: tuple[float, ...] = ()


def experiment_rvq_config(config):
    """Shared loader for flat experiment dimensions and optional coarse settings."""
    return RVQConfig(tuple(config['lengths']), config['codebook_size'], config['dim'],
                     coarse=CoarseConfig(**config.get('coarse', {})))


@dataclass(frozen=True)
class NCMConfig:
    """Static prompt bucket and number of autoregressive coarse grids."""
    ar_coarse_levels: int = 0
    prompt_len: int = 0

    def __post_init__(self):
        for name in ('ar_coarse_levels', 'prompt_len'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
