import io
import json

from ncm.engine_rpc import response, serve


def test_ping_is_jsonrpc_and_unknown_method_is_method_error():
    assert response({"jsonrpc": "2.0", "id": 9, "method": "engine.ping"}) == {
        "jsonrpc": "2.0", "id": 9, "result": {"status": "ready"}}
    error = response({"jsonrpc": "2.0", "id": "x", "method": "no.such.method"})
    assert error["error"]["code"] == -32601 and error["id"] == "x"


def test_stdio_is_one_json_response_per_input_line_including_parse_errors():
    source = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"engine.ping"}\nnot json\n')
    sink = io.StringIO()
    serve(source, sink)
    rows = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [row["id"] for row in rows] == [1, None]
    assert rows[0]["result"] == {"status": "ready"}
    assert rows[1]["error"]["code"] == -32700


def test_reconstruction_metrics_ignore_padding_and_return_mergeable_counts():
    import numpy as np
    from ncm.engine_rpc import compute_reconstruction_metrics

    target = np.array([[1, 2, 3], [4, 5, 6]])
    decoded = np.array([[1, 2, 0], [4, 0, 6]])
    mask = np.array([[True, True, False], [True, False, True]])
    result = compute_reconstruction_metrics(target, decoded,
        [np.array([[0, 1, -1], [0, -1, 1]])], codebook_size=2, mask=mask)
    assert float(result["ber"]) == 0.0
    assert float(result["exact_match"]) == 1.0
    assert int(result["valid_bytes"]) == 4
    np.testing.assert_array_equal(result["codebook_counts"], [[2, 2]])


def test_rvq_prior_generation_restores_codec_and_prior_and_is_seeded(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import serialization
    from ncm.autoregressive import AutoregressiveCategorical
    from ncm.config import LossConfig, TransformerConfig, experiment_rvq_config
    from ncm.engine_rpc import _generate_rvq_prior
    from ncm.prior_training import PriorTrainState
    from ncm.training import Batch, NCM, initialize

    config = dict(lengths=[2, 4], codebook_size=4, dim=4, width=8,
                  heads=2, layers=1, learning_rate=.001, seed=9)
    rvq = experiment_rvq_config(config)
    transformer = TransformerConfig(8, 2, 1)
    model = NCM(rvq, transformer, 256, LossConfig(prior=0.))
    optimizer = optax.adam(config['learning_rate'])
    tokens = jnp.zeros((1, 4), jnp.int32)
    batch = Batch(tokens, jnp.ones_like(tokens, bool), jnp.zeros_like(tokens),
                  jnp.zeros_like(tokens, bool))
    codec_state = initialize(model, optimizer, jax.random.key(9), batch,
                             initialize_codebooks=False)
    codec_path = tmp_path / 'codec.msgpack'
    codec_path.write_bytes(serialization.to_bytes(codec_state))

    prior = AutoregressiveCategorical(transformer, sum(rvq.lengths), rvq.codebook_size)
    prior_input = jnp.full((1, prior.max_length), prior.vocab_size, jnp.int32)
    prior_params = prior.init(jax.random.key(10), prior_input,
                              jnp.ones_like(prior_input, bool))['params']
    prior_optimizer = optax.adam(config['learning_rate'])
    prior_state = PriorTrainState(prior_params, prior_optimizer.init(prior_params),
                                  jnp.asarray(1, jnp.int32))
    prior_path = tmp_path / 'prior.msgpack'
    prior_path.write_bytes(serialization.to_bytes(prior_state))
    request = dict(model_config=config, codec_checkpoint=str(codec_path),
                  prior_checkpoint=str(prior_path), prompt=list(b'hi'),
                  max_length=2, temperature=.8, top_k=3, top_p=.95, seed=42)

    first = _generate_rvq_prior(request)
    second = _generate_rvq_prior(request)
    assert first['tokens'] == second['tokens']
    np.testing.assert_allclose(first['log_probabilities'], second['log_probabilities'])
    assert len(first['tokens'][0]) == len(b'hi') + 2
    assert first['tokens'][0][:len(b'hi')] == list(b'hi')
    assert len(first['log_probabilities'][0]) == sum(rvq.lengths)
    assert all(np.isfinite(first['log_probabilities'][0]))
    assert np.isfinite(first['metrics']['mean_entropy_nats'])
    assert first['metrics']['generated_bytes'] == 2
