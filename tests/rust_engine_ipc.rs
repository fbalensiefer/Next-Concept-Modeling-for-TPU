use ncm::rpc::{
    CheckpointRequest, EngineClient, EngineConfig, ForwardRequest, InitModelRequest, LossConfig,
    ModelConfig, RvqConfig, TensorBatch, TrainStepRequest, TransformerConfig,
};
use serde_json::{Value, json};
use std::{
    env,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_request() -> InitModelRequest {
    InitModelRequest {
        config: EngineConfig {
            rvq: RvqConfig {
                lengths: vec![1, 2],
                codebook_size: 2,
                dim: 2,
                ..RvqConfig::default()
            },
            transformer: TransformerConfig {
                width: 4,
                heads: 2,
                layers: 1,
                mlp_ratio: 2,
                dropout: 0.,
            },
            model: ModelConfig::default(),
            losses: LossConfig::default(),
            token_vocab_size: 256,
            learning_rate: 1e-3,
            init_batch_size: 1,
            device: Default::default(),
        },
        seed: 17,
    }
}

fn batch() -> TensorBatch {
    TensorBatch {
        tokens: vec![vec![1, 2]],
        mask: None,
        fallback_bytes: None,
        fallback_mask: None,
        prompt_ids: None,
        prompt_mask: None,
    }
}

#[test]
fn rust_drives_model_training_and_checkpoint_round_trip_over_stdio() {
    let mut engine = EngineClient::start(&repository_root()).expect("start JAX worker");
    let initialized = engine
        .init_model(&model_request())
        .expect("initialize model");
    assert_eq!(initialized.step, 0);
    assert!(!initialized.engine_id.is_empty());
    assert!(!initialized.devices.is_empty());

    let forward = engine
        .forward(&ForwardRequest {
            engine_id: initialized.engine_id.clone(),
            batch: batch(),
        })
        .expect("forward pass");
    assert!(forward.loss.is_finite());
    assert!(forward.metrics.is_object());
    assert!(!forward.codes.is_null());

    let trained = engine
        .train_step(&TrainStepRequest {
            engine_id: initialized.engine_id.clone(),
            batch: batch(),
            seed: 23,
            phase: 0,
        })
        .expect("training step");
    assert_eq!(trained.step, 1);
    assert!(trained.loss.is_finite());
    assert!(trained.grad_norm.is_finite());
    assert!(trained.step_time_ms.is_finite() && trained.step_time_ms >= 0.);
    let checkpoint_forward = engine
        .forward(&ForwardRequest {
            engine_id: initialized.engine_id.clone(),
            batch: batch(),
        })
        .expect("forward before checkpoint");

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let checkpoint =
        env::temp_dir().join(format!("ncm-rpc-{}-{unique}.msgpack", std::process::id()));
    let saved = engine
        .save_checkpoint(&CheckpointRequest {
            engine_id: initialized.engine_id.clone(),
            path: checkpoint.to_string_lossy().into_owned(),
        })
        .expect("save checkpoint");
    assert_eq!(saved.step, 1);

    let restored = engine
        .init_model(&model_request())
        .expect("initialize restore target");
    let loaded = engine
        .load_checkpoint(&CheckpointRequest {
            engine_id: restored.engine_id.clone(),
            path: checkpoint.to_string_lossy().into_owned(),
        })
        .expect("load checkpoint");
    assert_eq!(loaded.step, 1);

    let restored_forward = engine
        .forward(&ForwardRequest {
            engine_id: restored.engine_id,
            batch: batch(),
        })
        .expect("forward pass after restore");
    assert_eq!(checkpoint_forward.codes, restored_forward.codes);
    assert_eq!(checkpoint_forward.loss, restored_forward.loss);
    let _ = std::fs::remove_file(checkpoint);
}

#[test]
fn rust_client_propagates_worker_rpc_errors() {
    let mut engine = EngineClient::start(&repository_root()).expect("start JAX worker");
    let error = engine
        .call::<_, Value>("not.a.method", &json!({}))
        .expect_err("worker method error must propagate");
    assert!(error.contains("-32601"), "unexpected error: {error}");
}
