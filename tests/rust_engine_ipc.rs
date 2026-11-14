use ncm::rpc::{
    CausalArConfig, CausalArEvaluateRequest, CausalArInitRequest, CausalArTrainStepRequest,
    CheckpointRequest, EngineClient, EngineConfig, ForwardRequest, GenerateRequest,
    InitModelRequest, LossConfig, ModelConfig, RvqConfig, SpeculativeVerifyRequest, TensorBatch,
    TrainStepRequest, TransformerConfig,
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
    let generated = engine
        .generate(&GenerateRequest {
            engine_id: initialized.engine_id.clone(),
            prompts: vec![vec![], vec![]],
            max_new_tokens: 2,
            temperature: 1.,
            top_k: 0,
            seed: 12,
        })
        .expect("generate from NCM model");
    assert_eq!(generated.tokens.len(), 2);
    assert!(generated.tokens.iter().all(|row| row.len() == 2));

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

#[test]
fn rust_drives_jax_probe_training_over_stdio() {
    let mut engine = EngineClient::start(&repository_root()).expect("start JAX worker");
    let response: Value = engine
        .call(
            "probe.train",
            &json!({
                "train": [[0,0],[0,1],[1,0],[1,1]],
                "evaluation": [[0,0],[1,1]],
                "lengths": [1,1],
                "level": 1,
                "seed": 7,
                "steps": 1,
                "batch_size": 2,
                "causal": true
            }),
        )
        .expect("train conditional probe");
    assert!(response["nll"].as_f64().is_some_and(f64::is_finite));
}

#[test]
fn rust_drives_causal_ar_evaluation_generation_and_speculative_verification() {
    let mut engine = EngineClient::start(&repository_root()).expect("start JAX worker");
    let model = engine
        .causal_ar_init(&CausalArInitRequest {
            config: CausalArConfig {
                transformer: TransformerConfig {
                    width: 8,
                    heads: 2,
                    layers: 1,
                    mlp_ratio: 2,
                    dropout: 0.,
                },
                max_length: 8,
                learning_rate: 1e-3,
            },
            seed: 3,
            tokens: vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1]],
            mask: None,
        })
        .expect("initialize causal AR");
    let teacher = engine
        .causal_ar_init(&CausalArInitRequest {
            config: CausalArConfig {
                transformer: TransformerConfig {
                    width: 8,
                    heads: 2,
                    layers: 1,
                    mlp_ratio: 2,
                    dropout: 0.,
                },
                max_length: 8,
                learning_rate: 1e-3,
            },
            seed: 4,
            tokens: vec![vec![1, 2, 3, 4]],
            mask: None,
        })
        .expect("initialize teacher AR");
    let trained = engine
        .causal_ar_train_step(&CausalArTrainStepRequest {
            engine_id: model.engine_id.clone(),
            tokens: vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1]],
            mask: None,
            continuation_start: 1,
            teacher_engine_id: Some(teacher.engine_id),
            distillation_weight: 0.5,
        })
        .expect("train causal AR");
    assert_eq!(trained.step, 1);
    assert!(trained.loss.is_finite() && trained.grad_norm.is_finite());
    let eval = engine
        .causal_ar_evaluate(&CausalArEvaluateRequest {
            engine_id: model.engine_id.clone(),
            tokens: vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1]],
            mask: None,
            prompt_len: 1,
        })
        .expect("evaluate causal AR");
    assert_eq!(eval.token_count, 6);
    assert!(eval.perplexity.is_finite() && eval.log_likelihoods.len() == 2);
    let generated = engine
        .generate(&GenerateRequest {
            engine_id: model.engine_id.clone(),
            prompts: vec![vec![1, 2], vec![3, 4]],
            max_new_tokens: 3,
            temperature: 0.8,
            top_k: 5,
            seed: 10,
        })
        .expect("generate conditioned tokens");
    assert_eq!(generated.tokens.len(), 2);
    assert!(generated.tokens.iter().all(|x| x.len() == 3));
    assert_eq!(generated.probabilities[0].len(), 3);
    assert!(
        generated
            .probabilities
            .iter()
            .flatten()
            .all(|row| (row.iter().sum::<f32>() - 1.).abs() < 1e-4)
    );
    let draft_tokens = generated
        .tokens
        .iter()
        .map(|row| row[..2].to_vec())
        .collect();
    let proposal_probs = generated
        .probabilities
        .into_iter()
        .map(|row| row.into_iter().take(2).collect())
        .collect();
    let verified = engine
        .speculative_verify(&SpeculativeVerifyRequest {
            engine_id: model.engine_id,
            prefixes: vec![vec![1, 2], vec![3, 4]],
            draft_tokens,
            proposal_probs,
            temperature: 0.8,
            top_k: 5,
            seed: 13,
        })
        .expect("verify proposals");
    assert_eq!(verified.tokens.len(), 2);
    assert_eq!(verified.tokens[0].len(), 3);
    assert!(verified.acceptance_rate.is_finite() && (0. ..=1.).contains(&verified.acceptance_rate));
}
