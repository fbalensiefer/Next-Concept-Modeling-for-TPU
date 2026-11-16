use ncm::rpc::{
    CausalArConfig, CausalArEvaluateRequest, CausalArInitRequest, CausalArTrainStepRequest,
    CheckpointRequest, EngineClient, EngineConfig, ForwardRequest, GenerateAndVerifyAtomicRequest,
    GenerateRaggedRequest, GenerateRequest, InitModelRequest, LossConfig, ModelConfig, RvqConfig,
    SpeculativeVerifyRaggedRequest, SpeculativeVerifyRequest, TensorBatch, TrainStepRequest,
    TransformerConfig,
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

#[derive(Debug, PartialEq, Eq)]
struct LegacySpeculation {
    tokens: Vec<Vec<u32>>,
    accepted: usize,
    proposed: usize,
    accepted_by_position: Vec<usize>,
    proposed_by_position: Vec<usize>,
    batch_rounds: usize,
    sequence_rounds: usize,
    prefix_sum: usize,
}

struct LegacySpeculationConfig<'a> {
    target_id: &'a str,
    drafter_id: &'a str,
    ncm_drafter: bool,
    output_len: usize,
    draft_len: usize,
    seed: u32,
}

fn benchmark_seed(
    seed: u32,
    prompt_len: usize,
    batch: usize,
    draft: usize,
    repetition: usize,
    round: usize,
    stream: u32,
) -> u32 {
    let mut value = u64::from(seed)
        ^ (prompt_len as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (batch as u64).rotate_left(11)
        ^ (draft as u64).rotate_left(23)
        ^ (repetition as u64).wrapping_mul(0xd6e8feb86659fd93)
        ^ (round as u64).wrapping_mul(0xa0761d6478bd642f)
        ^ u64::from(stream);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    value as u32
}

fn legacy_speculation(
    engine: &mut EngineClient,
    prompts: &[Vec<i32>],
    run: LegacySpeculationConfig<'_>,
) -> LegacySpeculation {
    let LegacySpeculationConfig {
        target_id,
        drafter_id,
        ncm_drafter,
        output_len,
        draft_len,
        seed,
    } = run;
    let mut contexts = prompts.to_vec();
    let mut result = LegacySpeculation {
        tokens: Vec::new(),
        accepted: 0,
        proposed: 0,
        accepted_by_position: vec![0; draft_len],
        proposed_by_position: vec![0; draft_len],
        batch_rounds: 0,
        sequence_rounds: 0,
        prefix_sum: 0,
    };
    while contexts.iter().any(|row| row.len() < output_len) {
        let active = contexts
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (row.len() < output_len).then_some(index))
            .collect::<Vec<_>>();
        let remaining = active
            .iter()
            .map(|index| output_len - contexts[*index].len())
            .min()
            .unwrap();
        let width = draft_len.min(remaining);
        result.batch_rounds += 1;
        result.sequence_rounds += active.len();
        let active_contexts = active
            .iter()
            .map(|index| contexts[*index].clone())
            .collect::<Vec<_>>();
        result.prefix_sum += active_contexts.iter().map(Vec::len).sum::<usize>();

        let (draft_prompts, prompt_lengths) = if ncm_drafter {
            (vec![Vec::new(); active.len()], None)
        } else {
            let width = active_contexts
                .iter()
                .map(Vec::len)
                .max()
                .unwrap_or(0)
                .max(1);
            let padded = active_contexts
                .iter()
                .map(|row| {
                    let mut row = row.clone();
                    row.resize(width, 0);
                    row
                })
                .collect::<Vec<_>>();
            let lengths = active_contexts.iter().map(Vec::len).collect::<Vec<_>>();
            (padded, Some(lengths))
        };
        let draft_seed =
            benchmark_seed(seed, 0, prompts.len(), draft_len, 0, result.batch_rounds, 1);
        let generated = if let Some(prompt_lengths) = prompt_lengths {
            engine
                .generate_ragged(&GenerateRaggedRequest {
                    engine_id: drafter_id.into(),
                    prompts: draft_prompts,
                    prompt_lengths,
                    max_new_tokens: width,
                    temperature: 1.,
                    top_k: 0,
                    seed: draft_seed,
                })
                .unwrap()
        } else {
            engine
                .generate(&GenerateRequest {
                    engine_id: drafter_id.into(),
                    prompts: draft_prompts,
                    max_new_tokens: width,
                    temperature: 1.,
                    top_k: 0,
                    seed: draft_seed,
                })
                .unwrap()
        };
        let prefix_width = active_contexts.iter().map(Vec::len).max().unwrap_or(0);
        let prefixes = active_contexts
            .iter()
            .map(|row| {
                let mut row = row.clone();
                row.resize(prefix_width, 0);
                row
            })
            .collect::<Vec<_>>();
        let prefix_lengths = active_contexts.iter().map(Vec::len).collect::<Vec<_>>();
        let verified = engine
            .speculative_verify_ragged(&SpeculativeVerifyRaggedRequest {
                engine_id: target_id.into(),
                prefixes,
                prefix_lengths,
                draft_tokens: generated.tokens.clone(),
                proposal_probs: generated.probabilities,
                temperature: 1.,
                top_k: 0,
                seed: benchmark_seed(seed, 0, prompts.len(), draft_len, 0, result.batch_rounds, 2),
            })
            .unwrap();
        for (local, global) in active.iter().enumerate() {
            let accepted = verified.accepted_lengths[local];
            result.accepted += accepted;
            result.proposed += generated.tokens[local].len();
            for position in 0..generated.tokens[local].len() {
                result.proposed_by_position[position] += 1;
                if position < accepted {
                    result.accepted_by_position[position] += 1;
                }
            }
            contexts[*global].extend_from_slice(&generated.tokens[local][..accepted]);
            if contexts[*global].len() < output_len {
                contexts[*global].push(verified.tokens[local][width]);
            }
        }
    }
    result.tokens = contexts
        .into_iter()
        .map(|row| row.into_iter().map(|token| token as u32).collect())
        .collect();
    result
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
            prior_only: false,
            sequence_distillation: false,
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
fn generate_sequence_rpc_is_seeded_and_returns_typed_scores() {
    use ncm::rpc::{
        CausalArConfig, CausalArInitRequest, GenerateSequenceRequest, GenerationConfig,
    };
    let root = repository_root();
    let mut engine = EngineClient::start(&root).unwrap();
    let config: CausalArConfig = serde_json::from_value(json!({
        "transformer":{"width":8,"heads":2,"layers":1,"mlp_ratio":2},
        "max_length":12
    }))
    .unwrap();
    let model = engine
        .causal_ar_init(&CausalArInitRequest {
            config,
            seed: 7,
            tokens: vec![vec![0]],
            mask: None,
        })
        .unwrap();
    let request = GenerateSequenceRequest {
        engine_id: model.engine_id,
        prompts: vec![vec![65, 66]],
        config: GenerationConfig {
            max_length: 4,
            temperature: 0.8,
            top_k: 12,
            top_p: 0.9,
            seed: 42,
        },
    };
    let first = engine.generate_sequence(&request).unwrap();
    let second = engine.generate_sequence(&request).unwrap();
    assert_eq!(first.tokens, second.tokens);
    assert_eq!(first.log_probabilities, second.log_probabilities);
    assert_eq!(first.tokens[0].len(), 4);
    assert_eq!(first.log_probabilities[0].len(), 4);
    assert_eq!(first.metrics.token_count, 4);
    assert!(first.metrics.mean_log_probability.is_finite());
    let mut full_support = request.clone();
    full_support.config.top_p = 1.0;
    let unfiltered = engine.generate_sequence(&full_support).unwrap();
    assert!(
        unfiltered.log_probabilities[0]
            .iter()
            .all(|x| x.is_finite())
    );
}

#[test]
fn codec_rpc_and_prior_training_keep_the_codec_frozen() {
    let root = repository_root();
    let dir = env::temp_dir().join(format!("ncm codec rpc {}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut engine = EngineClient::start(&root).expect("start JAX worker");
    let codec = engine
        .init_model(&model_request())
        .expect("initialize codec");
    engine
        .train_step(&TrainStepRequest {
            engine_id: codec.engine_id.clone(),
            batch: batch(),
            seed: 101,
            phase: 0,
            prior_only: false,
            sequence_distillation: false,
        })
        .expect("pretrain shared codec");
    let codec_path = dir.join("codec.msgpack");
    let saved_codec = engine
        .save_codec(&CheckpointRequest {
            engine_id: codec.engine_id,
            path: codec_path.to_string_lossy().into_owned(),
        })
        .expect("save codec");
    let codec_fingerprint = saved_codec.codec_sha256.expect("codec fingerprint");

    let prior = engine
        .init_model(&model_request())
        .expect("initialize prior");
    engine
        .load_codec(&CheckpointRequest {
            engine_id: prior.engine_id.clone(),
            path: codec_path.to_string_lossy().into_owned(),
        })
        .expect("restore frozen codec");
    let before = dir.join("before.msgpack");
    let frozen_codec = engine
        .save_codec(&CheckpointRequest {
            engine_id: prior.engine_id.clone(),
            path: before.to_string_lossy().into_owned(),
        })
        .unwrap();
    assert_eq!(
        frozen_codec.codec_sha256.as_deref(),
        Some(codec_fingerprint.as_str())
    );
    engine
        .train_step(&TrainStepRequest {
            engine_id: prior.engine_id.clone(),
            batch: batch(),
            seed: 102,
            phase: 0,
            prior_only: true,
            sequence_distillation: false,
        })
        .expect("train prior while freezing codec");
    let after_prior = dir.join("after-prior.msgpack");
    let after_prior_state = engine
        .save_codec(&CheckpointRequest {
            engine_id: prior.engine_id.clone(),
            path: after_prior.to_string_lossy().into_owned(),
        })
        .unwrap();
    assert_eq!(
        after_prior_state.codec_sha256.as_deref(),
        Some(codec_fingerprint.as_str())
    );

    let distilled = engine
        .train_step(&TrainStepRequest {
            engine_id: prior.engine_id.clone(),
            batch: batch(),
            seed: 103,
            phase: 0,
            prior_only: false,
            sequence_distillation: true,
        })
        .expect("train sequence-distilled prior and decoder");
    assert!(distilled.loss.is_finite() && distilled.grad_norm.is_finite());
    let after_sequence = dir.join("after-sequence.msgpack");
    let after_sequence_state = engine
        .save_codec(&CheckpointRequest {
            engine_id: prior.engine_id,
            path: after_sequence.to_string_lossy().into_owned(),
        })
        .unwrap();
    assert_ne!(
        after_sequence_state.codec_sha256.as_deref(),
        Some(codec_fingerprint.as_str())
    );
    std::fs::remove_dir_all(dir).unwrap();
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
    let tokens_only = engine
        .generate_tokens(&GenerateRequest {
            engine_id: model.engine_id.clone(),
            prompts: vec![vec![1, 2], vec![3, 4]],
            max_new_tokens: 3,
            temperature: 0.8,
            top_k: 5,
            seed: 10,
        })
        .expect("generate tokens without probability payload");
    assert_eq!(tokens_only.tokens, generated.tokens);
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

#[test]
fn rust_drives_ragged_ar_generation_and_verification_including_empty_prefixes() {
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
            seed: 31,
            tokens: vec![vec![1]],
            mask: None,
        })
        .expect("initialize ragged AR");
    let prompts = vec![vec![0, 0], vec![4, 5]];
    let generated = engine
        .generate_ragged(&GenerateRaggedRequest {
            engine_id: model.engine_id.clone(),
            prompts: prompts.clone(),
            prompt_lengths: vec![0, 2],
            max_new_tokens: 2,
            temperature: 1.,
            top_k: 0,
            seed: 32,
        })
        .expect("generate with empty and nonempty prefixes");
    assert_eq!(generated.tokens.len(), 2);
    assert!(generated.tokens.iter().all(|row| row.len() == 2));
    assert_eq!(generated.probabilities.len(), 2);

    let verified = engine
        .speculative_verify_ragged(&SpeculativeVerifyRaggedRequest {
            engine_id: model.engine_id,
            prefixes: prompts,
            prefix_lengths: vec![0, 2],
            draft_tokens: generated.tokens,
            proposal_probs: generated.probabilities,
            temperature: 1.,
            top_k: 0,
            seed: 33,
        })
        .expect("verify proposals with ragged prefixes");
    assert_eq!(verified.accepted_lengths.len(), 2);
    assert!(
        verified
            .accepted_lengths
            .iter()
            .all(|accepted| *accepted <= 2)
    );
    assert!(verified.tokens.iter().all(|row| row.len() == 3));
}

#[test]
fn atomic_speculation_matches_legacy_round_seeds_tokens_and_accounting() {
    let mut engine = EngineClient::start(&repository_root()).expect("start JAX worker");
    let target = engine
        .causal_ar_init(&CausalArInitRequest {
            config: CausalArConfig {
                transformer: TransformerConfig {
                    width: 4,
                    heads: 2,
                    layers: 1,
                    mlp_ratio: 2,
                    dropout: 0.,
                },
                max_length: 8,
                learning_rate: 1e-3,
            },
            seed: 701,
            tokens: vec![vec![1]],
            mask: None,
        })
        .expect("initialize target AR");
    let ar_drafter = engine
        .causal_ar_init(&CausalArInitRequest {
            config: CausalArConfig {
                transformer: TransformerConfig {
                    width: 4,
                    heads: 2,
                    layers: 1,
                    mlp_ratio: 2,
                    dropout: 0.,
                },
                max_length: 8,
                learning_rate: 1e-3,
            },
            seed: 702,
            tokens: vec![vec![1]],
            mask: None,
        })
        .expect("initialize AR drafter");
    let ncm_drafter = engine
        .init_model(&model_request())
        .expect("initialize NCM drafter");
    let prompts = vec![vec![], vec![]];
    let seed = 0x1234_abcd;

    for (drafter_id, is_ncm) in [
        (ncm_drafter.engine_id.as_str(), true),
        (ar_drafter.engine_id.as_str(), false),
    ] {
        let atomic = engine
            .generate_and_verify_atomic(&GenerateAndVerifyAtomicRequest {
                target_engine_id: target.engine_id.clone(),
                drafter_engine_id: drafter_id.into(),
                prompts: prompts.clone(),
                prompt_len: 0,
                output_len: 2,
                draft_len: 1,
                temperature: 1.,
                top_k: 0,
                seed,
            })
            .expect("atomic speculative observation");
        assert!(atomic.engine_elapsed_ms.is_finite() && atomic.engine_elapsed_ms >= 0.);
        assert!((0. ..=1.).contains(&atomic.acceptance_rate));
        let legacy = legacy_speculation(
            &mut engine,
            &prompts,
            LegacySpeculationConfig {
                target_id: &target.engine_id,
                drafter_id,
                ncm_drafter: is_ncm,
                output_len: 2,
                draft_len: 1,
                seed,
            },
        );
        assert_eq!(atomic.tokens, legacy.tokens);
        assert_eq!(atomic.accepted_tokens, legacy.accepted);
        assert_eq!(atomic.proposed_tokens, legacy.proposed);
        assert_eq!(atomic.accepted_by_position, legacy.accepted_by_position);
        assert_eq!(atomic.proposed_by_position, legacy.proposed_by_position);
        assert_eq!(atomic.batch_rounds, legacy.batch_rounds);
        assert_eq!(atomic.sequence_rounds, legacy.sequence_rounds);
        assert_eq!(atomic.prefix_sum, legacy.prefix_sum);
    }
}
