use ncm::rpc::{
    CausalArConfig, CausalArEvaluateRequest, CausalArInitRequest, CausalArTrainStepRequest,
    CheckpointRequest, EngineClient, TransformerConfig,
};
use ncm::sweep::{DrafterTimingOptions, benchmark_drafters};
use serde_json::Value;
use std::{fs, path::PathBuf, process::Command};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn native_sweep_trains_saves_scores_and_audits_scheduled_checkpoints() {
    let root = root();
    let dir = std::env::temp_dir().join(format!("ncm sweep integration {}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("train.txt"), b"abcdabcdabcdabcd").unwrap();
    fs::write(dir.join("valid.txt"), b"dcbaabcdabcdabcd").unwrap();
    fs::write(
        dir.join("sweep.json"),
        format!(
            r#"{{
              "train_file":"{}", "validation_file":"{}", "sequence_length":4, "max_prompt_len":0,
              "seeds":[200], "prompt_lengths":[0], "steps":2, "checkpoint_steps":[1,2], "batch_size":1,
              "arms":[{{"name":"small-ncm", "kind":"ncm", "config":{{
                "rvq":{{"lengths":[1,2,4],"codebook_size":2,"dim":2}},
                "transformer":{{"width":4,"heads":2,"layers":1,"mlp_ratio":2}},
                "learning_rate":0.001
              }}}}]
            }}"#,
            dir.join("train.txt").display(),
            dir.join("valid.txt").display()
        ),
    )
    .unwrap();
    let python = std::env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into());
    let call = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_ncm-run"))
            .current_dir(&root)
            .env("NCM_PYTHON", &python)
            .args(args)
            .output()
            .unwrap()
    };
    let result = call(&[
        "train-sweep",
        "--config",
        dir.join("sweep.json").to_str().unwrap(),
        "--output",
        dir.join("run").to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let quality = call(&["quality-eval", "--sweep", dir.join("run").to_str().unwrap()]);
    assert!(
        quality.status.success(),
        "{}",
        String::from_utf8_lossy(&quality.stderr)
    );
    let audit = call(&[
        "checkpoint-audit",
        "--sweep",
        dir.join("run").to_str().unwrap(),
    ]);
    assert!(
        audit.status.success(),
        "{}",
        String::from_utf8_lossy(&audit.stderr)
    );

    let manifest: Value =
        serde_json::from_slice(&fs::read(dir.join("run/sweep.json")).unwrap()).unwrap();
    let quality: Value =
        serde_json::from_slice(&fs::read(dir.join("run/quality_rows.json")).unwrap()).unwrap();
    let audit: Value =
        serde_json::from_slice(&fs::read(dir.join("run/checkpoint_audit.json")).unwrap()).unwrap();
    assert_eq!(manifest["training_rows"].as_array().unwrap().len(), 1);
    assert_eq!(manifest["checkpoints"].as_array().unwrap().len(), 2);
    assert_eq!(quality["row_count"], 1);
    assert_eq!(audit["row_count"], 2);
    assert!(
        quality["rows"][0]["heldout_bpb"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
    assert_eq!(quality["rows"][0]["bpb_kind"], "upper_bound_bpb");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn causal_ar_checkpoints_restore_parameters_optimizer_and_step_over_rpc() {
    let root = root();
    let dir = std::env::temp_dir().join(format!("ncm ar checkpoint {}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ar.msgpack");
    let mut engine = EngineClient::start(&root).unwrap();
    let handle = engine
        .causal_ar_init(&CausalArInitRequest {
            config: CausalArConfig {
                transformer: TransformerConfig {
                    width: 4,
                    heads: 2,
                    layers: 1,
                    mlp_ratio: 2,
                    dropout: 0.0,
                },
                max_length: 4,
                learning_rate: 0.001,
            },
            seed: 200,
            tokens: vec![vec![0]],
            mask: None,
        })
        .unwrap();
    let tokens = vec![vec![1, 2, 3, 4]];
    engine
        .causal_ar_train_step(&CausalArTrainStepRequest {
            engine_id: handle.engine_id.clone(),
            tokens: tokens.clone(),
            mask: None,
            continuation_start: 0,
            teacher_engine_id: None,
            distillation_weight: 0.0,
        })
        .unwrap();
    let checkpoint = CheckpointRequest {
        engine_id: handle.engine_id.clone(),
        path: path.to_string_lossy().into_owned(),
    };
    let saved = engine.save_checkpoint(&checkpoint).unwrap();
    let before = engine
        .causal_ar_evaluate(&CausalArEvaluateRequest {
            engine_id: handle.engine_id.clone(),
            tokens: tokens.clone(),
            mask: None,
            prompt_len: 0,
        })
        .unwrap();
    engine
        .causal_ar_train_step(&CausalArTrainStepRequest {
            engine_id: handle.engine_id.clone(),
            tokens: vec![vec![4, 3, 2, 1]],
            mask: None,
            continuation_start: 0,
            teacher_engine_id: None,
            distillation_weight: 0.0,
        })
        .unwrap();
    let restored = engine.load_checkpoint(&checkpoint).unwrap();
    let after = engine
        .causal_ar_evaluate(&CausalArEvaluateRequest {
            engine_id: handle.engine_id,
            tokens,
            mask: None,
            prompt_len: 0,
        })
        .unwrap();
    assert_eq!(saved.step, 1);
    assert_eq!(restored.step, 1);
    assert_eq!(before.mean_nll, after.mean_nll);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_drafter_timing_writes_typed_reference_and_speculative_rows() {
    let root = root();
    let dir = std::env::temp_dir().join(format!("ncm drafter timing {}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("train.txt"), b"abcdabcdabcdabcdabcdabcd").unwrap();
    fs::write(dir.join("valid.txt"), b"dcbaabcdabcdabcdabcdabcd").unwrap();
    fs::write(
        dir.join("sweep.json"),
        format!(
            r#"{{
              "train_file":"{}", "validation_file":"{}", "sequence_length":4, "max_prompt_len":2,
              "seeds":[200], "prompt_lengths":[0,2], "steps":1, "checkpoint_steps":[1], "batch_size":1,
              "teacher":{{
                "transformer":{{"width":4,"heads":2,"layers":1,"mlp_ratio":2,"dropout":0.0}},
                "max_length":9,"learning_rate":0.001
              }},
              "arms":[
                {{"name":"teacher", "kind":"causal_ar", "config":{{
                  "transformer":{{"width":4,"heads":2,"layers":1,"mlp_ratio":2,"dropout":0.0}},
                  "max_length":9,"learning_rate":0.001
                }}}},
                {{"name":"small-ncm", "kind":"ncm", "config":{{
                  "rvq":{{"lengths":[1,2,4],"codebook_size":2,"dim":2}},
                  "transformer":{{"width":4,"heads":2,"layers":1,"mlp_ratio":2}},
                  "learning_rate":0.001
                }}}}
              ]
            }}"#,
            dir.join("train.txt").display(),
            dir.join("valid.txt").display()
        ),
    )
    .unwrap();
    ncm::sweep::train_sweep(&root, &dir.join("sweep.json"), &dir.join("run")).unwrap();

    let rows = benchmark_drafters(
        &root,
        &dir.join("run"),
        &dir.join("run"),
        "teacher",
        &[("ncm".into(), "small-ncm".into())],
        &DrafterTimingOptions {
            draft_lengths: vec![1],
            batch_sizes: vec![1],
            warmups: 0,
            repetitions: 1,
            temperature: 1.,
            top_k: 0,
        },
        false,
    )
    .unwrap();
    assert_eq!(rows, 4);

    let json: Value =
        serde_json::from_slice(&fs::read(dir.join("run/drafter_comparison.json")).unwrap())
            .unwrap();
    assert_eq!(json["row_count"], 4);
    assert_eq!(json["rows"].as_array().unwrap().len(), 4);
    assert!(json["rows"].as_array().unwrap().iter().all(|row| {
        row["median_ms"].as_f64().is_some_and(f64::is_finite)
            && row["continuation_sha256"]
                .as_str()
                .is_some_and(|hash| hash.len() == 64)
            && row["prompt_bucket_sha256"]
                .as_str()
                .is_some_and(|hash| hash.len() == 64)
            && row["batch"].as_u64() == Some(1)
    }));
    let csv = fs::read_to_string(dir.join("run/drafter_comparison.csv")).unwrap();
    assert_eq!(csv.lines().count(), 5);
    assert!(
        csv.lines()
            .next()
            .unwrap()
            .split(',')
            .any(|field| field == "acceptance_rate")
    );
    assert!(
        csv.lines()
            .any(|line| line.contains(",cached_teacher_reference,"))
    );
    assert!(csv.lines().any(|line| line.contains(",ncm,")));
    fs::remove_dir_all(dir).unwrap();
}
