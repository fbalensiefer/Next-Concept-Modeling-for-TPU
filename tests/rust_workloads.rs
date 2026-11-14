use std::{fs, path::PathBuf, process::Command};

#[test]
fn rust_train_command_prepares_data_trains_and_writes_checkpoints() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("ncm rust workload {}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("train.txt"), b"abababababab").unwrap();
    fs::write(dir.join("valid.txt"), b"ababacacacac").unwrap();
    fs::write(
        dir.join("config.json"),
        format!(
            r#"{{"train_file":"{}","validation_file":"{}","lengths":[1,2],"codebook_size":2,"dim":2,"width":4,"heads":2,"layers":1,"mlp_ratio":2,"batch_size":2,"seed":7,"learning_rate":0.001,"steps":1,"checkpoint_every":1}}"#,
            dir.join("train.txt").display(),
            dir.join("valid.txt").display()
        ),
    )
    .unwrap();

    let run = || {
        Command::new(env!("CARGO_BIN_EXE_ncm-run"))
            .current_dir(&root)
            .env(
                "NCM_PYTHON",
                std::env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into()),
            )
            .args([
                "train",
                "--config",
                dir.join("config.json").to_str().unwrap(),
                "--output",
                dir.join("run").to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(dir.join("run/checkpoint.msgpack").is_file());
    let metrics = fs::read_to_string(dir.join("run/metrics.jsonl")).unwrap();
    assert_eq!(metrics.lines().count(), 1);
    let record: serde_json::Value = serde_json::from_str(metrics.lines().next().unwrap()).unwrap();
    assert_eq!(record["validation"]["valid_bytes"], 12);
    assert_eq!(record["validation"]["batches"], 3);

    let resumed = Command::new(env!("CARGO_BIN_EXE_ncm-run"))
        .current_dir(&root)
        .env(
            "NCM_PYTHON",
            std::env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into()),
        )
        .args([
            "train",
            "--config",
            dir.join("config.json").to_str().unwrap(),
            "--output",
            dir.join("run").to_str().unwrap(),
            "--resume",
        ])
        .output()
        .unwrap();
    assert!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert_eq!(
        fs::read_to_string(dir.join("run/metrics.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    fs::remove_dir_all(dir).unwrap();
}
