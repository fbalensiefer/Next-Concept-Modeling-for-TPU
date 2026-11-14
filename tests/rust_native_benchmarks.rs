use ncm::model_benchmarks::{DecisiveBenchmarkRunner, DrafterComparisonRunner};
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn fixture() -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ncm-native-bench-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&d).unwrap();
    d
}
fn config(path: &std::path::Path) {
    fs::write(path,r#"{"transformer":{"width":8,"heads":2,"layers":1,"mlp_ratio":2,"dropout":0.0},"max_length":8,"learning_rate":0.001}"#).unwrap();
}

#[test]
fn native_decisive_runner_reports_batched_token_loss_statistics() {
    let d = fixture();
    let model = d.join("model.json");
    config(&model);
    let train = d.join("train.txt");
    let held = d.join("held.txt");
    fs::write(&train, b"abcdefghABCDEFGH").unwrap();
    fs::write(&held, b"abcdefgh").unwrap();
    let output = d.join("decisive");
    DecisiveBenchmarkRunner::run(&root(), &model, &held, Some(&train), &output, 1, 2, 1, 7)
        .unwrap();
    let report: Value =
        serde_json::from_slice(&fs::read(output.join("results.json")).unwrap()).unwrap();
    assert_eq!(report["heldout"]["metrics"]["tokens"], 7);
    assert!(
        report["heldout"]["metrics"]["mean_nll_nats"]
            .as_f64()
            .unwrap()
            .is_finite()
    );
    assert!(output.join("summary.md").is_file());
    let _ = fs::remove_dir_all(d);
}

#[test]
fn native_drafter_runner_verifies_jsonl_prompt_suites_and_matches_archived_csv_columns() {
    let d = fixture();
    let target = d.join("target.json");
    let drafter = d.join("draft.json");
    config(&target);
    config(&drafter);
    let prompts = d.join("prompts.jsonl");
    fs::write(
        &prompts,
        "{\"id\":\"one\",\"prompt\":\"hi\"}\n{\"id\":\"two\",\"prompt\":\"ok\"}\n",
    )
    .unwrap();
    let output = d.join("drafter");
    DrafterComparisonRunner::run(
        &root(),
        &target,
        &[("small_ar".into(), drafter)],
        &prompts,
        &output,
        3,
        2,
        0.,
        0,
        11,
    )
    .unwrap();
    let report: Value =
        serde_json::from_slice(&fs::read(output.join("results.json")).unwrap()).unwrap();
    assert_eq!(report["rows"].as_array().unwrap().len(), 4);
    assert_eq!(report["aggregates"].as_array().unwrap().len(), 1);
    assert_eq!(report["aggregates"][0]["prompt_count"], 2);
    assert!(report["paired_throughput_vs_teacher"]["small_ar"]["ci95"].is_array());
    assert!(output.join("paired_statistics.json").is_file());
    let csv = fs::read_to_string(output.join("drafter_comparison.csv")).unwrap();
    let header = csv.lines().next().unwrap();
    let archived =
        fs::read_to_string(root().join("reports/validation-controls/smoke/drafter_comparison.csv"))
            .unwrap();
    let old = archived.lines().next().unwrap();
    for field in [
        "drafter",
        "draft_len",
        "prompt_len",
        "batch",
        "acceptance",
        "tokens_per_pass",
        "median_ms",
        "speedup_vs_cached_teacher",
        "per_position_acceptance",
        "measured_rounds",
    ] {
        assert!(
            header.split(',').any(|x| x == field),
            "native report lacks {field}"
        );
        assert!(old.split(',').any(|x| x == field), "baseline lacks {field}");
    }
    assert!(output.join("summary.md").is_file());
    let _ = fs::remove_dir_all(d);
}
