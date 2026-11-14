use std::{fs, path::PathBuf};

#[test]
fn native_maintenance_summary_matches_committed_python_analysis() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temporary = std::env::temp_dir().join(format!("ncm native report {}", std::process::id()));
    let data = temporary.join("reports/maintenance-ablation");
    let _ = fs::remove_dir_all(&temporary);
    fs::create_dir_all(&data).unwrap();
    fs::copy(
        root.join("reports/maintenance-ablation/results.json"),
        data.join("results.json"),
    )
    .unwrap();

    let actual = ncm::reports::summarize_maintenance(&temporary).unwrap();
    let expected: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("reports/maintenance-ablation/analysis.json")).unwrap(),
    )
    .unwrap();
    for treatment in ["minimal_ema", "ema_recovery", "reference", "fixed_binary"] {
        assert_eq!(
            actual["summary"][treatment]["models"],
            expected["summary"][treatment]["models"]
        );
        assert_eq!(
            actual["summary"][treatment]["monitoring_failures"],
            expected["summary"][treatment]["monitoring_failures"]
        );
        let actual_tv = actual["summary"][treatment]["mean_factorized_oracle_tv"]
            .as_f64()
            .unwrap();
        let expected_tv = expected["summary"][treatment]["mean_factorized_oracle_tv"]
            .as_f64()
            .unwrap();
        assert!(
            (actual_tv - expected_tv).abs() < 1e-12,
            "{treatment}: {actual_tv} != {expected_tv}"
        );
    }
    let _ = fs::remove_dir_all(temporary);
}
