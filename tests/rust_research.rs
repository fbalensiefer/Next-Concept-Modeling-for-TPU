use serde_json::Value;
use std::{fs, path::PathBuf};

fn temp_root(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ncm {label} {}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn native_domain_summary_matches_archived_protocol_and_statistics() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temporary = temp_root("domain summary");
    let input = temporary.join("reports/fresh-domain-validation");
    fs::create_dir_all(&input).unwrap();
    for name in [
        "results.json",
        "protocol.json",
        "teachers.json",
        "source_diagnostics.json",
        "source_baselines.json",
        "numeric_incident.json",
        "continuation.json",
    ] {
        let source = root.join("reports/fresh-domain-validation").join(name);
        if source.is_file() {
            fs::copy(source, input.join(name)).unwrap();
        }
    }
    let actual = ncm::research::summarize_domain_validation(&temporary, true).unwrap();
    let expected: Value = serde_json::from_slice(
        &fs::read(root.join("reports/fresh-domain-validation/analysis.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(actual["complete_models"], expected["complete_models"]);
    assert_eq!(actual["all_initial_hashes_matched"], true);
    for domain in [
        "original_mixture",
        "biased_independent",
        "persistent_markov",
        "noisy_parity",
        "skewed_modes",
        "alternating_modes",
    ] {
        for treatment in ["ema_recovery", "reference", "fixed_binary"] {
            let observed = &actual["summary"][domain][treatment];
            let archived = &expected["summary"][domain][treatment];
            assert_eq!(observed["models"], archived["models"]);
            assert_eq!(observed["fidelity_failures"], archived["fidelity_failures"]);
            let difference = (observed["mean"]["factorized_oracle_tv"].as_f64().unwrap()
                - archived["mean"]["factorized_oracle_tv"].as_f64().unwrap())
            .abs();
            assert!(difference < 1e-12, "{domain}/{treatment}: {difference}");
        }
        for treatment in ["ema_recovery", "fixed_binary"] {
            assert_eq!(
                actual["paired_vs_reference"][domain][treatment]["wins"],
                expected["paired_vs_reference"][domain][treatment]["wins"]
            );
            for index in 0..2 {
                let difference = (actual["paired_vs_reference"][domain][treatment]
                    ["exact_seed_bootstrap_percentile_95"][index]
                    .as_f64()
                    .unwrap()
                    - expected["paired_vs_reference"][domain][treatment]
                        ["exact_seed_bootstrap_percentile_95"][index]
                        .as_f64()
                        .unwrap())
                .abs();
                assert!(difference < 1e-12, "{domain}/{treatment} CI: {difference}");
            }
        }
    }
    fs::remove_dir_all(temporary).unwrap();
}

#[test]
fn native_dependency_summary_matches_archived_fit_aggregates() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temporary = temp_root("dependency summary");
    let folder = temporary.join("reports/dependency-validation");
    fs::create_dir_all(&folder).unwrap();
    fs::copy(
        root.join("reports/dependency-validation/results.json"),
        folder.join("results.json"),
    )
    .unwrap();
    let actual = ncm::research::summarize_dependency_results(&temporary).unwrap();
    let expected: Value = serde_json::from_slice(
        &fs::read(root.join("reports/dependency-validation/analysis.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(actual["eligible"], expected["eligible"]);
    assert_eq!(actual["rejected"], expected["rejected"]);
    assert_eq!(
        actual["population_gate_misses"],
        expected["population_gate_misses"]
    );
    for domain in [
        "original_mixture",
        "biased_independent",
        "persistent_markov",
        "noisy_parity",
        "skewed_modes",
        "alternating_modes",
    ] {
        for model in [
            "independent",
            "adjacent_pairs",
            "conditional_tree",
            "joint_fine",
        ] {
            let observed = &actual["summary"][domain][model];
            let archived = &expected["summary"][domain][model];
            assert_eq!(observed["codecs"], archived["codecs"]);
            assert_eq!(observed["eligible"], archived["eligible"]);
            for metric in [
                "all_mean_tv",
                "eligible_mean_tv",
                "eligible_mean_validation_nll",
            ] {
                let difference =
                    (observed[metric].as_f64().unwrap() - archived[metric].as_f64().unwrap()).abs();
                assert!(
                    difference < 1e-12,
                    "{domain}/{model}/{metric}: {difference}"
                );
            }
        }
    }
    fs::remove_dir_all(temporary).unwrap();
}

#[test]
fn native_domain_source_oracles_match_archived_enumeration() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temporary = temp_root("source oracles");
    let actual = ncm::research::domain_source_baselines(&temporary).unwrap();
    let expected: Value = serde_json::from_slice(
        &fs::read(root.join("reports/invertible-baselines/results.json")).unwrap(),
    )
    .unwrap();
    for domain in [
        "original_mixture",
        "biased_independent",
        "persistent_markov",
        "noisy_parity",
        "skewed_modes",
        "alternating_modes",
    ] {
        let observed = &actual[domain];
        let archived = &expected[domain];
        assert_eq!(observed["population"], archived["population"]);
        let entropy_difference = (observed["entropy_bits"].as_f64().unwrap()
            - archived["entropy_bits"].as_f64().unwrap())
        .abs();
        assert!(entropy_difference < 1e-12);
        for schedule in ["independent", "midpoint_bridge", "last_bit_conditional"] {
            if archived["schedules"][schedule].is_null() {
                assert!(observed["schedules"][schedule].is_null());
                continue;
            }
            assert_eq!(
                observed["schedules"][schedule]["levels"],
                archived["schedules"][schedule]["levels"]
            );
            for metric in ["tv", "kl_nats"] {
                let difference = (observed["schedules"][schedule][metric].as_f64().unwrap()
                    - archived["schedules"][schedule][metric].as_f64().unwrap())
                .abs();
                assert!(
                    difference < 1e-12,
                    "{domain}/{schedule}/{metric}: {difference}"
                );
            }
        }
        if !archived["invertible_innovations"].is_null() {
            assert_eq!(
                observed["invertible_innovations"]["bijective"],
                archived["invertible_innovations"]["bijective"]
            );
            let tv_difference = (observed["invertible_innovations"]["tv"].as_f64().unwrap()
                - archived["invertible_innovations"]["tv"].as_f64().unwrap())
            .abs();
            assert!(tv_difference < 1e-12);
        }
    }
    fs::remove_dir_all(temporary).unwrap();
}
