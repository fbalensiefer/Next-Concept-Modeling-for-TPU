//! Rust-native summaries for committed research workload results.
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};

fn read_rows(path: &Path) -> Result<Vec<Value>, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid {}: {e}", path.display()))
}
fn num(row: &Value, path: &[&str]) -> Result<f64, String> {
    let value = path.iter().try_fold(row, |v, key| {
        v.get(*key).ok_or_else(|| format!("missing field {key}"))
    })?;
    value
        .as_f64()
        .ok_or_else(|| format!("non-numeric field {path:?}"))
}
fn mean(xs: &[f64]) -> Result<f64, String> {
    if xs.is_empty() {
        return Err("cannot average an empty group".into());
    }
    Ok(xs.iter().sum::<f64>() / xs.len() as f64)
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    text.push('\n');
    fs::write(path, text).map_err(|e| e.to_string())
}
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let rank = q * (sorted.len() - 1) as f64;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

/// Reproduce the committed fresh-domain analysis using only its JSON evidence.
pub fn summarize_domain_validation(root: &Path, from_reports: bool) -> Result<Value, String> {
    let base = root.join(if from_reports {
        "reports/fresh-domain-validation"
    } else {
        "runs/fresh-domain-validation"
    });
    let output = root.join("reports/fresh-domain-validation");
    let rows = read_rows(&base.join("results.json"))?;
    if rows.is_empty() {
        return Err("domain results are empty".into());
    }
    let domains: Vec<String> = rows
        .iter()
        .filter_map(|r| r["domain"].as_str().map(str::to_owned))
        .fold(Vec::new(), |mut acc, d| {
            if !acc.contains(&d) {
                acc.push(d);
            }
            acc
        });
    let arms: Vec<String> = rows
        .iter()
        .filter_map(|r| r["arm"].as_str().map(str::to_owned))
        .fold(Vec::new(), |mut acc, d| {
            if !acc.contains(&d) {
                acc.push(d);
            }
            acc
        });
    let treatments: Vec<String> = rows
        .iter()
        .filter_map(|r| r["treatment"].as_str().map(str::to_owned))
        .fold(Vec::new(), |mut acc, d| {
            if !acc.contains(&d) {
                acc.push(d);
            }
            acc
        });
    let expected = domains.len() * 5 * arms.len() * treatments.len();
    let keys: BTreeSet<String> = rows
        .iter()
        .map(|r| {
            format!(
                "{}|{}|{}|{}",
                r["domain"], r["seed"], r["arm"], r["treatment"]
            )
        })
        .collect();
    if rows.len() != expected || keys.len() != expected {
        return Err(format!(
            "incomplete or duplicate design: expected {expected} model rows, found {}",
            rows.len()
        ));
    }
    for domain in &domains {
        for seed in 31..36 {
            let hashes: BTreeSet<String> = rows
                .iter()
                .filter(|r| r["domain"].as_str() == Some(domain.as_str()) && r["seed"] == seed)
                .map(|r| r["initial_state_sha256"].as_str().unwrap_or("").to_owned())
                .collect();
            if hashes.len() != 1 {
                return Err(format!(
                    "initial-state hashes do not match for {domain}, seed {seed}"
                ));
            }
        }
    }
    let mut summary = serde_json::Map::new();
    let mut by_arm = serde_json::Map::new();
    let mut paired = serde_json::Map::new();
    for domain in &domains {
        let group = |pred: &dyn Fn(&Value) -> bool| {
            rows.iter()
                .filter(|r| pred(r) && r["domain"].as_str() == Some(domain.as_str()))
                .collect::<Vec<_>>()
        };
        let mut treatments_json = serde_json::Map::new();
        for treatment in &treatments {
            let selected = group(&|r| r["treatment"].as_str() == Some(treatment.as_str()));
            let metric = |name: &str| {
                selected
                    .iter()
                    .map(|r| num(r, &["after", name]))
                    .collect::<Result<Vec<_>, _>>()
            };
            let reconstruction = metric("sequence_reconstruction")?;
            let missing = metric("factorized_missing_teacher_mass")?;
            let timings = selected
                .iter()
                .filter_map(|r| r["codec_seconds"].as_f64())
                .collect::<Vec<_>>();
            if timings.is_empty() {
                return Err(format!("no codec timing values for {domain}/{treatment}"));
            }
            let mut sorted_timings = timings.clone();
            sorted_timings.sort_by(f64::total_cmp);
            let median = if sorted_timings.len() % 2 == 0 {
                (sorted_timings[sorted_timings.len() / 2 - 1]
                    + sorted_timings[sorted_timings.len() / 2])
                    / 2.
            } else {
                sorted_timings[sorted_timings.len() / 2]
            };
            let means = [
                "sequence_reconstruction",
                "factorized_oracle_tv",
                "factorized_oracle_kl_nats",
                "factorized_missing_teacher_mass",
                "factorized_outside_mass",
                "max_sequence_reconstruction_from_codes",
                "coarse_perplexity",
            ]
            .iter()
            .map(|m| Ok(((*m).to_owned(), json!(mean(&metric(m)?)?))))
            .collect::<Result<serde_json::Map<_, _>, String>>()?;
            let codec_flops: BTreeSet<u64> = selected
                .iter()
                .filter_map(|r| r["codec_step_flops"].as_u64())
                .collect();
            treatments_json.insert(treatment.clone(), json!({
                "models": selected.len(),
                "fidelity_failures": selected.iter().filter(|r| r["failures"].as_array().is_some_and(|a| !a.is_empty())).count(),
                "reconstruction_failures": selected.iter().filter(|r| r["failures"].as_array().is_some_and(|a| a.iter().any(|x| x=="reconstruction"))).count(),
                "coverage_failures": selected.iter().filter(|r| r["failures"].as_array().is_some_and(|a| a.iter().any(|x| x=="missing_mass"))).count(),
                "collision_ceiling_failures": selected.iter().filter(|r| r["failures"].as_array().is_some_and(|a| a.iter().any(|x| x=="collision_ceiling"))).count(),
                "any_collision_models": selected.iter().filter(|r| r["after"]["code_collision_groups"].as_array().is_some_and(|a| !a.is_empty())).count(),
                "final_coarse_collapse": selected.iter().filter(|r| r["after"]["active_coarse_codes"].as_f64().unwrap_or(0.) < 2.).count(),
                "usage_window_failures": selected.iter().filter(|r| r["collapsed_fraction"].as_f64().unwrap_or(0.) > 0.05).count(),
                "mean": means,
                "min_reconstruction": reconstruction.iter().copied().fold(f64::INFINITY,f64::min),
                "max_missing_mass": missing.iter().copied().fold(f64::NEG_INFINITY,f64::max),
                "median_codec_seconds": median,
                "missing_codec_timings": selected.len()-timings.len(),
                "codec_step_flops": codec_flops
            }));
        }
        summary.insert(domain.clone(), Value::Object(treatments_json));
        let mut arms_json = serde_json::Map::new();
        for arm in &arms {
            let mut arm_treatments = serde_json::Map::new();
            for treatment in &treatments {
                let selected = group(&|r| {
                    r["arm"].as_str() == Some(arm.as_str())
                        && r["treatment"].as_str() == Some(treatment.as_str())
                });
                let tv = selected
                    .iter()
                    .map(|r| num(r, &["after", "factorized_oracle_tv"]))
                    .collect::<Result<Vec<_>, _>>()?;
                let recon = selected
                    .iter()
                    .map(|r| num(r, &["after", "sequence_reconstruction"]))
                    .collect::<Result<Vec<_>, _>>()?;
                arm_treatments.insert(treatment.clone(),json!({"fidelity_failures":selected.iter().filter(|r|r["failures"].as_array().is_some_and(|a|!a.is_empty())).count(),"mean_tv":mean(&tv)?,"mean_reconstruction":mean(&recon)?}));
            }
            arms_json.insert(arm.clone(), Value::Object(arm_treatments));
        }
        by_arm.insert(domain.clone(), Value::Object(arms_json));
        let mut domain_paired = serde_json::Map::new();
        for candidate in ["ema_recovery", "fixed_binary"] {
            let mut differences = Vec::new();
            for seed in 31..36 {
                let seed_mean = |t: &str| -> Result<f64, String> {
                    let xs = rows
                        .iter()
                        .filter(|r| {
                            r["domain"].as_str() == Some(domain.as_str())
                                && r["seed"] == seed
                                && r["treatment"].as_str() == Some(t)
                        })
                        .map(|r| num(r, &["after", "factorized_oracle_tv"]))
                        .collect::<Result<Vec<_>, _>>()?;
                    mean(&xs)
                };
                differences.push(seed_mean(candidate)? - seed_mean("reference")?);
            }
            let avg = mean(&differences)?;
            let mut boot = Vec::with_capacity(3125);
            for a in 0..5 {
                for b in 0..5 {
                    for c in 0..5 {
                        for d in 0..5 {
                            for e in 0..5 {
                                boot.push(
                                    (differences[a]
                                        + differences[b]
                                        + differences[c]
                                        + differences[d]
                                        + differences[e])
                                        / 5.,
                                );
                            }
                        }
                    }
                }
            }
            boot.sort_by(f64::total_cmp);
            domain_paired.insert(candidate.into(),json!({"seed_mean_tv_differences":differences,"mean":avg,"wins":differences.iter().filter(|x|**x<0.).count(),"exact_seed_bootstrap_percentile_95":[percentile(&boot,0.025),percentile(&boot,0.975)],"descriptive_mean_within_tolerance":avg<=0.02}));
        }
        paired.insert(domain.clone(), Value::Object(domain_paired));
    }
    let result = json!({"summary":summary,"by_objective_arm":by_arm,"paired_vs_reference":paired,"bootstrap_note":"All 5^5 resamples of five seed-level, objective-arm-averaged differences. Descriptive percentile intervals; no multiplicity-adjusted noninferiority claim.","all_initial_hashes_matched":true,"complete_models":rows.len()});
    if !from_reports {
        for name in [
            "protocol.json",
            "results.json",
            "teachers.json",
            "source_diagnostics.json",
            "source_baselines.json",
            "numeric_incident.json",
            "continuation.json",
        ] {
            let src = base.join(name);
            if src.exists() {
                fs::copy(&src, output.join(name)).map_err(|e| e.to_string())?;
            } else if ["protocol.json", "results.json", "teachers.json"].contains(&name) {
                return Err(format!("required input missing: {}", src.display()));
            }
        }
    }
    write_json(&output.join("analysis.json"), &result)?;
    Ok(result)
}

/// Recompute the dependency-prior aggregate from the 480 saved fits.
pub fn summarize_dependency_results(root: &Path) -> Result<Value, String> {
    let folder = root.join("reports/dependency-validation");
    let rows = read_rows(&folder.join("results.json"))?;
    if rows.len() != 480 {
        return Err(format!("expected 480 dependency fits, got {}", rows.len()));
    }
    let models = [
        "independent",
        "adjacent_pairs",
        "conditional_tree",
        "joint_fine",
    ];
    let domains: Vec<String> = rows
        .iter()
        .filter_map(|r| r["domain"].as_str().map(str::to_owned))
        .fold(Vec::new(), |mut a, d| {
            if !a.contains(&d) {
                a.push(d)
            }
            a
        });
    let base: Vec<&Value> = rows
        .iter()
        .filter(|r| r["model"] == "independent")
        .collect();
    let mut summary = serde_json::Map::new();
    for domain in &domains {
        let mut models_json = serde_json::Map::new();
        for model in models {
            let all: Vec<&Value> = rows
                .iter()
                .filter(|r| r["domain"].as_str() == Some(domain.as_str()) && r["model"] == model)
                .collect();
            let safe: Vec<&Value> = all
                .iter()
                .copied()
                .filter(|r| r["codec_eligible"].as_bool() == Some(true))
                .collect();
            let vals = |name: &str| -> Result<Vec<f64>, String> {
                safe.iter().map(|r| num(r, &[name])).collect()
            };
            let rounds = safe
                .iter()
                .map(|r| r["complexity"]["serial_rounds"].as_u64())
                .collect::<Option<Vec<_>>>()
                .ok_or("missing serial rounds")?;
            if rounds.is_empty() {
                return Err(format!("no eligible rows for {domain}/{model}"));
            }
            models_json.insert(model.into(),json!({"codecs":all.len(),"eligible":safe.len(),"all_mean_tv":mean(&all.iter().map(|r|num(r,&["output_tv"])).collect::<Result<Vec<_>,_>>()?)?,"eligible_mean_tv":mean(&vals("output_tv")?)?,"eligible_mean_validation_nll":mean(&vals("validation_code_nll")?)?,"min_rounds":rounds.iter().min(),"max_rounds":rounds.iter().max()}));
        }
        summary.insert(domain.clone(), Value::Object(models_json));
    }
    let misses:Vec<Value>=base.iter().filter(|r|r["codec_eligible"]==true&&!r["population_codec_failures"].as_array().unwrap_or(&vec![]).is_empty()).map(|r|json!({"domain":r["domain"],"seed":r["seed"],"arm":r["arm"],"validation_sequence_accuracy":r["validation_sequence_accuracy"],"population_failures":r["population_codec_failures"]})).collect();
    let result = json!({"summary":summary,"eligible":base.iter().filter(|r|r["codec_eligible"]==true).count(),"rejected":base.iter().filter(|r|r["codec_eligible"]!=true).count(),"population_gate_misses":misses});
    write_json(&folder.join("analysis.json"), &result)?;
    Ok(result)
}

fn source_distribution(name: &str, length: usize) -> (Vec<Vec<u8>>, Vec<f64>) {
    let count = 1usize << length;
    let words = (0..count)
        .map(|n| {
            (0..length)
                .map(|i| ((n >> (length - i - 1)) & 1) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let uniform = 1. / count as f64;
    let probabilities = words
        .iter()
        .enumerate()
        .map(|(index, word)| match name {
            "original_mixture" => {
                uniform * 0.25
                    + if index == 0 || index + 1 == count {
                        0.375
                    } else {
                        0.
                    }
            }
            "biased_independent" => word
                .iter()
                .map(|b| if *b == 1 { 0.2 } else { 0.8 })
                .product(),
            "persistent_markov" => {
                0.5 * word
                    .windows(2)
                    .map(|w| if w[0] == w[1] { 0.85 } else { 0.15 })
                    .product::<f64>()
            }
            "noisy_parity" => {
                uniform * 0.1
                    + if word.iter().map(|x| *x as usize).sum::<usize>() % 2 == 0 {
                        0.9 / (count / 2) as f64
                    } else {
                        0.
                    }
            }
            "skewed_modes" => {
                uniform * 0.1
                    + if index == 0 {
                        0.8
                    } else if index + 1 == count {
                        0.1
                    } else {
                        0.
                    }
            }
            "alternating_modes" => {
                let is_first = word.iter().enumerate().all(|(i, b)| *b == (i % 2) as u8);
                let is_second = word.iter().enumerate().all(|(i, b)| *b != (i % 2) as u8);
                uniform * 0.2 + if is_first || is_second { 0.4 } else { 0. }
            }
            _ => unreachable!("known domain name"),
        })
        .collect();
    (words, probabilities)
}

fn scheduled_distribution(words: &[Vec<u8>], p: &[f64], levels: &[Vec<usize>]) -> Vec<f64> {
    let mut q = vec![1.; p.len()];
    let mut past: Vec<usize> = Vec::new();
    for level in levels {
        for (i, word) in words.iter().enumerate() {
            let same_context = |j: usize| {
                past.iter()
                    .all(|position| words[j][*position] == word[*position])
            };
            let denominator: f64 = (0..p.len())
                .filter(|j| same_context(*j))
                .map(|j| p[j])
                .sum();
            for position in level {
                let numerator: f64 = (0..p.len())
                    .filter(|j| same_context(*j) && words[*j][*position] == word[*position])
                    .map(|j| p[j])
                    .sum();
                q[i] *= numerator / denominator;
            }
        }
        past.extend(level);
    }
    q
}

fn midpoint_levels(length: usize) -> Vec<Vec<usize>> {
    let mut levels = Vec::new();
    let mut intervals = vec![(0, length)];
    while !intervals.is_empty() {
        let mut level = Vec::new();
        let mut next = Vec::new();
        for (left, right) in intervals {
            let middle = (left + right - 1) / 2;
            level.push(middle);
            if left < middle {
                next.push((left, middle));
            }
            if middle + 1 < right {
                next.push((middle + 1, right));
            }
        }
        levels.push(level);
        intervals = next;
    }
    levels
}

/// Enumerate exact source-law controls and known invertible binary innovations.
pub fn domain_source_baselines(root: &Path) -> Result<Value, String> {
    let domains = [
        ("original_mixture", 4),
        ("biased_independent", 4),
        ("persistent_markov", 8),
        ("noisy_parity", 8),
        ("skewed_modes", 8),
        ("alternating_modes", 8),
    ];
    let mut result = serde_json::Map::new();
    for (name, length) in domains {
        let (words, p) = source_distribution(name, length);
        let mut schedules = serde_json::Map::new();
        let mut levels_by_name = vec![("independent", vec![(0..length).collect::<Vec<_>>()])];
        if name == "persistent_markov" {
            levels_by_name.push(("midpoint_bridge", midpoint_levels(length)));
        }
        if name == "noisy_parity" {
            levels_by_name.push((
                "last_bit_conditional",
                vec![(0..length - 1).collect(), vec![length - 1]],
            ));
        }
        for (label, levels) in levels_by_name {
            let q = scheduled_distribution(&words, &p, &levels);
            let tv = p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.;
            let kl = p.iter().zip(&q).map(|(a, b)| a * (a / b).ln()).sum::<f64>();
            schedules.insert(
                label.into(),
                json!({"levels":levels,"serial_rounds":levels.len(),"tv":tv,"kl_nats":kl}),
            );
        }
        let entropy_bits = -p.iter().map(|x| x * x.log2()).sum::<f64>();
        let mut row =
            json!({"population":p.len(),"entropy_bits":entropy_bits,"schedules":schedules});
        if name == "persistent_markov" || name == "noisy_parity" {
            let transformed = words
                .iter()
                .map(|word| {
                    if name == "persistent_markov" {
                        let mut out = vec![word[0]];
                        out.extend(word.windows(2).map(|w| w[0] ^ w[1]));
                        out
                    } else {
                        let mut out = word[..length - 1].to_vec();
                        out.push((word.iter().map(|x| *x as usize).sum::<usize>() % 2) as u8);
                        out
                    }
                })
                .collect::<Vec<_>>();
            let marginal = (0..length)
                .map(|pos| {
                    p.iter()
                        .zip(&transformed)
                        .filter(|(_, w)| w[pos] == 1)
                        .map(|(weight, _)| weight)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let product = transformed
                .iter()
                .map(|word| {
                    (0..length)
                        .map(|pos| {
                            if word[pos] == 1 {
                                marginal[pos]
                            } else {
                                1. - marginal[pos]
                            }
                        })
                        .product::<f64>()
                })
                .collect::<Vec<_>>();
            let invertible = name == "persistent_markov" || name == "noisy_parity";
            let tv = p
                .iter()
                .zip(&product)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 2.;
            let max_error = p
                .iter()
                .zip(&product)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            row["invertible_innovations"] = json!({"marginal_one_probabilities":marginal,"tv":tv,"max_probability_error":max_error,"bijective":invertible});
        }
        result.insert(name.into(), row);
    }
    let value = Value::Object(result);
    write_json(
        &root.join("reports/invertible-baselines/results.json"),
        &value,
    )?;
    Ok(value)
}
