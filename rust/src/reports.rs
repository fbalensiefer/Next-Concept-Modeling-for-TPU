//! Rust-native analysis of committed host-side experiment results.
use serde_json::{Value, json};
use std::{fs, path::Path};

fn number(row: &Value, path: &[&str]) -> Result<f64, String> {
    let value = path.iter().try_fold(row, |current, key| {
        current
            .get(*key)
            .ok_or_else(|| format!("missing report field {key}"))
    })?;
    value
        .as_f64()
        .ok_or_else(|| format!("report field {:?} is not numeric", path))
}
fn mean(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot summarize an empty group".into());
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}
fn rows_for<'a>(rows: &'a [Value], treatment: &str) -> Vec<&'a Value> {
    rows.iter()
        .filter(|row| {
            row["treatment"] == treatment
                && (treatment != "ema_recovery" || row["study"] == "maintenance-ablation")
        })
        .collect()
}
fn summarize_treatment(rows: &[&Value]) -> Result<Value, String> {
    let after = |row: &&Value, metric| number(row, &["after", metric]);
    let recon = rows
        .iter()
        .map(|r| after(r, "sequence_reconstruction"))
        .collect::<Result<Vec<_>, _>>()?;
    let tv = rows
        .iter()
        .map(|r| after(r, "factorized_oracle_tv"))
        .collect::<Result<Vec<_>, _>>()?;
    let kl = rows
        .iter()
        .map(|r| after(r, "factorized_oracle_kl_nats"))
        .collect::<Result<Vec<_>, _>>()?;
    let active = rows
        .iter()
        .map(|r| number(r, &["after", "active_coarse_codes"]))
        .collect::<Result<Vec<_>, _>>()?;
    let unique = rows
        .iter()
        .map(|r| number(r, &["after", "population_unique_encodings"]))
        .collect::<Result<Vec<_>, _>>()?;
    let perplexity = rows
        .iter()
        .map(|r| number(r, &["after", "coarse_perplexity"]))
        .collect::<Result<Vec<_>, _>>()?;
    let collapsed = rows
        .iter()
        .map(|r| number(r, &["collapsed_fraction"]))
        .collect::<Result<Vec<_>, _>>()?;
    let missing = rows
        .iter()
        .map(|r| number(r, &["audited_after", "factorized_missing_teacher_mass"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut flops = rows
        .iter()
        .map(|r| number(r, &["codec_step_flops"]))
        .collect::<Result<Vec<_>, _>>()?;
    flops.sort_by(f64::total_cmp);
    flops.dedup_by(|a, b| a == b);
    let failures = recon
        .iter()
        .zip(&unique)
        .zip(&perplexity)
        .zip(&collapsed)
        .zip(&missing)
        .filter(|((((r, u), p), c), m)| {
            **r < 0.98 || **u != 16. || **p < 1.5 || **c > 0.05 || **m != 0.
        })
        .count();
    Ok(
        json!({"models":rows.len(),"mean_sequence_reconstruction":mean(&recon)?,"minimum_sequence_reconstruction":recon.iter().copied().fold(f64::INFINITY,f64::min),
        "mean_factorized_oracle_tv":mean(&tv)?,"mean_factorized_oracle_kl_nats":mean(&kl)?,
        "population_collapsed_models":active.iter().filter(|x|**x<2.).count(),"monitoring_failures":collapsed.iter().filter(|x|**x>0.05).count(),
        "collision_models":unique.iter().filter(|x|**x<16.).count(),"missing_support_models":missing.iter().filter(|x|**x>0.).count(),
        "all_fidelity_usage_gates_passed":failures==0,"codec_step_flops":flops}),
    )
}

/// Reproduce the summary of five paired seeds and all treatment arms.
pub fn summarize_maintenance(root: &Path) -> Result<Value, String> {
    let folder = root.join("reports/maintenance-ablation");
    let input = folder.join("results.json");
    let rows: Vec<Value> = serde_json::from_slice(&fs::read(&input).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let treatments = ["minimal_ema", "ema_recovery", "reference", "fixed_binary"];
    let mut summary = serde_json::Map::new();
    for name in treatments {
        let selected = rows_for(&rows, name);
        summary.insert(name.into(), summarize_treatment(&selected)?);
    }
    let comparisons = [
        ("ema_recovery", "minimal_ema"),
        ("ema_recovery", "reference"),
        ("fixed_binary", "ema_recovery"),
    ];
    let mut paired = serde_json::Map::new();
    for (left, right) in comparisons {
        let mut diffs = Vec::new();
        for seed in 13..18 {
            let value = |treatment: &str| -> Result<f64, String> {
                let selected = rows
                    .iter()
                    .filter(|row| {
                        row["seed"] == seed
                            && row["treatment"] == treatment
                            && (treatment != "ema_recovery"
                                || row["study"] == "maintenance-ablation")
                    })
                    .collect::<Vec<_>>();
                mean(
                    &selected
                        .iter()
                        .map(|r| number(r, &["after", "factorized_oracle_tv"]))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            };
            diffs.push(value(left)? - value(right)?);
        }
        let avg = mean(&diffs)?;
        let low = diffs.iter().copied().fold(f64::INFINITY, f64::min);
        let high = diffs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        paired.insert(
            format!("{left} minus {right}"),
            json!({"seed_mean_tv_differences":diffs,"mean":avg,"range":[low,high]}),
        );
    }
    let analysis = json!({"summary":summary,"paired_seed_analysis":paired});
    fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    let mut text = serde_json::to_string_pretty(&analysis).map_err(|e| e.to_string())?;
    text.push('\n');
    fs::write(folder.join("analysis.json"), text).map_err(|e| e.to_string())?;
    Ok(analysis)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn treatment_filter_and_means_preserve_protocol_scope() {
        let rows = vec![
            json!({"treatment":"ema_recovery","study":"maintenance-ablation","after":{"sequence_reconstruction":1.0,"factorized_oracle_tv":0.2,"factorized_oracle_kl_nats":0.1,"active_coarse_codes":2,"population_unique_encodings":16,"coarse_perplexity":2},"collapsed_fraction":0.01,"audited_after":{"factorized_missing_teacher_mass":0},"codec_step_flops":100}),
            json!({"treatment":"ema_recovery","study":"binary-maintenance-comparison","after":{"sequence_reconstruction":0.5}}),
        ];
        let selected = rows_for(&rows, "ema_recovery");
        assert_eq!(selected.len(), 1);
        assert_eq!(
            summarize_treatment(&selected).unwrap()["mean_sequence_reconstruction"],
            1.0
        );
    }
}
