//! Report provenance, JSON/CSV serialization, and paired bootstrap summaries.
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn identity(root: impl AsRef<Path>) -> Result<Value, String> {
    let root = root.as_ref();
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("git identity unavailable: {e}"))?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed".into());
    }
    let commit = String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())?
        .trim()
        .to_owned();
    let source = root.join("src/ncm");
    let mut hashes = BTreeMap::new();
    for item in fs::read_dir(&source).map_err(|e| e.to_string())? {
        let path = item.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|ext| ext == "py") {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            hashes.insert(
                rel,
                Value::String(digest(&fs::read(path).map_err(|e| e.to_string())?)),
            );
        }
    }
    let mut rust_hashes = BTreeMap::new();
    let mut pending = vec![root.join("rust/src")];
    while let Some(directory) = pending.pop() {
        for item in fs::read_dir(directory).map_err(|e| e.to_string())? {
            let path = item.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| e.to_string())?
                    .to_string_lossy()
                    .replace('\\', "/");
                rust_hashes.insert(
                    rel,
                    Value::String(digest(&fs::read(path).map_err(|e| e.to_string())?)),
                );
            }
        }
    }
    Ok(json!({"git_commit":commit,"source_sha256":hashes,"rust_source_sha256":rust_hashes}))
}

pub fn write_json(
    path: impl AsRef<Path>,
    root: impl AsRef<Path>,
    data: &Value,
) -> Result<(), String> {
    let identity = identity(root)?;
    let mut report = identity.as_object().cloned().ok_or("invalid identity")?;
    if let Some(values) = data.as_object() {
        for (key, value) in values {
            report.insert(key.clone(), value.clone());
        }
    }
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut text =
        serde_json::to_string_pretty(&Value::Object(report)).map_err(|e| e.to_string())?;
    text.push('\n');
    fs::write(path, text).map_err(|e| e.to_string())
}

fn csv_cell(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
        _ => value.to_string(),
    }
}
fn csv_quote(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

pub fn write_csv(
    path: impl AsRef<Path>,
    root: impl AsRef<Path>,
    rows: &[Map<String, Value>],
) -> Result<(), String> {
    let commit = identity(root)?["git_commit"]
        .as_str()
        .ok_or("missing commit")?
        .to_owned();
    let mut fields = Vec::<String>::new();
    for row in rows {
        for key in row.keys() {
            if !fields.contains(key) {
                fields.push(key.clone());
            }
        }
    }
    if !fields.iter().any(|x| x == "git_commit") {
        fields.push("git_commit".into());
    }
    let mut output = String::new();
    output.push_str(
        &fields
            .iter()
            .map(|x| csv_quote(x))
            .collect::<Vec<_>>()
            .join(","),
    );
    output.push('\n');
    for row in rows {
        let cells = fields
            .iter()
            .map(|field| {
                let value = if field == "git_commit" {
                    row.get(field)
                        .cloned()
                        .unwrap_or_else(|| Value::String(commit.clone()))
                } else {
                    row.get(field).cloned().unwrap_or(Value::Null)
                };
                csv_quote(&csv_cell(&value))
            })
            .collect::<Vec<_>>();
        output.push_str(&cells.join(","));
        output.push('\n');
    }
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(path, output).map_err(|e| e.to_string())
}

#[derive(Clone, Debug, PartialEq)]
pub struct BootstrapSummary {
    pub mean_difference: f64,
    pub ci95: [f64; 2],
    pub seeds: usize,
    pub label: &'static str,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let rank = q * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
}

pub fn paired_bootstrap(
    a: &[f64],
    b: &[f64],
    seed: u64,
    repetitions: usize,
) -> Result<BootstrapSummary, String> {
    if a.is_empty()
        || a.len() != b.len()
        || repetitions == 0
        || a.iter().chain(b).any(|x| !x.is_finite())
    {
        return Err("paired finite observations and positive repetitions required".into());
    }
    let delta: Vec<_> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let mean = delta.iter().sum::<f64>() / delta.len() as f64;
    let mut state = seed;
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let mut sum = 0.;
        for _ in 0..delta.len() {
            let index = ((splitmix64(&mut state) as u128 * delta.len() as u128) >> 64) as usize;
            sum += delta[index];
        }
        samples.push(sum / delta.len() as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(BootstrapSummary {
        mean_difference: mean,
        ci95: [percentile(&samples, 0.025), percentile(&samples, 0.975)],
        seeds: delta.len(),
        label: if delta.len() < 10 {
            "exploratory"
        } else {
            "bootstrap"
        },
    })
}

pub fn summary_json(summary: &BootstrapSummary) -> Value {
    json!({"mean_difference":summary.mean_difference,"ci95":summary.ci95,"seeds":summary.seeds,"label":summary.label})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_bootstrap_is_seeded_and_paired() {
        let a = [3., 4., 6.];
        let b = [1., 2., 3.];
        let x = paired_bootstrap(&a, &b, 9, 1000).unwrap();
        assert_eq!(x, paired_bootstrap(&a, &b, 9, 1000).unwrap());
        assert!((x.mean_difference - 7. / 3.).abs() < 1e-12);
        assert_eq!(x.seeds, 3);
    }
    #[test]
    fn csv_quotes_delimiters() {
        assert_eq!(csv_quote("a,\"b\""), "\"a,\"\"b\"\"\"");
    }

    #[test]
    fn json_and_csv_reports_include_provenance_and_round_trip() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let folder = std::env::temp_dir().join(format!("ncm reports {}", std::process::id()));
        fs::create_dir_all(&folder).unwrap();
        let json_path = folder.join("report.json");
        write_json(
            &json_path,
            root,
            &json!({"warning":true,"count":3,"values":[0.1,0.2]}),
        )
        .unwrap();
        let report: Value = serde_json::from_slice(&fs::read(&json_path).unwrap()).unwrap();
        assert_eq!(report["warning"], true);
        assert!(report["git_commit"].as_str().is_some());
        assert!(
            report["source_sha256"]["src/ncm/engine_rpc.py"]
                .as_str()
                .is_some()
        );
        assert!(
            report["rust_source_sha256"]["rust/src/workloads.rs"]
                .as_str()
                .is_some()
        );

        let csv_path = folder.join("results.csv");
        let rows =
            vec![serde_json::from_value(json!({"arm":"a","note":"x,y","score":1.25})).unwrap()];
        write_csv(&csv_path, root, &rows).unwrap();
        let csv = fs::read_to_string(csv_path).unwrap();
        assert!(csv.lines().next().unwrap().contains("git_commit"));
        assert!(csv.contains("\"x,y\""));
        let _ = fs::remove_dir_all(folder);
    }
}
