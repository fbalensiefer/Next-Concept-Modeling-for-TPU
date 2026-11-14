//! Synthetic within-level TC gate. Rust owns source laws, sample streams, gates,
//! aggregation and artifacts; the JAX worker owns the learned probe models.
use crate::{rpc::EngineClient, validation_io};
use serde_json::{Map, Value, json};
use std::{collections::HashSet, fs, path::Path};

const SOURCES: [&str; 6] = [
    "markov",
    "parity",
    "copied_bit",
    "hmm",
    "constant_mixture",
    "uniform",
];

fn source_law(name: &str, length: usize) -> Result<(Vec<Vec<i32>>, Vec<f64>), String> {
    if length == 0 || length > 16 {
        return Err("probe length must be in 1..=16".into());
    }
    let count = 1usize << length;
    let words: Vec<Vec<i32>> = (0..count)
        .map(|word| {
            (0..length)
                .map(|i| ((word >> (length - i - 1)) & 1) as i32)
                .collect()
        })
        .collect();
    let mut law = Vec::with_capacity(count);
    for word in &words {
        let p = match name {
            "markov" => (0..length.saturating_sub(1)).fold(0.5, |p, i| {
                p * if word[i] == word[i + 1] { 0.85 } else { 0.15 }
            }),
            "parity" => {
                let parity = word[..length - 1].iter().sum::<i32>() % 2;
                (if word[length - 1] == parity { 0.9 } else { 0.1 })
                    / (2f64.powi(length as i32 - 1))
            }
            "copied_bit" => {
                if word.iter().all(|x| *x == word[0]) {
                    0.5
                } else {
                    0.0
                }
            }
            "hmm" => {
                let mut state = [0.5, 0.5];
                for (i, bit) in word.iter().enumerate() {
                    for (s, v) in state.iter_mut().enumerate() {
                        *v *= if *bit == s as i32 { 0.85 } else { 0.15 };
                    }
                    if i + 1 < length {
                        let old = state;
                        state = [old[0] * 0.9 + old[1] * 0.1, old[0] * 0.1 + old[1] * 0.9];
                    }
                }
                state[0] + state[1]
            }
            "constant_mixture" => {
                let ones = word.iter().filter(|x| **x == 1).count() as i32;
                0.5 * (0.1f64.powi(ones) * 0.9f64.powi(length as i32 - ones)
                    + 0.9f64.powi(ones) * 0.1f64.powi(length as i32 - ones))
            }
            "uniform" => 1.0 / count as f64,
            _ => return Err(format!("unknown probe source {name}")),
        };
        law.push(p);
    }
    let total: f64 = law.iter().sum();
    for p in &mut law {
        *p /= total;
    }
    Ok((words, law))
}

fn conditional_tc(words: &[Vec<i32>], law: &[f64], lengths: &[usize]) -> Vec<f64> {
    fn entropy(words: &[Vec<i32>], law: &[f64], kept: &[usize]) -> f64 {
        if kept.is_empty() {
            return 0.;
        }
        let mut masses = vec![0.; 1usize << kept.len()];
        for (word, p) in words.iter().zip(law) {
            let key = kept
                .iter()
                .fold(0usize, |v, i| (v << 1) | word[*i] as usize);
            masses[key] += p;
        }
        -masses
            .into_iter()
            .filter(|p| *p > 0.)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }
    let mut start = 0;
    let mut output = Vec::new();
    for &n in lengths {
        let prefix: Vec<usize> = (0..start).collect();
        let mut tc = 0.;
        for j in 0..n {
            let mut group = prefix.clone();
            group.push(start + j);
            tc += entropy(words, law, &group);
        }
        let mut whole = prefix.clone();
        whole.extend(start..start + n);
        tc -= entropy(words, law, &whole)
            + (n.saturating_sub(1) as f64) * entropy(words, law, &prefix);
        output.push(tc.max(0.));
        start += n;
    }
    output
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn sample_ids(law: &[f64], count: usize, state: &mut u64) -> Vec<usize> {
    (0..count)
        .map(|_| {
            let x = (next(state) as f64) / (u64::MAX as f64);
            let mut c = 0.;
            law.iter()
                .position(|p| {
                    c += p;
                    c >= x
                })
                .unwrap_or(law.len() - 1)
        })
        .collect()
}
fn rows(value: &Value) -> Result<Vec<Map<String, Value>>, String> {
    value
        .as_array()
        .ok_or("internal rows error")?
        .iter()
        .map(|v| {
            v.as_object()
                .cloned()
                .ok_or_else(|| "row must be object".into())
        })
        .collect()
}

pub fn run_tc_probe(root: &Path, output: &Path, steps: usize) -> Result<bool, String> {
    if steps == 0 {
        return Err("steps must be positive".into());
    }
    let output = if output.is_absolute() {
        output.to_owned()
    } else {
        root.join(output)
    };
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let mut engine = EngineClient::start(root)?;
    let mut result_rows = Vec::new();
    for seed in [200u64, 201, 202] {
        for length in [8usize, 10] {
            let lengths = [length / 2, length - length / 2];
            for (source_index, source) in SOURCES.iter().enumerate() {
                let (words, law) = source_law(source, length)?;
                let exact = conditional_tc(&words, &law, &lengths);
                for size in [256usize, 512, 2048] {
                    let stream = source_index as u64 * 100000 + length as u64 * 1000 + size as u64;
                    let mut a = seed ^ stream.rotate_left(17);
                    let train_ids = sample_ids(&law, size, &mut a);
                    let mut b = seed ^ stream.rotate_left(17) ^ 0xa5a5a5a55a5a5a5a;
                    let eval_ids = sample_ids(&law, 4096, &mut b);
                    let train_set: HashSet<usize> = train_ids.iter().copied().collect();
                    let eval_set: HashSet<usize> = eval_ids.iter().copied().collect();
                    let overlap = eval_set.intersection(&train_set).count();
                    let train: Vec<Vec<i32>> =
                        train_ids.iter().map(|i| words[*i].clone()).collect();
                    let evaluation: Vec<Vec<i32>> =
                        eval_ids.iter().map(|i| words[*i].clone()).collect();
                    for (level, exact_tc) in exact.iter().enumerate() {
                        let mut estimates = [0f64; 2];
                        for (causal, slot) in [(false, 0), (true, 1)] {
                            let answer:Value=engine.call("probe.train",&json!({"train":train,"evaluation":evaluation,"lengths":lengths,"level":level,"seed":seed,"steps":steps,"batch_size":64,"causal":causal}))?;
                            estimates[slot] = answer["nll"]
                                .as_f64()
                                .ok_or("probe worker returned invalid nll")?;
                        }
                        let estimate = estimates[0] - estimates[1];
                        let error = (estimate - exact_tc).abs();
                        let tolerance = 0.02f64.max(0.1 * exact_tc);
                        result_rows.push(json!({"seed":seed,"source":source,"length":length,"training_size":size,"level":level,"exact_tc_nats":exact_tc,"delta_nll_nats":estimate,"parallel_nll_nats":estimates[0],"ar_nll_nats":estimates[1],"absolute_error":error,"tolerance":tolerance,"numerical_gate":error<=tolerance,"identity_overlap":overlap,"disjoint_identity_gate":overlap==0}));
                    }
                }
            }
        }
    }
    let pass = result_rows
        .iter()
        .filter(|r| r["numerical_gate"] == true)
        .count() as f64
        / result_rows.len() as f64;
    let disjoint = result_rows
        .iter()
        .all(|r| r["disjoint_identity_gate"] == true);
    let passed = pass >= 0.9 && disjoint;
    let coarse: f64 = result_rows
        .iter()
        .filter(|r| r["level"] == 0)
        .map(|r| r["delta_nll_nats"].as_f64().unwrap().max(0.))
        .sum();
    let fine: f64 = result_rows
        .iter()
        .filter(|r| r["level"] == 1)
        .map(|r| r["delta_nll_nats"].as_f64().unwrap().max(0.))
        .sum();
    let share = coarse / (coarse + fine).max(1e-30);
    let identity = validation_io::identity(root)?;
    validation_io::write_json(
        output.join("probe_results.json"),
        root,
        &json!({"rows":result_rows,"numerical_pass_fraction":pass,"disjoint_identity_gate":disjoint,"real_text_gate":passed,"status":"complete","seeds":[200,201,202]}),
    )?;
    let source_rows = rows(&Value::Array(result_rows.clone()))?;
    let commit = identity["git_commit"]
        .as_str()
        .unwrap_or("unknown")
        .to_owned();
    let csv_rows = source_rows
        .iter()
        .map(|row| {
            let mut record = Map::new();
            record.insert("git_commit".into(), Value::String(commit.clone()));
            for (key, value) in row {
                record.insert(key.clone(), value.clone());
            }
            record
        })
        .collect::<Vec<_>>();
    validation_io::write_csv(output.join("probe_results.csv"), root, &csv_rows)?;
    let mut text = format!(
        "# TC probe decision\n\ngit_commit: {}\n\nNumerical gate pass fraction: {pass:.6}; required 0.90.\nStrict disjoint-sequence-identity gate: {disjoint}.\nReal-text gate: {passed}.\n\nSynthetic pooled estimated TC share: coarse {share:.6}, fine {:.6}.\nSynthetic >=30% coarse-share decision: {}.\nThese pooled finite-model estimates are biased diagnostics, not real-text TC.\n",
        identity["git_commit"].as_str().unwrap_or("unknown"),
        1. - share,
        share >= 0.3
    );
    if !passed {
        text.push_str("\nNo TC claim is permitted on real text. Real-text code probing is stopped.\nCopied-bit has only two distinct sequences; identity-disjoint train/eval\nwith the same nondegenerate source law is impossible. Overlaps and numerical\nfailures remain in the table. No extra hybrid training is triggered by this decision;\nindependent preregistered benchmark models proceed as planned.\n");
    } else {
        text.push_str("\nEnumerable gate passes. No text codes were supplied; real-text TC remains untested.\n");
    }
    fs::write(output.join("tc_probe_decision.md"), text).map_err(|e| e.to_string())?;
    Ok(passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_laws_normalize_and_tc_matches_archived_values() {
        let archived = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/reports/validation-controls/tc-probe/probe_results.csv"
        ))
        .unwrap();
        let mut records = std::collections::HashMap::new();
        for line in archived.lines().skip(1) {
            let fields = line.split(',').collect::<Vec<_>>();
            if fields.get(1) == Some(&"200") && fields.get(4) == Some(&"256") {
                records.insert(
                    (
                        fields[2].to_owned(),
                        fields[3].parse::<usize>().unwrap(),
                        fields[5].parse::<usize>().unwrap(),
                    ),
                    fields[6].parse::<f64>().unwrap(),
                );
            }
        }
        for length in [8usize, 10] {
            for source in SOURCES {
                let (words, law) = source_law(source, length).unwrap();
                assert_eq!(words.len(), 1usize << length);
                assert!((law.iter().sum::<f64>() - 1.).abs() < 1e-14);
                let lengths = [length / 2, length - length / 2];
                let tc = conditional_tc(&words, &law, &lengths);
                for (level, tc_value) in tc.iter().enumerate() {
                    let expected = records.get(&(source.to_owned(), length, level)).unwrap();
                    assert!(
                        (tc_value - expected).abs() < 1e-12,
                        "{source} length {length} level {level}: {} != {expected}",
                        tc_value
                    );
                }
            }
        }
        let required = [
            "seed",
            "source",
            "length",
            "training_size",
            "level",
            "exact_tc_nats",
            "delta_nll_nats",
            "parallel_nll_nats",
            "ar_nll_nats",
            "absolute_error",
            "tolerance",
            "numerical_gate",
            "identity_overlap",
            "disjoint_identity_gate",
        ];
        for key in required {
            assert!(
                archived
                    .lines()
                    .next()
                    .unwrap()
                    .split(',')
                    .any(|x| x == key),
                "archived schema missing {key}"
            );
        }
    }
}
