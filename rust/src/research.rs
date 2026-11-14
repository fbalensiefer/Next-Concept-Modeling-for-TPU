//! Rust-native summaries for committed research workload results.
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, io::Read, path::Path};
use zip::ZipArchive;

struct NpyArray {
    shape: Vec<usize>,
    values: Vec<f32>,
    values64: Vec<f64>,
}

fn npy_array(bytes: &[u8]) -> Result<NpyArray, String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("invalid NumPy array header".into());
    }
    let major = bytes[6];
    let (header_length, data_offset): (usize, usize) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else if major == 2 || major == 3 {
        if bytes.len() < 12 {
            return Err("truncated NumPy v2 array header".into());
        }
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    } else {
        return Err(format!("unsupported NumPy format version {major}"));
    };
    let end = data_offset
        .checked_add(header_length)
        .ok_or("NumPy header overflow")?;
    let header = std::str::from_utf8(
        bytes
            .get(data_offset..end)
            .ok_or("truncated NumPy header")?,
    )
    .map_err(|e| e.to_string())?;
    let fortran_order =
        header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True");
    let descr_start = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or("NumPy dtype missing")?;
    let dtype = header[descr_start..]
        .split(':')
        .nth(1)
        .ok_or("NumPy dtype malformed")?
        .trim_start();
    let quote = dtype
        .as_bytes()
        .first()
        .copied()
        .ok_or("NumPy dtype malformed")? as char;
    let dtype_end = dtype[1..].find(quote).ok_or("NumPy dtype malformed")? + 1;
    let dtype = &dtype[1..dtype_end];
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("NumPy shape missing")?;
    let shape_text = header[shape_start..]
        .split('(')
        .nth(1)
        .and_then(|v| v.split(')').next())
        .ok_or("NumPy shape malformed")?;
    let shape = shape_text
        .split(',')
        .filter_map(|x| x.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    let count = shape.iter().product::<usize>();
    let data = &bytes[end..];
    let values64 = match dtype {
        "<f4" | "=f4" | "|f4" => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) as f64)
            .collect::<Vec<_>>(),
        "<f8" | "=f8" | "|f8" => data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect::<Vec<_>>(),
        "<i4" | "=i4" | "|i4" => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as f64)
            .collect::<Vec<_>>(),
        other => return Err(format!("unsupported NumPy dtype {other}")),
    };
    if values64.len() != count {
        return Err(format!(
            "NumPy shape {:?} expects {count} values, found {}",
            shape,
            values64.len()
        ));
    }
    let values64 = if fortran_order {
        (0..count)
            .map(|linear| {
                let mut remaining = linear;
                let mut coordinates = vec![0usize; shape.len()];
                for axis in (0..shape.len()).rev() {
                    coordinates[axis] = remaining % shape[axis];
                    remaining /= shape[axis];
                }
                let mut stride = 1;
                let mut offset = 0;
                for axis in 0..shape.len() {
                    offset += coordinates[axis] * stride;
                    stride *= shape[axis];
                }
                values64[offset]
            })
            .collect()
    } else {
        values64
    };
    let values = values64.iter().map(|x| *x as f32).collect();
    Ok(NpyArray {
        shape,
        values,
        values64,
    })
}

fn npz_arrays(path: &Path) -> Result<std::collections::HashMap<String, NpyArray>, String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut arrays = std::collections::HashMap::new();
    for index in 0..archive.len() {
        let mut member = archive.by_index(index).map_err(|e| e.to_string())?;
        let name = member
            .name()
            .strip_suffix(".npy")
            .ok_or("NPZ member is not a NumPy array")?
            .to_owned();
        let mut bytes = Vec::new();
        member.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        arrays.insert(name, npy_array(&bytes)?);
    }
    Ok(arrays)
}

fn array<'a>(
    arrays: &'a std::collections::HashMap<String, NpyArray>,
    name: &str,
    shape: &[usize],
) -> Result<&'a [f32], String> {
    let value = arrays
        .get(name)
        .ok_or_else(|| format!("NPZ is missing {name}"))?;
    if value.shape != shape {
        return Err(format!(
            "NPZ {name} shape mismatch: {:?} != {shape:?}",
            value.shape
        ));
    }
    Ok(&value.values)
}

fn array64<'a>(
    arrays: &'a std::collections::HashMap<String, NpyArray>,
    name: &str,
    shape: &[usize],
) -> Result<&'a [f64], String> {
    let value = arrays
        .get(name)
        .ok_or_else(|| format!("NPZ is missing {name}"))?;
    if value.shape != shape {
        return Err(format!(
            "NPZ {name} shape mismatch: {:?} != {shape:?}",
            value.shape
        ));
    }
    Ok(&value.values64)
}

fn softmax2(logits: [f32; 2]) -> [f32; 2] {
    let max = logits[0].max(logits[1]);
    let e = [(logits[0] - max).exp(), (logits[1] - max).exp()];
    let s = e[0] + e[1];
    [e[0] / s, e[1] / s]
}

struct HeadDiagnostic<'a> {
    teacher: &'a [f32],
    scale: &'a [f32],
    mean: &'a [f32],
    projection: &'a [f32],
    kernel: &'a [f32],
    bias: &'a [f32],
}

fn predict_kl(h: &[f32], p: &[f64], head: &HeadDiagnostic<'_>) -> f64 {
    let mut expectation = 0.;
    for (row, weight) in p.iter().enumerate().take(16) {
        let mut row_sum = 0.;
        for t in 0..4 {
            let mut restored = [0f32; 16];
            for (j, restored_value) in restored.iter_mut().enumerate() {
                let mut x = 0f32;
                for k in 0..8 {
                    x = (h[(row * 4 + t) * 8 + k] * head.scale[k])
                        .mul_add(head.projection[j * 8 + k], x);
                }
                *restored_value = x + head.mean[j];
            }
            let mut logits = [0f32; 2];
            for (c, logit) in logits.iter_mut().enumerate() {
                for (j, value) in restored.iter().enumerate() {
                    *logit = value.mul_add(head.kernel[j * 2 + c], *logit);
                }
                *logit += head.bias[c];
            }
            let q = softmax2(logits);
            let tr = softmax2({
                let base = (row * 4 + t) * 16;
                let mut x = [0f32; 2];
                for (c, value) in x.iter_mut().enumerate() {
                    for j in 0..16 {
                        *value = head.teacher[base + j].mul_add(head.kernel[j * 2 + c], *value);
                    }
                    *value += head.bias[c];
                }
                x
            });
            let mut kl = 0f32;
            for c in 0..2 {
                kl += tr[c] * (tr[c].ln() - q[c].max(1e-30).ln());
            }
            row_sum += kl;
        }
        expectation += weight * (row_sum / 4.) as f64;
    }
    expectation
}

fn weighted_mse(h: &[f32], target: &[f32], p: &[f64]) -> f64 {
    let mut result = 0.;
    for (row, weight) in p.iter().enumerate().take(16) {
        let start = row * 32;
        let mut lanes = [0f32; 8];
        for block in 0..4 {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let i = start + block * 8 + lane;
                let d = h[i] - target[i];
                *sum += d * d;
            }
        }
        let sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        result += weight * (sum / 32.) as f64;
    }
    result
}

/// Replay the committed feature-head alignment diagnostic without importing model code.
pub fn diagnose_alignment(root: &Path) -> Result<Value, String> {
    let folder = root.join("reports/alignment-diagnosis");
    let meta: Value =
        serde_json::from_slice(&fs::read(folder.join("inputs.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let path = folder.join("diagnostic_inputs.npz");
    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    if crate::corpus::sha256(&bytes)
        != meta["sha256"]
            .as_str()
            .ok_or("inputs.json has no archive hash")?
    {
        return Err("alignment diagnostic archive hash mismatch".into());
    }
    let arrays = npz_arrays(&path)?;
    let p = array64(&arrays, "probability", &[16])?;
    let target = array(&arrays, "canonical", &[16, 4, 8])?;
    let means = array(&arrays, "mean", &[16])?;
    let projection = array(&arrays, "projection", &[16, 8])?;
    let scale = array(&arrays, "scale", &[8])?;
    let kernel = array(&arrays, "head_kernel", &[16, 2])?;
    let bias = array(&arrays, "head_bias", &[2])?;
    let teacher = array(&arrays, "teacher_hidden", &[16, 4, 16])?;
    let head = HeadDiagnostic {
        teacher,
        scale,
        mean: means,
        projection,
        kernel,
        bias,
    };
    let mut target_range = 0f32;
    for feature in 0..8 {
        let lo = (0..16)
            .map(|r| target[r * 32 + feature])
            .fold(f32::INFINITY, f32::min);
        let hi = (0..16)
            .map(|r| target[r * 32 + feature])
            .fold(f32::NEG_INFINITY, f32::max);
        target_range = target_range.max(hi - lo);
    }
    let mut pair_diff = 0f32;
    for r in (0..16).step_by(2) {
        for i in 0..32 {
            pair_diff = pair_diff.max((target[r * 32 + i] - target[(r + 1) * 32 + i]).abs());
        }
    }
    let mut last_entropy = 0f64;
    for i in (0..16).step_by(2) {
        let mass = p[i] + p[i + 1];
        let a = p[i] / mass;
        let b = p[i + 1] / mass;
        last_entropy -= mass * (a * a.log2() + b * b.log2());
    }
    let mut rows = Vec::new();
    let models = meta["models"].as_array().ok_or("models must be array")?;
    let latents = array(&arrays, "latents", &[20, 16, 4, 8])?;
    let quantized = array(&arrays, "quantized", &[20, 16, 4, 8])?;
    for (index, model) in models.iter().enumerate() {
        let start = index * 512;
        let latent = &latents[start..start + 512];
        let quant = &quantized[start..start + 512];
        rows.push(json!({"seed":model["seed"],"arm":model["arm"],"codec_sha256":model["codec_sha256"],"latent_mse":weighted_mse(latent,target,p),"quantized_mse":weighted_mse(quant,target,p),"latent_predictive_kl":predict_kl(latent,p,&head),"quantized_predictive_kl":predict_kl(quant,p,&head)}));
    }
    let predictive = predict_kl(target, p, &head);
    let invariants = json!({"first_position_target_max_range":target_range,"last_bit_pair_target_max_difference":pair_diff,"last_bit_conditional_entropy_bits":last_entropy,"pca_teacher_predictive_kl_nats":predictive});
    let arms = rows.iter().filter_map(|r| r["arm"].as_str()).fold(
        Vec::<String>::new(),
        |mut out, name| {
            if !out.iter().any(|x| x == name) {
                out.push(name.to_owned());
            }
            out
        },
    );
    let mut means_by_arm = serde_json::Map::new();
    for arm in arms {
        let mut fields = serde_json::Map::new();
        for key in [
            "latent_mse",
            "quantized_mse",
            "latent_predictive_kl",
            "quantized_predictive_kl",
        ] {
            let xs = rows
                .iter()
                .filter(|r| r["arm"] == arm)
                .map(|r| r[key].as_f64().unwrap())
                .collect::<Vec<_>>();
            fields.insert(key.into(), json!(mean(&xs)?));
        }
        means_by_arm.insert(arm, Value::Object(fields));
    }
    let result = json!({"scope":"One historical teacher and finite domain; feature-head compatibility diagnostic, not a causal intervention.","invariants":invariants,"means":means_by_arm,"models":rows});
    write_json(&folder.join("results.json"), &result)?;
    Ok(result)
}

/// Publish the fixed reconstruction-capacity report from an immutable run ledger.
pub fn summarize_reconstruction_capacity(run: &Path, output: &Path) -> Result<(), String> {
    let source = if run.is_absolute() {
        run.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(run)
    };
    let destination = if output.is_absolute() {
        output.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(output)
    };
    let rows = read_rows(&source.join("summary.json"))?;
    if rows.len() != 6
        || rows.iter().any(|r| {
            r["records"]
                .as_array()
                .and_then(|a| a.last())
                .is_none_or(|x| x["step"] != 2000)
        })
    {
        return Err("Require all six complete prespecified fits".into());
    }
    fs::create_dir_all(&destination).map_err(|e| e.to_string())?;
    for name in ["protocol.json", "summary.json", "controls.json"] {
        fs::copy(source.join(name), destination.join(name)).map_err(|e| e.to_string())?;
    }
    let mut lines=vec!["# Fixed-length reconstruction: duration and code capacity".into(),String::new(),"Six fresh CPU fits use the same 128-byte windows, training orders and seeds as paired controls. Codebook size (16 versus 64) is the sole architecture change. All runs use 2,000 updates, batch size four, reconstruction-first losses and the final checkpoint. Validation remains descriptive; test reconstruction is unevaluated. Source snapshots are preserved under the run directory.".into(),String::new(),"| Codebook | Seed | Step | Train byte accuracy | Validation byte accuracy | Validation exact windows | Punctuation accuracy |".into(),"|---|---|---|---|---|---|---|".into()];
    for row in &rows {
        for record in row["records"].as_array().unwrap() {
            let step = record["step"].as_u64().unwrap_or(0);
            if ![100, 500, 1000, 2000].contains(&step) {
                continue;
            }
            let train = &record["train"];
            let valid = &record["validation"];
            lines.push(format!(
                "| {} | {} | {step} | {:.4}% | {:.4}% | {:.4}% | {:.4}% |",
                row["codebook_size"],
                row["seed"],
                train["byte_accuracy"].as_f64().unwrap() * 100.,
                valid["byte_accuracy"].as_f64().unwrap() * 100.,
                valid["exact_window_accuracy"].as_f64().unwrap() * 100.,
                valid["slices"]["punctuation"]["accuracy"].as_f64().unwrap() * 100.
            ));
        }
    }
    lines.extend([
        String::new(),
        "## Paired descriptive effects".into(),
        String::new(),
    ]);
    for size in [16, 64] {
        let selected = rows
            .iter()
            .filter(|r| r["codebook_size"] == size)
            .collect::<Vec<_>>();
        let changes = selected
            .iter()
            .map(|r| {
                r["records"].as_array().unwrap().last().unwrap()["validation"]["byte_accuracy"]
                    .as_f64()
                    .unwrap()
                    - r["records"][0]["validation"]["byte_accuracy"]
                        .as_f64()
                        .unwrap()
            })
            .collect::<Vec<_>>();
        lines.push(format!("- {size} entries, 100 to 2,000 updates: mean validation byte accuracy change {:.2} percentage points; seed range {:.2} to {:.2}.",100.*mean(&changes)?,100.*changes.iter().copied().fold(f64::INFINITY,f64::min),100.*changes.iter().copied().fold(f64::NEG_INFINITY,f64::max)));
    }
    let mut paired = Vec::new();
    for seed in [0, 1, 2] {
        let get = |size| {
            rows.iter()
                .find(|r| r["seed"] == seed && r["codebook_size"] == size)
                .ok_or("missing paired seed")
        };
        let small = get(16)?;
        let large = get(64)?;
        paired.push(large["records"].as_array().unwrap().last().unwrap()["validation"]["byte_accuracy"].as_f64().unwrap()-small["records"].as_array().unwrap().last().unwrap()["validation"]["byte_accuracy"].as_f64().unwrap());
    }
    lines.extend([format!("- 64 versus 16 entries at 2,000 updates: mean paired difference {:.2} percentage points; seed range {:.2} to {:.2}.",100.*mean(&paired)?,100.*paired.iter().copied().fold(f64::INFINITY,f64::min),100.*paired.iter().copied().fold(f64::NEG_INFINITY,f64::max)),String::new(),"These ranges describe three model seeds on one fixed validation book. They are not population confidence intervals or independent corpus replications. The follow-up was chosen after observing the initial pilot, so its findings are exploratory.".into(),String::new(),"## Cost and controls".into(),String::new(),"| Representation | Nominal bits/input byte | Reconstruction training byte exposure |".into(),"|---|---|---|".into(),"| Raw bytes | 8 | 0 |".into(),"| Fixed byte pairs | 8 | 0 |".into(),"| RVQ, 16 entries | 7 | 1,024,000 per fit |".into(),"| RVQ, 64 entries | 10.5 | 1,024,000 per fit |".into(),String::new(),"Both exact controls reconstruct every train/validation window perfectly. Packing byte pairs does not compress them: the output vocabulary has 65,536 categories, whose prediction heads and sampling costs must be charged in any prior comparison. RVQ nominal bits exclude codec weights, codebook tables and framing. More codebook entries change both information capacity and optimization; this experiment does not isolate an information-theoretic cause.".into(),String::new(),"| Codebook | Seed | Codec parameters | Codebook embedding elements | Full run wall seconds |".into(),"|---|---|---|---|---|".into()]);
    for row in &rows {
        lines.push(format!(
            "| {} | {} | {} | {} | {:.1} |",
            row["codebook_size"],
            row["seed"],
            row["codec_parameters"],
            row["codebook_embedding_elements"],
            row["elapsed_seconds"].as_f64().unwrap()
        ));
    }
    lines.extend([String::new(),"Wall time includes compilation, full train/validation evaluation every 100 steps and serialization. It is not a matched training-cost or generation-latency measure. No TPU was used.".into(),String::new(),"## Decision".into(),String::new()]);
    let improves = rows.iter().all(|r| {
        r["records"].as_array().unwrap().last().unwrap()["validation"]["byte_accuracy"]
            .as_f64()
            .unwrap()
            > r["records"][0]["validation"]["byte_accuracy"]
                .as_f64()
                .unwrap()
    });
    let large_wins = paired.iter().all(|x| *x > 0.);
    lines.push(format!("{}{}These observations support a bounded next check of the strongest codec on independent documents and consequential error patterns. Keep the current codecs experimental; no application fidelity tolerance has been established. Raw and fixed-block controls already offer exact reconstruction without codec training; a learned codec must still demonstrate generation value.",if improves{"Longer training improves validation byte accuracy in every seed for both codebook sizes. "}else{"Training-duration effects vary across seeds; inspect all curves. "},if large_wins{"The larger codebook also improves final byte accuracy in every paired seed. "}else{"The larger codebook does not improve final byte accuracy in every paired seed. "}));
    lines.extend([String::new(),"The remaining real-text checks include rare patterns, UTF-8 validity, independent documents, a frozen application fidelity tolerance and trained prior comparisons. No claims about generation quality or acceleration follow from these reconstruction results.".into(),String::new(),"## Reproduction".into(),String::new(),"```sh".into(),"cargo run --bin ncm-run -- reconstruction-capacity --output runs/reconstruction-capacity-v1".into(),"cargo run --bin ncm-run -- summarize-reconstruction-capacity --run runs/reconstruction-capacity-v1 --output reports/reconstruction-capacity".into(),"```".into(),String::new(),"Use a fresh run directory. Requires the initial pilot’s processed files and protocol. [Protocol](protocol.json), [all checkpoint metrics](summary.json), [exact controls](controls.json).".into(),String::new()]);
    fs::write(destination.join("report.md"), lines.join("\n")).map_err(|e| e.to_string())
}

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
