//! Rust-owned training and evaluation orchestration over the JAX worker IPC.
use crate::{
    corpus,
    rpc::{
        CheckpointRequest, DeviceConfig, EngineClient, EngineConfig, ForwardRequest,
        InitModelRequest, LossConfig, ModelConfig, RvqConfig, TensorBatch, TrainStepRequest,
        TransformerConfig,
    },
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug)]
struct Dataset {
    tokens: Vec<Vec<i32>>,
    mask: Vec<Vec<bool>>,
    sha256: String,
}

fn raw_dataset(path: &Path, length: usize) -> Result<Dataset, String> {
    if length == 0 {
        return Err("model sequence length must be positive".into());
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.is_empty() {
        return Err("training and validation data must be nonempty".into());
    }
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for chunk in bytes.chunks(length) {
        let mut row = vec![0; length];
        let mut valid = vec![false; length];
        for (i, b) in chunk.iter().enumerate() {
            row[i] = *b as i32;
            valid[i] = true;
        }
        tokens.push(row);
        mask.push(valid);
    }
    Ok(Dataset {
        tokens,
        mask,
        sha256: corpus::sha256(&bytes),
    })
}
fn document_dataset(path: &Path, length: usize) -> Result<Dataset, String> {
    let windows = corpus::document_masked_byte_windows(path, length).map_err(|e| e.to_string())?;
    Ok(Dataset {
        tokens: windows
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(i32::from).collect())
            .collect(),
        mask: windows.masks,
        sha256: windows.sha256,
    })
}
fn full_document_dataset(path: &Path, length: usize) -> Result<Dataset, String> {
    let windows = corpus::document_byte_windows(path, length).map_err(|e| e.to_string())?;
    let mask = windows
        .rows
        .iter()
        .map(|row| vec![true; row.len()])
        .collect();
    Ok(Dataset {
        tokens: windows
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(i32::from).collect())
            .collect(),
        mask,
        sha256: windows.sha256,
    })
}
fn positive_usize(config: &Value, key: &str, default: usize) -> Result<usize, String> {
    let n = config
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or(default as u64);
    if n == 0 {
        return Err(format!("{key} must be positive"));
    }
    usize::try_from(n).map_err(|_| format!("{key} is too large"))
}
fn json_or<T: serde::de::DeserializeOwned>(
    config: &Value,
    key: &str,
    default: T,
) -> Result<T, String> {
    match config.get(key) {
        Some(v) => serde_json::from_value(v.clone()).map_err(|e| format!("invalid {key}: {e}")),
        None => Ok(default),
    }
}
fn batch_at(data: &Dataset, indices: &[usize]) -> TensorBatch {
    TensorBatch {
        tokens: indices.iter().map(|i| data.tokens[*i].clone()).collect(),
        mask: Some(indices.iter().map(|i| data.mask[*i].clone()).collect()),
        fallback_bytes: None,
        fallback_mask: None,
        prompt_ids: None,
        prompt_mask: None,
    }
}
fn resolve_path(root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}
fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn choose_indices(count: usize, population: usize, state: &mut u64) -> Vec<usize> {
    (0..count)
        .map(|_| ((next_random(state) as u128 * population as u128) >> 64) as usize)
        .collect()
}
fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut content = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    content.push('\n');
    fs::write(path, content).map_err(|e| e.to_string())
}
fn add_weighted(total: &mut Value, value: &Value, weight: f64) -> Result<(), String> {
    match (total, value) {
        (Value::Number(left), Value::Number(right)) => {
            let sum = left.as_f64().ok_or("metric is not numeric")?
                + right.as_f64().ok_or("metric is not numeric")? * weight;
            *left = serde_json::Number::from_f64(sum).ok_or("non-finite aggregated metric")?;
        }
        (Value::Array(left), Value::Array(right)) if left.len() == right.len() => {
            for (a, b) in left.iter_mut().zip(right) {
                add_weighted(a, b, weight)?;
            }
        }
        (Value::Object(left), Value::Object(right)) if left.keys().eq(right.keys()) => {
            for (key, item) in left {
                add_weighted(item, &right[key], weight)?;
            }
        }
        _ => return Err("metric shape changed during evaluation".into()),
    }
    Ok(())
}
fn scale_metrics(value: &mut Value, scale: f64) {
    match value {
        Value::Number(number) => {
            if let Some(scaled) = number
                .as_f64()
                .and_then(|n| serde_json::Number::from_f64(n * scale))
            {
                *number = scaled;
            }
        }
        Value::Array(items) => {
            for item in items {
                scale_metrics(item, scale);
            }
        }
        Value::Object(items) => {
            for item in items.values_mut() {
                scale_metrics(item, scale);
            }
        }
        _ => {}
    }
}
fn evaluate(
    engine: &mut EngineClient,
    engine_id: &str,
    data: &Dataset,
    batch_size: usize,
) -> Result<Value, String> {
    let mut valid_bytes = 0usize;
    let mut loss_sum = 0.0;
    let mut metrics: Option<Value> = None;
    let mut batches = 0usize;
    for start in (0..data.tokens.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.tokens.len());
        let indices = (start..end).collect::<Vec<_>>();
        let count = indices
            .iter()
            .map(|i| data.mask[*i].iter().filter(|valid| **valid).count())
            .sum::<usize>();
        let response = engine.forward(&ForwardRequest {
            engine_id: engine_id.to_owned(),
            batch: batch_at(data, &indices),
        })?;
        loss_sum += response.loss * count as f64;
        valid_bytes += count;
        batches += 1;
        if let Some(total) = metrics.as_mut() {
            add_weighted(total, &response.metrics, count as f64)?;
        } else {
            let mut first = response.metrics;
            scale_metrics(&mut first, count as f64);
            metrics = Some(first);
        }
    }
    if valid_bytes == 0 {
        return Err("evaluation data contains no valid bytes".into());
    }
    let mut metrics = metrics.unwrap_or(Value::Null);
    scale_metrics(&mut metrics, 1. / valid_bytes as f64);
    Ok(
        json!({"loss":loss_sum/valid_bytes as f64,"valid_bytes":valid_bytes,"batches":batches,"metrics":metrics}),
    )
}

/// Train the NCM codec from byte text files. All corpus handling, batching,
/// run identity, checkpoint scheduling, and metric reporting stay in Rust.
pub fn train_codec(
    root: &Path,
    config_path: &Path,
    output: &Path,
    resume: bool,
) -> Result<(), String> {
    let config_path = if config_path.is_absolute() {
        config_path.to_owned()
    } else {
        root.join(config_path)
    };
    let output = if output.is_absolute() {
        output.to_owned()
    } else {
        root.join(output)
    };
    let config: Value = serde_json::from_slice(&fs::read(&config_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let lengths: Vec<usize> = json_or(&config, "lengths", vec![1, 2, 4])?;
    let sequence_length = *lengths.last().ok_or("lengths must be nonempty")?;
    if lengths.contains(&0) || lengths.windows(2).any(|w| w[0] > w[1]) {
        return Err("lengths must be positive and ordered coarse to fine".into());
    }
    let train_path = resolve_path(
        root,
        PathBuf::from(
            config
                .get("train_file")
                .and_then(Value::as_str)
                .ok_or("train_file is required")?,
        ),
    );
    let valid_path = resolve_path(
        root,
        PathBuf::from(
            config
                .get("validation_file")
                .and_then(Value::as_str)
                .ok_or("validation_file is required")?,
        ),
    );
    let data_format = config
        .get("data_format")
        .and_then(Value::as_str)
        .unwrap_or("text");
    let (train, valid) = match data_format {
        "text" => (
            raw_dataset(&train_path, sequence_length)?,
            raw_dataset(&valid_path, sequence_length)?,
        ),
        "documents_jsonl" => {
            corpus::validate_disjoint_splits(&[&train_path, &valid_path])
                .map_err(|e| e.to_string())?;
            (
                document_dataset(&train_path, sequence_length)?,
                document_dataset(&valid_path, sequence_length)?,
            )
        }
        "documents_full_windows" => {
            let test_file = config
                .get("test_file")
                .and_then(Value::as_str)
                .ok_or("test_file is required for documents_full_windows")?;
            let test_path = resolve_path(root, PathBuf::from(test_file));
            corpus::validate_disjoint_splits(&[&train_path, &valid_path, &test_path])
                .map_err(|e| e.to_string())?;
            (
                full_document_dataset(&train_path, sequence_length)?,
                full_document_dataset(&valid_path, sequence_length)?,
            )
        }
        _ => return Err(format!("unsupported data_format: {data_format}")),
    };
    if train.sha256 == valid.sha256 {
        return Err("training and validation files must differ".into());
    }
    let batch_size = positive_usize(&config, "batch_size", 8)?;
    let steps = positive_usize(&config, "steps", 1)?;
    let checkpoint_every = positive_usize(&config, "checkpoint_every", 100)?;
    let seed = config.get("seed").and_then(Value::as_u64).unwrap_or(0);
    let rvq = RvqConfig {
        lengths,
        codebook_size: positive_usize(&config, "codebook_size", 8)?,
        dim: positive_usize(&config, "dim", 16)?,
        decay: config.get("decay").and_then(Value::as_f64).unwrap_or(0.99) as f32,
        epsilon: config
            .get("epsilon")
            .and_then(Value::as_f64)
            .unwrap_or(1e-5) as f32,
        entropy_temperature: config
            .get("entropy_temperature")
            .and_then(Value::as_f64)
            .unwrap_or(1.) as f32,
        nested_dropout: config
            .get("nested_dropout")
            .and_then(Value::as_f64)
            .unwrap_or(0.) as f32,
        coarse: config.get("coarse").cloned().unwrap_or_else(|| json!({})),
    };
    let transformer = TransformerConfig {
        width: positive_usize(&config, "width", 256)?,
        heads: positive_usize(&config, "heads", 8)?,
        layers: positive_usize(&config, "layers", 4)?,
        mlp_ratio: positive_usize(&config, "mlp_ratio", 4)?,
        dropout: config.get("dropout").and_then(Value::as_f64).unwrap_or(0.) as f32,
    };
    let model = ModelConfig {
        ar_coarse_levels: config
            .get("ar_coarse_levels")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        prompt_len: config
            .get("prompt_len")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
    };
    let losses = LossConfig {
        beta: config.get("beta").and_then(Value::as_f64).unwrap_or(0.25) as f32,
        entropy: config
            .get("entropy")
            .and_then(Value::as_f64)
            .unwrap_or(0.01) as f32,
        concept: config.get("concept").and_then(Value::as_f64).unwrap_or(0.) as f32,
        prior: config.get("prior").and_then(Value::as_f64).unwrap_or(1.) as f32,
        entropy_weights: Vec::new(),
    };
    let learning_rate = config
        .get("learning_rate")
        .and_then(Value::as_f64)
        .unwrap_or(0.0003);
    if rvq.codebook_size < 2
        || !transformer.width.is_multiple_of(transformer.heads)
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err("invalid codebook size, transformer dimensions, or learning rate".into());
    }
    let engine_config = EngineConfig {
        rvq,
        transformer,
        model,
        losses,
        token_vocab_size: 256,
        learning_rate: learning_rate as f32,
        init_batch_size: batch_size,
        device: json_or(&config, "device", DeviceConfig::default())?,
    };
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let manifest_path = output.join("config.json");
    let checkpoint = output.join("checkpoint.msgpack");
    let metrics_path = output.join("metrics.jsonl");
    let test_identity = config
        .get("test_file")
        .and_then(Value::as_str)
        .map(|path| {
            let path = resolve_path(root, PathBuf::from(path));
            fs::read(path)
                .map(|bytes| corpus::sha256(&bytes))
                .map_err(|e| e.to_string())
        })
        .transpose()?;
    let identity = json!({"config":config,"provenance":crate::validation_io::identity(root)?,"train_sha256":train.sha256,"validation_sha256":valid.sha256,"test_sha256":test_identity,"train_windows":train.tokens.len(),"validation_windows":valid.tokens.len()});
    if manifest_path.exists() {
        let previous: Value =
            serde_json::from_slice(&fs::read(&manifest_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if previous != identity {
            return Err("run configuration or input data changed".into());
        }
    }
    if checkpoint.exists() && !resume {
        return Err("existing checkpoint: use --resume or choose a new output directory".into());
    }
    if resume && !checkpoint.exists() {
        return Err("cannot resume: checkpoint.msgpack is missing".into());
    }
    write_json(&manifest_path, &identity)?;
    let mut engine = EngineClient::start(root)?;
    let initialized = engine.init_model(&InitModelRequest {
        config: engine_config,
        seed: u32::try_from(seed).map_err(|_| "seed must fit the JAX uint32 PRNG key")?,
    })?;
    let mut step = initialized.step;
    if resume && checkpoint.exists() {
        step = engine
            .load_checkpoint(&CheckpointRequest {
                engine_id: initialized.engine_id.clone(),
                path: checkpoint.to_string_lossy().into_owned(),
            })?
            .step;
    }
    for index in step..steps as u64 {
        let mut rng = seed ^ 0xA0761D6478BD642F ^ index.wrapping_mul(0xE7037ED1A0B428DB);
        let indices = choose_indices(batch_size, train.tokens.len(), &mut rng);
        let telemetry = engine.train_step(&TrainStepRequest {
            engine_id: initialized.engine_id.clone(),
            batch: batch_at(&train, &indices),
            seed: (seed.wrapping_add(index + 1) % (u32::MAX as u64 + 1)) as u32,
            phase: 0,
        })?;
        if (index + 1) % checkpoint_every as u64 == 0 || index + 1 == steps as u64 {
            let evaluation = evaluate(&mut engine, &initialized.engine_id, &valid, batch_size)?;
            let temporary = output.join("checkpoint.tmp");
            engine.save_checkpoint(&CheckpointRequest {
                engine_id: initialized.engine_id.clone(),
                path: temporary.to_string_lossy().into_owned(),
            })?;
            fs::rename(&temporary, &checkpoint).map_err(|e| e.to_string())?;
            let record = json!({"step":telemetry.step,"train":{"loss":telemetry.loss,"grad_norm":telemetry.grad_norm,"step_time_ms":telemetry.step_time_ms,"metrics":telemetry.metrics},"validation":evaluation});
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metrics_path)
                .map_err(|e| e.to_string())?;
            writeln!(file, "{}", record).map_err(|e| e.to_string())?;
            println!("{record}");
        }
    }
    Ok(())
}

fn process_gutenberg(raw: &[u8], identifier: u32) -> Result<(String, usize), String> {
    let text = std::str::from_utf8(raw.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(raw))
        .map_err(|e| format!("Gutenberg {identifier} is not UTF-8: {e}"))?
        .replace("\r\n", "\n");
    let mut start = None;
    let mut end = None;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end_matches('\n');
        if start.is_none() && trimmed.starts_with("*** START OF ") && trimmed.contains("***") {
            start = Some(offset + line.len());
        }
        if start.is_some() && trimmed.starts_with("*** END OF ") && trimmed.contains("***") {
            end = Some(offset);
            break;
        }
        offset += line.len();
    }
    let from = start.ok_or_else(|| format!("Gutenberg {identifier} start marker not found"))?;
    let to = end.ok_or_else(|| format!("Gutenberg {identifier} end marker not found"))?;
    if from >= to {
        return Err(format!("invalid Gutenberg boundaries for {identifier}"));
    }
    let body = text[from..to].trim().to_owned();
    if body.is_empty() {
        return Err(format!("Gutenberg {identifier} body is empty"));
    }
    let bytes = body.len();
    Ok((body, bytes))
}

/// Prepare the three Gutenberg splits and train nine Rust-scheduled codecs via the JAX RPC worker.
pub fn run_real_text_reconstruction(
    root: &Path,
    output: &Path,
    steps: usize,
) -> Result<(), String> {
    if steps == 0 {
        return Err("steps must be positive".into());
    }
    let output = if output.is_absolute() {
        output.to_owned()
    } else {
        root.join(output)
    };
    if output.join("protocol.json").exists() {
        return Err("Existing protocol; use a new output directory".into());
    }
    let mut corpus_rows = Vec::new();
    for (split, id) in [("train", 11u32), ("validation", 12), ("test", 13)] {
        let raw_path = output.join("raw").join(format!("{id}.txt"));
        let raw = fs::read(&raw_path).map_err(|e| format!("{}: {e}", raw_path.display()))?;
        let (text, byte_count) = process_gutenberg(&raw, id)?;
        let document = json!({"id":format!("gutenberg-{id}"),"text":text});
        let mut line = serde_json::to_string(&document).map_err(|e| e.to_string())?;
        line.push('\n');
        let processed = output.join(format!("{split}.jsonl"));
        fs::write(&processed, &line).map_err(|e| e.to_string())?;
        corpus_rows.push(json!({"split":split,"id":id,"source":format!("https://www.gutenberg.org/ebooks/{id}"),"raw_sha256":corpus::sha256(&raw),"processed_sha256":corpus::sha256(line.as_bytes()),"bytes":byte_count}));
    }
    let mut engine = EngineClient::start(root)?;
    let engine_info = engine.describe()?;
    drop(engine);
    let identity = crate::validation_io::identity(root)?;
    let mut sources = identity["source_sha256"].clone();
    let workload_path = root.join("rust/src/workloads.rs");
    sources["rust/src/workloads.rs"] = json!(corpus::sha256(
        &fs::read(workload_path).map_err(|e| e.to_string())?
    ));
    let protocol = json!({"corpus":corpus_rows,"lengths":[32,128,512],"seeds":[0,1,2],"steps":steps,"batch_size":4,"sources":sources,"python":engine_info["python"],"versions":{"jax":engine_info["jax"],"jaxlib":engine_info["jaxlib"],"flax":engine_info["flax"],"optax":engine_info["optax"],"numpy":engine_info["numpy"]},"devices":engine_info["devices"],"selection":"Final checkpoint; validation is descriptive; no test reconstruction","limitations":"One book per split, one author, short bounded fits; not a quality or speed comparison"});
    write_json(&output.join("protocol.json"), &protocol)?;
    let mut summary = Vec::new();
    for length in [32usize, 128, 512] {
        for seed in [0u64, 1, 2] {
            let config = json!({"data_format":"documents_full_windows","train_file":output.join("train.jsonl"),"validation_file":output.join("validation.jsonl"),"test_file":output.join("test.jsonl"),"lengths":[length/4,length/2,length],"codebook_size":16,"dim":16,"width":16,"heads":2,"layers":1,"batch_size":4,"seed":seed,"learning_rate":0.003,"steps":steps,"checkpoint_every":steps,"entropy":0.0,"prior":0.0});
            let config_path = output.join(format!("config-length-{length}-seed-{seed}.json"));
            write_json(&config_path, &config)?;
            let destination = output.join(format!("length-{length}-seed-{seed}"));
            let started = Instant::now();
            train_codec(root, &config_path, &destination, false)?;
            let elapsed = started.elapsed().as_secs_f64();
            let metrics =
                fs::read_to_string(destination.join("metrics.jsonl")).map_err(|e| e.to_string())?;
            let record = metrics
                .lines()
                .last()
                .ok_or("training produced no metric record")?;
            let record: Value = serde_json::from_str(record).map_err(|e| e.to_string())?;
            summary.push(json!({"length":length,"seed":seed,"elapsed_seconds":elapsed,"training_byte_exposure":steps*4*length,"metrics":record}));
            write_json(&output.join("summary.json"), &Value::Array(summary.clone()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::process_gutenberg;

    #[test]
    fn strips_gutenberg_headers_and_normalizes_line_endings() {
        let input = b"\xef\xbb\xbfHeader\r\n*** START OF THE BOOK ***\r\n\r\nText\r\n*** END OF THE BOOK ***\r\nFooter\r\n";
        let (body, bytes) = process_gutenberg(input, 11).unwrap();
        assert_eq!(body, "Text");
        assert_eq!(bytes, 4);
    }

    #[test]
    fn rejects_missing_gutenberg_boundaries() {
        assert!(process_gutenberg(b"plain text", 13).is_err());
    }
}
