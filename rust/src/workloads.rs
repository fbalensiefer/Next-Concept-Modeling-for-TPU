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
    let identity = json!({"config":config,"provenance":crate::validation_io::identity(root)?,"train_sha256":train.sha256,"validation_sha256":valid.sha256,"train_windows":train.tokens.len(),"validation_windows":valid.tokens.len()});
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
