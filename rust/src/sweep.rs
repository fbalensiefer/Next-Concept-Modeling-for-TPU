//! Typed multi-arm training, held-out quality evaluation, and checkpoint audits.
use crate::{
    corpus,
    rpc::{
        CausalArConfig, CausalArEvaluateRequest, CausalArInitRequest, CausalArTrainStepRequest,
        CheckpointRequest, EngineClient, EngineConfig, ForwardRequest,
        GenerateAndVerifyAtomicRequest, GenerateRequest, InitModelRequest, TensorBatch,
        TrainStepRequest,
    },
    validation_io,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepConfig {
    #[serde(default)]
    pub profile: Option<String>,
    pub train_file: PathBuf,
    pub validation_file: PathBuf,
    pub sequence_length: usize,
    pub max_prompt_len: usize,
    pub seeds: Vec<u32>,
    pub prompt_lengths: Vec<usize>,
    pub steps: usize,
    pub checkpoint_steps: Vec<usize>,
    pub batch_size: usize,
    #[serde(default)]
    pub train_windows: Vec<usize>,
    #[serde(default)]
    pub validation_windows: Vec<usize>,
    #[serde(default)]
    pub codec_pretrain: Option<CodecPretrainConfig>,
    #[serde(default)]
    pub teacher: Option<CausalArConfig>,
    #[serde(default)]
    pub benchmark: Option<DrafterBenchmarkConfig>,
    pub arms: Vec<SweepArm>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecPretrainConfig {
    pub config: Box<EngineConfig>,
    pub steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrafterBenchmarkConfig {
    pub target_arm: String,
    pub drafters: Vec<DrafterArm>,
    #[serde(default = "default_draft_lengths")]
    pub draft_lengths: Vec<usize>,
    #[serde(default = "default_batch_sizes")]
    pub batch_sizes: Vec<usize>,
    #[serde(default = "default_warmups")]
    pub warmups: usize,
    #[serde(default = "default_repetitions")]
    pub repetitions: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: usize,
}

fn default_draft_lengths() -> Vec<usize> {
    vec![2, 4, 8, 16]
}
fn default_batch_sizes() -> Vec<usize> {
    vec![1, 16, 128]
}
fn default_warmups() -> usize {
    5
}
fn default_repetitions() -> usize {
    30
}
fn default_temperature() -> f32 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrafterArm {
    pub label: String,
    pub arm: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrafterTimingOptions {
    pub draft_lengths: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub warmups: usize,
    pub repetitions: usize,
    pub temperature: f32,
    pub top_k: usize,
}

impl Default for DrafterTimingOptions {
    fn default() -> Self {
        Self {
            draft_lengths: vec![2, 4, 8, 16],
            batch_sizes: vec![1, 16, 128],
            warmups: 5,
            repetitions: 30,
            temperature: 1.0,
            top_k: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SweepArm {
    pub name: String,
    #[serde(default)]
    pub distillation_weight: f32,
    #[serde(default)]
    pub training_mode: TrainingMode,
    #[serde(flatten)]
    pub model: SweepModel,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    #[default]
    Joint,
    PriorOnly,
    SequenceDistillation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "snake_case")]
pub enum SweepModel {
    Ncm(Box<EngineConfig>),
    CausalAr(CausalArConfig),
}

#[derive(Clone, Debug)]
struct Dataset {
    tokens: Vec<Vec<i32>>,
    masks: Vec<Vec<bool>>,
    sha256: String,
}

#[derive(Clone)]
struct CheckpointRow {
    seed: u32,
    prompt_len: usize,
    arm: String,
    kind: String,
    step: usize,
    path: String,
}

struct Session {
    engine_id: String,
    model: SweepModel,
}

fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

fn validate_output_location(root: &Path, config: &SweepConfig, path: &Path) -> Result<(), String> {
    if config.profile.as_deref() != Some("real128") {
        return Ok(());
    }
    let expected = root.join("reports/validation-controls/real128-rust-native");
    if resolve(root, path) != expected {
        return Err(format!(
            "real128 artifacts are restricted to {}",
            expected.display()
        ));
    }
    Ok(())
}

fn load_dataset(path: &Path, width: usize) -> Result<Dataset, String> {
    if width == 0 {
        return Err("sequence_length plus max_prompt_len must be positive".into());
    }
    if path.extension().is_some_and(|x| x == "jsonl") {
        let windows =
            corpus::document_masked_byte_windows(path, width).map_err(|e| e.to_string())?;
        if windows.rows.is_empty() {
            return Err(format!("no usable data windows in {}", path.display()));
        }
        return Ok(Dataset {
            tokens: windows
                .rows
                .into_iter()
                .map(|r| r.into_iter().map(i32::from).collect())
                .collect(),
            masks: windows.masks,
            sha256: windows.sha256,
        });
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.is_empty() {
        return Err(format!("empty data file: {}", path.display()));
    }
    let mut tokens = Vec::new();
    let mut masks = Vec::new();
    for chunk in bytes.chunks(width) {
        let mut row = vec![0; width];
        let mut mask = vec![false; width];
        for (i, byte) in chunk.iter().enumerate() {
            row[i] = i32::from(*byte);
            mask[i] = true;
        }
        tokens.push(row);
        masks.push(mask);
    }
    Ok(Dataset {
        tokens,
        masks,
        sha256: corpus::sha256(&bytes),
    })
}

fn select_windows(data: &Dataset, indices: &[usize], label: &str) -> Result<Dataset, String> {
    if indices.is_empty() {
        return Ok(data.clone());
    }
    if indices.iter().copied().collect::<BTreeSet<_>>().len() != indices.len() {
        return Err(format!("{label} window indices must be unique"));
    }
    let mut tokens = Vec::with_capacity(indices.len());
    let mut masks = Vec::with_capacity(indices.len());
    for index in indices {
        tokens.push(
            data.tokens
                .get(*index)
                .ok_or_else(|| format!("{label} window index {index} is outside the dataset"))?
                .clone(),
        );
        masks.push(data.masks[*index].clone());
    }
    Ok(Dataset {
        tokens,
        masks,
        sha256: data.sha256.clone(),
    })
}

fn validate_profile_windows(
    config: &SweepConfig,
    data: &Dataset,
    label: &str,
) -> Result<(), String> {
    if config.profile.as_deref() != Some("real128") {
        return Ok(());
    }
    if data.tokens.len() != 32
        || data.masks.iter().any(|mask| {
            mask.get(..config.max_prompt_len + 1)
                .is_none_or(|context| context.iter().any(|valid| !valid))
        })
    {
        return Err(format!(
            "real128 {label} windows must be the 32 fully eligible historical windows"
        ));
    }
    Ok(())
}

fn validate_config(config: &SweepConfig) -> Result<(), String> {
    if config.sequence_length == 0 || config.steps == 0 || config.batch_size == 0 {
        return Err("sequence_length, steps, and batch_size must be positive".into());
    }
    if config
        .max_prompt_len
        .checked_add(config.sequence_length)
        .is_none()
    {
        return Err("sequence_length plus max_prompt_len overflows".into());
    }
    if config.seeds.is_empty() || config.prompt_lengths.is_empty() || config.arms.is_empty() {
        return Err("seeds, prompt_lengths, and arms must be nonempty".into());
    }
    if config.seeds.iter().copied().collect::<BTreeSet<_>>().len() != config.seeds.len()
        || config
            .prompt_lengths
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != config.prompt_lengths.len()
    {
        return Err("seeds and prompt_lengths must be unique".into());
    }
    if config
        .prompt_lengths
        .iter()
        .any(|p| *p > config.max_prompt_len)
    {
        return Err("prompt length exceeds max_prompt_len".into());
    }
    for (name, windows) in [
        ("train", &config.train_windows),
        ("validation", &config.validation_windows),
    ] {
        if windows.iter().copied().collect::<BTreeSet<_>>().len() != windows.len() {
            return Err(format!("{name} window indices must be unique"));
        }
    }
    if config.checkpoint_steps.is_empty()
        || config.checkpoint_steps.windows(2).any(|w| w[0] >= w[1])
        || config.checkpoint_steps.last().copied() != Some(config.steps)
    {
        return Err("checkpoint_steps must be strictly increasing and end at steps".into());
    }
    let names: BTreeSet<_> = config.arms.iter().map(|a| a.name.as_str()).collect();
    if names.len() != config.arms.len() || config.arms.iter().any(|a| a.name.trim().is_empty()) {
        return Err("arm names must be nonempty and unique".into());
    }
    for arm in &config.arms {
        if !arm.distillation_weight.is_finite() || !(0.0..=1.0).contains(&arm.distillation_weight) {
            return Err(format!(
                "arm {} has an invalid distillation weight",
                arm.name
            ));
        }
        if arm.distillation_weight > 0.0
            && (!matches!(arm.model, SweepModel::CausalAr(_)) || config.teacher.is_none())
        {
            return Err(format!(
                "arm {} requires a causal-AR teacher for distillation",
                arm.name
            ));
        }
        if arm.training_mode != TrainingMode::Joint && !matches!(arm.model, SweepModel::Ncm(_)) {
            return Err(format!(
                "specialized training mode for {} requires an NCM model",
                arm.name
            ));
        }
        match &arm.model {
            SweepModel::Ncm(model) => {
                if model.rvq.lengths.last().copied() != Some(config.sequence_length) {
                    return Err(format!(
                        "NCM arm {} max RVQ length must equal sequence_length",
                        arm.name
                    ));
                }
            }
            SweepModel::CausalAr(model) => {
                if model.max_length < config.sequence_length + config.max_prompt_len {
                    return Err(format!(
                        "AR arm {} max_length must cover prompt plus sequence",
                        arm.name
                    ));
                }
            }
        }
    }
    if let Some(codec) = &config.codec_pretrain {
        if codec.steps == 0
            || codec.config.rvq.lengths.last().copied() != Some(config.sequence_length)
        {
            return Err(
                "codec pretraining requires positive steps and matching sequence bucket".into(),
            );
        }
        for arm in &config.arms {
            if let SweepModel::Ncm(model) = &arm.model
                && (model.rvq.lengths != codec.config.rvq.lengths
                    || model.rvq.codebook_size != codec.config.rvq.codebook_size
                    || model.rvq.dim != codec.config.rvq.dim
                    || model.transformer.width != codec.config.transformer.width
                    || model.transformer.heads != codec.config.transformer.heads
                    || model.transformer.layers != codec.config.transformer.layers
                    || model.transformer.mlp_ratio != codec.config.transformer.mlp_ratio)
            {
                return Err(format!(
                    "NCM arm {} is incompatible with the shared frozen codec",
                    arm.name
                ));
            }
        }
    }
    if let Some(benchmark) = &config.benchmark {
        if benchmark.target_arm.trim().is_empty()
            || benchmark.drafters.is_empty()
            || benchmark
                .drafters
                .iter()
                .any(|d| d.label.trim().is_empty() || d.arm.trim().is_empty())
            || benchmark
                .drafters
                .iter()
                .map(|d| d.label.as_str())
                .collect::<BTreeSet<_>>()
                .len()
                != benchmark.drafters.len()
        {
            return Err("benchmark target and drafter labels must be nonempty and unique".into());
        }
        if !config.arms.iter().any(|a| a.name == benchmark.target_arm) && config.teacher.is_none() {
            return Err("benchmark target is neither a sweep arm nor a configured teacher".into());
        }
        if benchmark
            .drafters
            .iter()
            .any(|d| !names.contains(d.arm.as_str()))
        {
            return Err("benchmark drafter references an unknown arm".into());
        }
    }
    if config.profile.as_deref() == Some("real128") {
        let expected_train = (0..31).chain([786]).collect::<Vec<_>>();
        let expected_valid = (0..31).chain([899]).collect::<Vec<_>>();
        let expected_arms = [
            "ar_scratch",
            "ar_distilled",
            "ncm_coarse_0",
            "ncm_coarse_1",
            "ncm_sequence_distilled",
        ];
        let expected_drafters = ["ncm", "ar_distilled", "ncm_sequence_distilled"];
        let benchmark = config
            .benchmark
            .as_ref()
            .ok_or("real128 profile requires benchmark config")?;
        if config.seeds != [200, 201, 202]
            || config.prompt_lengths != [0, 16, 64]
            || config.sequence_length != 128
            || config.max_prompt_len != 64
            || config.steps != 8
            || config.checkpoint_steps != [2, 4, 8]
            || config.batch_size != 4
            || config.train_windows != expected_train
            || config.validation_windows != expected_valid
            || config
                .arms
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                != expected_arms
            || config.teacher.is_none()
            || config
                .teacher
                .as_ref()
                .is_none_or(|teacher| teacher.max_length < 209)
            || config.codec_pretrain.as_ref().is_none_or(|c| c.steps != 8)
            || benchmark.target_arm != "teacher"
            || benchmark
                .drafters
                .iter()
                .map(|d| d.label.as_str())
                .collect::<Vec<_>>()
                != expected_drafters
            || benchmark.draft_lengths != [2, 4, 8, 16]
            || benchmark.batch_sizes != [1, 16, 128]
            || benchmark.warmups != 5
            || benchmark.repetitions != 30
        {
            return Err("real128 profile differs from the versioned historical grid".into());
        }
        if config
            .arms
            .iter()
            .find(|a| a.name == "ar_distilled")
            .is_none_or(|a| (a.distillation_weight - 0.5).abs() > f32::EPSILON)
            || config
                .arms
                .iter()
                .find(|a| a.name == "ncm_coarse_0")
                .is_none_or(|a| a.training_mode != TrainingMode::PriorOnly)
            || config
                .arms
                .iter()
                .find(|a| a.name == "ncm_coarse_1")
                .is_none_or(|a| a.training_mode != TrainingMode::PriorOnly)
            || config
                .arms
                .iter()
                .find(|a| a.name == "ncm_sequence_distilled")
                .is_none_or(|a| a.training_mode != TrainingMode::SequenceDistillation)
        {
            return Err(
                "real128 profile is missing the historical distillation/training modes".into(),
            );
        }
    }
    Ok(())
}

fn session(
    engine: &mut EngineClient,
    arm: &SweepArm,
    seed: u32,
    prompt_len: usize,
    batch: usize,
) -> Result<Session, String> {
    match &arm.model {
        SweepModel::Ncm(source) => {
            let mut config = source.as_ref().clone();
            config.model.prompt_len = prompt_len;
            config.init_batch_size = batch;
            let handle = engine.init_model(&InitModelRequest {
                config: config.clone(),
                seed,
            })?;
            Ok(Session {
                engine_id: handle.engine_id,
                model: SweepModel::Ncm(Box::new(config)),
            })
        }
        SweepModel::CausalAr(config) => {
            let handle = engine.causal_ar_init(&CausalArInitRequest {
                config: config.clone(),
                seed,
                tokens: vec![vec![0]],
                mask: None,
            })?;
            Ok(Session {
                engine_id: handle.engine_id,
                model: SweepModel::CausalAr(config.clone()),
            })
        }
    }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn sample_window_ids(length: usize, batch: usize, rng: &mut u64) -> Vec<usize> {
    (0..batch)
        .map(|_| ((xorshift(rng) as u128 * length as u128) >> 64) as usize)
        .collect()
}

fn selected_batch(
    data: &Dataset,
    ids: &[usize],
    config: &SweepConfig,
    prompt_len: usize,
    ncm: bool,
) -> TensorBatch {
    let start = config.max_prompt_len - prompt_len;
    let end = config.max_prompt_len + config.sequence_length;
    let (tokens, masks): (Vec<_>, Vec<_>) = ids
        .iter()
        .map(|i| {
            let row = &data.tokens[*i];
            let mask = &data.masks[*i];
            if ncm {
                (
                    row[config.max_prompt_len..end].to_vec(),
                    mask[config.max_prompt_len..end].to_vec(),
                )
            } else {
                (row[start..end].to_vec(), mask[start..end].to_vec())
            }
        })
        .unzip();
    let prompt_ids = if ncm && prompt_len > 0 {
        Some(
            ids.iter()
                .map(|i| data.tokens[*i][start..config.max_prompt_len].to_vec())
                .collect(),
        )
    } else {
        None
    };
    let prompt_mask = prompt_ids
        .as_ref()
        .map(|rows: &Vec<Vec<i32>>| rows.iter().map(|r| vec![true; r.len()]).collect());
    TensorBatch {
        tokens,
        mask: Some(masks),
        fallback_bytes: None,
        fallback_mask: None,
        prompt_ids,
        prompt_mask,
    }
}

fn write_rows(path: &Path, root: &Path, rows: &[Map<String, Value>]) -> Result<(), String> {
    validation_io::write_csv(path, root, rows)
}

pub fn train_sweep(root: &Path, config_path: &Path, output_path: &Path) -> Result<(), String> {
    let config_path = resolve(root, config_path);
    let config: SweepConfig =
        serde_json::from_slice(&fs::read(&config_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid sweep config: {e}"))?;
    validate_config(&config)?;
    validate_output_location(root, &config, output_path)?;
    let train = select_windows(
        &load_dataset(
            &resolve(root, &config.train_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.train_windows,
        "training",
    )?;
    let valid = select_windows(
        &load_dataset(
            &resolve(root, &config.validation_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.validation_windows,
        "validation",
    )?;
    validate_profile_windows(&config, &train, "training")?;
    validate_profile_windows(&config, &valid, "validation")?;
    let output = resolve(root, output_path);
    if output.exists()
        && fs::read_dir(&output)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err(format!(
            "output directory is not empty: {}",
            output.display()
        ));
    }
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let mut checkpoints = Vec::<CheckpointRow>::new();
    let mut teacher_checkpoints = Vec::<Value>::new();
    let mut training_rows = Vec::<Map<String, Value>>::new();
    let started_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let mut runtime = None;
    let mut execution_log =
        fs::File::create(output.join("execution.jsonl")).map_err(|e| e.to_string())?;
    for seed in &config.seeds {
        let codec_path = if let Some(codec) = &config.codec_pretrain {
            let mut engine = EngineClient::start(root)?;
            let mut codec_config = codec.config.as_ref().clone();
            codec_config.model.prompt_len = 0;
            let codec_arm = SweepArm {
                name: "codec_pretrain".into(),
                distillation_weight: 0.0,
                training_mode: TrainingMode::Joint,
                model: SweepModel::Ncm(Box::new(codec_config.clone())),
            };
            let model = session(&mut engine, &codec_arm, *seed, 0, config.batch_size)?;
            let mut rng = (u64::from(*seed) << 32) ^ 0xC0DEC0DE;
            for step in 1..=codec.steps {
                let ids = sample_window_ids(train.tokens.len(), config.batch_size, &mut rng);
                let batch = selected_batch(&train, &ids, &config, 0, true);
                engine.train_step(&TrainStepRequest {
                    engine_id: model.engine_id.clone(),
                    batch,
                    seed: seed.wrapping_add(step as u32),
                    phase: 0,
                    prior_only: false,
                    sequence_distillation: false,
                })?;
            }
            let relative = PathBuf::from(format!("seed-{seed}/codec.msgpack"));
            engine.save_codec(&CheckpointRequest {
                engine_id: model.engine_id,
                path: output.join(&relative).to_string_lossy().into_owned(),
            })?;
            Some(relative)
        } else {
            None
        };

        let teacher_path = if let Some(teacher_config) = &config.teacher {
            let mut engine = EngineClient::start(root)?;
            let teacher = engine.causal_ar_init(&CausalArInitRequest {
                config: teacher_config.clone(),
                seed: *seed,
                tokens: vec![vec![0]],
                mask: None,
            })?;
            let mut rng = (u64::from(*seed) << 32) ^ 0x7EAC_4E12;
            let mut losses = Vec::new();
            for step in 1..=config.steps {
                let ids = sample_window_ids(train.tokens.len(), config.batch_size, &mut rng);
                let batch = selected_batch(&train, &ids, &config, config.max_prompt_len, false);
                let response = engine.causal_ar_train_step(&CausalArTrainStepRequest {
                    engine_id: teacher.engine_id.clone(),
                    tokens: batch.tokens,
                    mask: batch.mask,
                    continuation_start: config.max_prompt_len,
                    teacher_engine_id: None,
                    distillation_weight: 0.0,
                })?;
                losses.push(json!({"step":step,"loss":response.loss,"grad_norm":response.grad_norm,"step_time_ms":response.step_time_ms}));
            }
            let relative = PathBuf::from(format!("seed-{seed}/teacher.msgpack"));
            let absolute = output.join(&relative);
            engine.save_checkpoint(&CheckpointRequest {
                engine_id: teacher.engine_id,
                path: absolute.to_string_lossy().into_owned(),
            })?;
            teacher_checkpoints.push(json!({"seed":seed,"step":config.steps,"path":relative,"sha256":corpus::sha256(&fs::read(&absolute).map_err(|e| e.to_string())?)}));
            writeln!(
                execution_log,
                "{}",
                json!({"seed":seed,"arm":"teacher","training":losses})
            )
            .map_err(|e| e.to_string())?;
            Some(relative)
        } else {
            None
        };

        let mut sequence_samples = std::collections::BTreeMap::<usize, Dataset>::new();
        if config
            .arms
            .iter()
            .any(|arm| arm.training_mode == TrainingMode::SequenceDistillation)
        {
            let teacher_config = config
                .teacher
                .as_ref()
                .ok_or("sequence distillation requires a teacher")?;
            let teacher_path = teacher_path.as_ref().ok_or("teacher checkpoint missing")?;
            for prompt_len in &config.prompt_lengths {
                let mut engine = EngineClient::start(root)?;
                let teacher = engine.causal_ar_init(&CausalArInitRequest {
                    config: teacher_config.clone(),
                    seed: *seed,
                    tokens: vec![vec![0]],
                    mask: None,
                })?;
                engine.load_checkpoint(&CheckpointRequest {
                    engine_id: teacher.engine_id.clone(),
                    path: output.join(teacher_path).to_string_lossy().into_owned(),
                })?;
                let prompts = train
                    .tokens
                    .iter()
                    .map(|row| {
                        row[config.max_prompt_len - *prompt_len..config.max_prompt_len].to_vec()
                    })
                    .collect::<Vec<_>>();
                let generated = engine.generate(&GenerateRequest {
                    engine_id: teacher.engine_id,
                    prompts: prompts.clone(),
                    max_new_tokens: config.sequence_length,
                    temperature: 1.0,
                    top_k: 0,
                    seed: derive_seed(
                        *seed,
                        *prompt_len,
                        train.tokens.len(),
                        config.sequence_length,
                        0,
                        0,
                        41,
                    ),
                })?;
                let mut tokens = Vec::with_capacity(prompts.len());
                let mut masks = Vec::with_capacity(prompts.len());
                for (prompt, continuation) in prompts.into_iter().zip(generated.tokens) {
                    let mut row = vec![0; config.max_prompt_len + config.sequence_length];
                    let mut mask = vec![false; row.len()];
                    let prompt_start = config.max_prompt_len - *prompt_len;
                    row[prompt_start..config.max_prompt_len].copy_from_slice(&prompt);
                    mask[prompt_start..config.max_prompt_len].fill(true);
                    row[config.max_prompt_len..].copy_from_slice(&continuation);
                    mask[config.max_prompt_len..].fill(true);
                    tokens.push(row);
                    masks.push(mask);
                }
                let sample_hash = corpus::sha256(
                    &tokens
                        .iter()
                        .flatten()
                        .map(|x| *x as u8)
                        .collect::<Vec<_>>(),
                );
                let sample_path = output.join(format!(
                    "seed-{seed}/prompt-{prompt_len}/teacher_samples.json"
                ));
                validation_io::write_json(
                    &sample_path,
                    root,
                    &json!({"seed":seed,"prompt_len":prompt_len,"teacher_checkpoint_sha256":teacher_checkpoints.last().and_then(|r| r["sha256"].as_str()),"sample_sha256":sample_hash,"rows":tokens.iter().zip(&masks).map(|(t,m)| json!({"tokens":t,"mask":m})).collect::<Vec<_>>()}),
                )?;
                sequence_samples.insert(
                    *prompt_len,
                    Dataset {
                        tokens,
                        masks,
                        sha256: sample_hash,
                    },
                );
            }
        }

        for prompt_len in &config.prompt_lengths {
            for arm in &config.arms {
                let run_dir = output
                    .join(format!("seed-{seed}"))
                    .join(format!("prompt-{prompt_len}"))
                    .join(&arm.name);
                fs::create_dir_all(&run_dir).map_err(|e| e.to_string())?;
                let mut engine = EngineClient::start(root)?;
                if runtime.is_none() {
                    runtime = Some(engine.describe()?);
                }
                let current = session(&mut engine, arm, *seed, *prompt_len, config.batch_size)?;
                let ncm = matches!(current.model, SweepModel::Ncm(_));
                let teacher_engine_id = if arm.distillation_weight > 0.0 {
                    let teacher_config = config.teacher.as_ref().ok_or("teacher config missing")?;
                    let teacher_path = teacher_path.as_ref().ok_or("teacher checkpoint missing")?;
                    let teacher = engine.causal_ar_init(&CausalArInitRequest {
                        config: teacher_config.clone(),
                        seed: *seed,
                        tokens: vec![vec![0]],
                        mask: None,
                    })?;
                    engine.load_checkpoint(&CheckpointRequest {
                        engine_id: teacher.engine_id.clone(),
                        path: output.join(teacher_path).to_string_lossy().into_owned(),
                    })?;
                    Some(teacher.engine_id)
                } else {
                    None
                };
                if ncm && let Some(codec_path) = &codec_path {
                    engine.load_codec(&CheckpointRequest {
                        engine_id: current.engine_id.clone(),
                        path: output.join(codec_path).to_string_lossy().into_owned(),
                    })?;
                }
                let mut rng =
                    (u64::from(*seed) << 32) ^ (prompt_len << 1) as u64 ^ 0xA0761D6478BD642F;
                let mut total_loss = 0.0;
                let mut total_step_ms = 0.0;
                let mut saved = BTreeSet::new();
                for step in 1..=config.steps {
                    let source = if arm.training_mode == TrainingMode::SequenceDistillation {
                        sequence_samples
                            .get(prompt_len)
                            .ok_or("teacher samples missing")?
                    } else {
                        &train
                    };
                    let ids = sample_window_ids(source.tokens.len(), config.batch_size, &mut rng);
                    let batch = selected_batch(source, &ids, &config, *prompt_len, ncm);
                    let (loss, grad_norm, step_ms) = match &current.model {
                        SweepModel::Ncm(_) => {
                            let response = engine.train_step(&TrainStepRequest {
                                engine_id: current.engine_id.clone(),
                                batch,
                                seed: seed.wrapping_add(step as u32),
                                phase: 0,
                                prior_only: arm.training_mode == TrainingMode::PriorOnly,
                                sequence_distillation: arm.training_mode
                                    == TrainingMode::SequenceDistillation,
                            })?;
                            (response.loss, response.grad_norm, response.step_time_ms)
                        }
                        SweepModel::CausalAr(_) => {
                            let response =
                                engine.causal_ar_train_step(&CausalArTrainStepRequest {
                                    engine_id: current.engine_id.clone(),
                                    tokens: batch.tokens,
                                    mask: batch.mask,
                                    continuation_start: *prompt_len,
                                    teacher_engine_id: teacher_engine_id.clone(),
                                    distillation_weight: arm.distillation_weight,
                                })?;
                            (response.loss, response.grad_norm, response.step_time_ms)
                        }
                    };
                    if !loss.is_finite() || !grad_norm.is_finite() || !step_ms.is_finite() {
                        return Err(format!(
                            "non-finite telemetry for {} seed {seed} prompt {prompt_len} step {step}",
                            arm.name
                        ));
                    }
                    total_loss += loss;
                    total_step_ms += step_ms;
                    let record = json!({"seed":seed,"prompt_len":prompt_len,"arm":arm.name,"step":step,"loss":loss,"grad_norm":grad_norm,"step_time_ms":step_ms});
                    writeln!(execution_log, "{record}").map_err(|e| e.to_string())?;
                    if config.checkpoint_steps.binary_search(&step).is_ok() {
                        let relative = PathBuf::from(format!(
                            "seed-{seed}/prompt-{prompt_len}/{}/step-{step}.msgpack",
                            arm.name
                        ));
                        let absolute = output.join(&relative);
                        engine.save_checkpoint(&CheckpointRequest {
                            engine_id: current.engine_id.clone(),
                            path: absolute.to_string_lossy().into_owned(),
                        })?;
                        checkpoints.push(CheckpointRow {
                            seed: *seed,
                            prompt_len: *prompt_len,
                            arm: arm.name.clone(),
                            kind: if ncm { "ncm" } else { "causal_ar" }.into(),
                            step,
                            path: relative.to_string_lossy().into_owned(),
                        });
                        saved.insert(step);
                    }
                }
                if saved.len() != config.checkpoint_steps.len() {
                    return Err("checkpoint schedule did not save every configured step".into());
                }
                let mut row = Map::new();
                row.insert("seed".into(), json!(seed));
                row.insert("prompt_len".into(), json!(prompt_len));
                row.insert("arm".into(), json!(arm.name));
                row.insert("steps".into(), json!(config.steps));
                row.insert(
                    "mean_train_loss".into(),
                    json!(total_loss / config.steps as f64),
                );
                row.insert("total_step_time_ms".into(), json!(total_step_ms));
                row.insert("train_windows".into(), json!(train.tokens.len()));
                row.insert("validation_windows".into(), json!(valid.tokens.len()));
                training_rows.push(row);
            }
        }
    }
    let checkpoint_values = checkpoints.iter().map(|r| json!({"seed":r.seed,"prompt_len":r.prompt_len,"arm":r.arm,"kind":r.kind,"step":r.step,"path":r.path})).collect::<Vec<_>>();
    let training_values = training_rows
        .iter()
        .map(|r| Value::Object(r.clone()))
        .collect::<Vec<_>>();
    let manifest = json!({"schema":"ncm-rust-sweep-v1","started_at_unix_ms":started_at_unix_ms,"runtime":runtime,"config":config,"train_sha256":train.sha256,"validation_sha256":valid.sha256,"teacher_checkpoints":teacher_checkpoints,"training_rows":training_values,"checkpoints":checkpoint_values,"expected_training_rows":config.seeds.len()*config.prompt_lengths.len()*config.arms.len(),"expected_checkpoint_rows":config.seeds.len()*config.prompt_lengths.len()*config.arms.len()*config.checkpoint_steps.len()});
    validation_io::write_json(output.join("sweep.json"), root, &manifest)?;
    write_rows(&output.join("training_rows.csv"), root, &training_rows)?;
    Ok(())
}

fn load_manifest(
    root: &Path,
    sweep_dir: &Path,
) -> Result<(SweepConfig, Vec<CheckpointRow>), String> {
    let path = resolve(root, sweep_dir).join("sweep.json");
    let value: Value = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let config: SweepConfig = serde_json::from_value(
        value
            .get("config")
            .cloned()
            .ok_or("sweep manifest has no config")?,
    )
    .map_err(|e| e.to_string())?;
    validate_config(&config)?;
    let checkpoints = value
        .get("checkpoints")
        .and_then(Value::as_array)
        .ok_or("sweep manifest has no checkpoint list")?
        .iter()
        .map(|row| {
            Ok(CheckpointRow {
                seed: row["seed"]
                    .as_u64()
                    .ok_or("checkpoint seed must be an integer")? as u32,
                prompt_len: row["prompt_len"]
                    .as_u64()
                    .ok_or("checkpoint prompt_len must be an integer")?
                    as usize,
                arm: row["arm"]
                    .as_str()
                    .ok_or("checkpoint arm must be a string")?
                    .to_owned(),
                kind: row["kind"]
                    .as_str()
                    .ok_or("checkpoint kind must be a string")?
                    .to_owned(),
                step: row["step"]
                    .as_u64()
                    .ok_or("checkpoint step must be an integer")? as usize,
                path: row["path"]
                    .as_str()
                    .ok_or("checkpoint path must be a string")?
                    .to_owned(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((config, checkpoints))
}

fn evaluate_session(
    engine: &mut EngineClient,
    current: &Session,
    data: &Dataset,
    config: &SweepConfig,
    prompt_len: usize,
) -> Result<(f64, String, usize), String> {
    let mut total_nll = 0.0;
    let mut total_tokens = 0usize;
    let mut recon_total = 0.0;
    let mut prior_total = 0.0;
    let mut reconstruction_weight = 0usize;
    for start in (0..data.tokens.len()).step_by(config.batch_size) {
        let ids = (start..(start + config.batch_size).min(data.tokens.len())).collect::<Vec<_>>();
        let batch = selected_batch(
            data,
            &ids,
            config,
            prompt_len,
            matches!(current.model, SweepModel::Ncm(_)),
        );
        match &current.model {
            SweepModel::CausalAr(_) => {
                let response = engine.causal_ar_evaluate(&CausalArEvaluateRequest {
                    engine_id: current.engine_id.clone(),
                    tokens: batch.tokens,
                    mask: batch.mask,
                    prompt_len,
                })?;
                total_nll += response.nll;
                total_tokens += response.token_count;
            }
            SweepModel::Ncm(_) => {
                let valid = batch
                    .mask
                    .as_ref()
                    .ok_or("NCM evaluation mask absent")?
                    .iter()
                    .map(|m| m.iter().filter(|v| **v).count())
                    .sum::<usize>();
                if valid == 0 {
                    continue;
                }
                let response = engine.forward(&ForwardRequest {
                    engine_id: current.engine_id.clone(),
                    batch,
                })?;
                let reconstruction = response
                    .metrics
                    .get("reconstruction")
                    .and_then(Value::as_f64)
                    .ok_or("NCM forward metrics lack reconstruction loss")?;
                let prior = response
                    .metrics
                    .get("prior")
                    .and_then(Value::as_f64)
                    .ok_or("NCM forward metrics lack prior loss")?;
                let batch_codes = count_valid_codes(&response.codes);
                if batch_codes == 0 {
                    return Err("NCM forward returned no valid code positions".into());
                }
                recon_total += reconstruction * valid as f64;
                prior_total += prior * batch_codes as f64;
                reconstruction_weight += valid;
                total_tokens += valid;
            }
        }
    }
    if total_tokens == 0 {
        return Err("validation split has no valid continuation tokens".into());
    }
    match &current.model {
        SweepModel::CausalAr(_) => Ok((
            total_nll / total_tokens as f64 / std::f64::consts::LN_2,
            "exact_bpb".into(),
            total_tokens,
        )),
        SweepModel::Ncm(_) => {
            let nats = recon_total + prior_total;
            Ok((
                nats / reconstruction_weight as f64 / std::f64::consts::LN_2,
                "upper_bound_bpb".into(),
                total_tokens,
            ))
        }
    }
}

fn count_valid_codes(value: &Value) -> usize {
    match value {
        Value::Number(number) => usize::from(number.as_i64().is_some_and(|x| x >= 0)),
        Value::Array(values) => values.iter().map(count_valid_codes).sum(),
        _ => 0,
    }
}

fn rows_to_csv(path: &Path, root: &Path, rows: &[Value]) -> Result<(), String> {
    let maps = rows
        .iter()
        .map(|r| {
            r.as_object()
                .cloned()
                .ok_or("report row must be an object".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    validation_io::write_csv(path, root, &maps)
}

pub fn quality_eval(root: &Path, sweep_dir: &Path) -> Result<(), String> {
    let output = resolve(root, sweep_dir);
    let (config, checkpoints) = load_manifest(root, sweep_dir)?;
    validate_output_location(root, &config, sweep_dir)?;
    let train = select_windows(
        &load_dataset(
            &resolve(root, &config.train_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.train_windows,
        "training",
    )?;
    let valid = select_windows(
        &load_dataset(
            &resolve(root, &config.validation_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.validation_windows,
        "validation",
    )?;
    validate_profile_windows(&config, &train, "training")?;
    validate_profile_windows(&config, &valid, "validation")?;
    let mut rows = Vec::new();
    for seed in &config.seeds {
        for prompt_len in &config.prompt_lengths {
            for arm in &config.arms {
                let final_step = config.steps;
                let checkpoint = checkpoints
                    .iter()
                    .find(|c| {
                        c.seed == *seed
                            && c.prompt_len == *prompt_len
                            && c.arm == arm.name
                            && c.step == final_step
                    })
                    .ok_or("final checkpoint missing")?;
                let mut engine = EngineClient::start(root)?;
                let current = session(&mut engine, arm, *seed, *prompt_len, config.batch_size)?;
                engine.load_checkpoint(&CheckpointRequest {
                    engine_id: current.engine_id.clone(),
                    path: output.join(&checkpoint.path).to_string_lossy().into_owned(),
                })?;
                let (score, metric_kind, tokens) =
                    evaluate_session(&mut engine, &current, &valid, &config, *prompt_len)?;
                if !score.is_finite() || tokens == 0 {
                    return Err(format!(
                        "invalid quality result for {} seed {seed} prompt {prompt_len}",
                        arm.name
                    ));
                }
                rows.push(json!({"seed":seed,"prompt_len":prompt_len,"arm":arm.name,"heldout_bpb":score,"bpb_kind":metric_kind,"continuation_bytes":tokens,"train_sha256":train.sha256,"validation_sha256":valid.sha256}));
            }
        }
    }
    rows_to_csv(&output.join("quality_rows.csv"), root, &rows)?;
    validation_io::write_json(
        output.join("quality_rows.json"),
        root,
        &json!({"schema":"ncm-rust-quality-v1","rows":rows,"row_count":rows.len()}),
    )
}

pub fn checkpoint_audit(root: &Path, sweep_dir: &Path) -> Result<(), String> {
    let output = resolve(root, sweep_dir);
    let (config, checkpoints) = load_manifest(root, sweep_dir)?;
    validate_output_location(root, &config, sweep_dir)?;
    let train = select_windows(
        &load_dataset(
            &resolve(root, &config.train_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.train_windows,
        "training",
    )?;
    let valid = select_windows(
        &load_dataset(
            &resolve(root, &config.validation_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.validation_windows,
        "validation",
    )?;
    validate_profile_windows(&config, &train, "training")?;
    validate_profile_windows(&config, &valid, "validation")?;
    let mut rows = Vec::new();
    for seed in &config.seeds {
        for prompt_len in &config.prompt_lengths {
            for arm in &config.arms {
                let arm_checkpoints = checkpoints
                    .iter()
                    .filter(|c| c.seed == *seed && c.prompt_len == *prompt_len && c.arm == arm.name)
                    .collect::<Vec<_>>();
                if arm_checkpoints.len() != config.checkpoint_steps.len() {
                    return Err(format!(
                        "checkpoint count mismatch for {} seed {seed} prompt {prompt_len}",
                        arm.name
                    ));
                }
                let mut engine = EngineClient::start(root)?;
                let current = session(&mut engine, arm, *seed, *prompt_len, config.batch_size)?;
                for checkpoint in arm_checkpoints {
                    let started = Instant::now();
                    let path = output.join(&checkpoint.path);
                    let handle = engine.load_checkpoint(&CheckpointRequest {
                        engine_id: current.engine_id.clone(),
                        path: path.to_string_lossy().into_owned(),
                    })?;
                    let (score, metric_kind, tokens) =
                        evaluate_session(&mut engine, &current, &valid, &config, *prompt_len)?;
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    if !score.is_finite() || !elapsed_ms.is_finite() || tokens == 0 {
                        return Err(format!(
                            "invalid checkpoint observation for {} seed {seed} prompt {prompt_len} step {}",
                            arm.name, checkpoint.step
                        ));
                    }
                    let bytes = fs::metadata(&path).map_err(|e| e.to_string())?.len();
                    let checkpoint_sha256 =
                        corpus::sha256(&fs::read(&path).map_err(|e| e.to_string())?);
                    rows.push(json!({"seed":seed,"prompt_len":prompt_len,"arm":arm.name,"checkpoint":checkpoint.step,"engine_step":handle.step,"heldout_bpb":score,"bpb_kind":metric_kind,"continuation_bytes":tokens,"checkpoint_bytes":bytes,"checkpoint_sha256":checkpoint_sha256,"train_sha256":train.sha256,"validation_sha256":valid.sha256,"audit_time_ms":elapsed_ms,"checkpoint_path":checkpoint.path}));
                }
            }
        }
    }
    rows_to_csv(&output.join("checkpoint_audit.csv"), root, &rows)?;
    validation_io::write_json(
        output.join("checkpoint_audit.json"),
        root,
        &json!({"schema":"ncm-rust-checkpoint-audit-v1","rows":rows,"row_count":rows.len()}),
    )
}

pub fn validate_suite(
    root: &Path,
    output_dir: &Path,
    timing_rows: usize,
    quality_rows_expected: usize,
    checkpoint_rows_expected: usize,
) -> Result<(), String> {
    let output = resolve(root, output_dir);
    let manifest_path = output.join("sweep.json");
    let manifest: Value =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid sweep manifest: {e}"))?;
    let config: SweepConfig = serde_json::from_value(
        manifest
            .get("config")
            .cloned()
            .ok_or("sweep manifest has no config")?,
    )
    .map_err(|e| format!("invalid sweep config in manifest: {e}"))?;
    validate_config(&config)?;
    validate_output_location(root, &config, output_dir)?;
    let timing = read_csv_table(&output.join("drafter_comparison.csv"))?;
    let quality: Value = serde_json::from_slice(
        &fs::read(output.join("quality_rows.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let checkpoints: Value = serde_json::from_slice(
        &fs::read(output.join("checkpoint_audit.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let quality_csv = read_csv_table(&output.join("quality_rows.csv"))?;
    let checkpoint_csv = read_csv_table(&output.join("checkpoint_audit.csv"))?;
    let quality_rows = quality
        .get("rows")
        .and_then(Value::as_array)
        .ok_or("quality rows missing")?;
    let checkpoint_rows = checkpoints
        .get("rows")
        .and_then(Value::as_array)
        .ok_or("checkpoint rows missing")?;
    let benchmark = config
        .benchmark
        .as_ref()
        .ok_or("sweep config has no drafter benchmark protocol")?;
    let raw_train = load_dataset(
        &resolve(root, &config.train_file),
        config.sequence_length + config.max_prompt_len,
    )?;
    let raw_valid = load_dataset(
        &resolve(root, &config.validation_file),
        config.sequence_length + config.max_prompt_len,
    )?;
    if manifest["train_sha256"].as_str() != Some(raw_train.sha256.as_str())
        || manifest["validation_sha256"].as_str() != Some(raw_valid.sha256.as_str())
    {
        return Err("sweep manifest corpus hashes do not match current inputs".into());
    }
    let train = select_windows(&raw_train, &config.train_windows, "training")?;
    let valid = select_windows(&raw_valid, &config.validation_windows, "validation")?;
    validate_profile_windows(&config, &train, "training")?;
    validate_profile_windows(&config, &valid, "validation")?;
    let windows = benchmark_windows(
        &valid,
        &config,
        config.profile.as_deref() == Some("real128"),
    )?;
    let continuation_hash = benchmark_rows_hash(
        &valid,
        &windows,
        config.max_prompt_len,
        config.sequence_length,
    );

    let mut expected_timing = BTreeSet::new();
    let mut prompt_hashes = BTreeMap::new();
    for seed in &config.seeds {
        for prompt in &config.prompt_lengths {
            let start = config.max_prompt_len - *prompt;
            prompt_hashes.insert(
                (*seed, *prompt),
                benchmark_rows_hash(&valid, &windows, start, *prompt),
            );
            for batch in &benchmark.batch_sizes {
                expected_timing.insert((
                    *seed,
                    "cached_teacher_reference".to_owned(),
                    *prompt,
                    *batch,
                    0,
                ));
                for drafter in &benchmark.drafters {
                    for draft in &benchmark.draft_lengths {
                        expected_timing.insert((
                            *seed,
                            drafter.label.clone(),
                            *prompt,
                            *batch,
                            *draft,
                        ));
                    }
                }
            }
        }
    }
    let strict = config.profile.as_deref() == Some("real128");
    if strict
        && (timing.rows.len() != 351
            || quality_rows.len() != 45
            || checkpoint_rows.len() != 135
            || timing_rows != 351
            || quality_rows_expected != 45
            || checkpoint_rows_expected != 135)
    {
        return Err(
            "real128 suite must contain 351 timing, 45 quality, and 135 checkpoint rows".into(),
        );
    }
    if timing.rows.len() != timing_rows
        || quality_rows.len() != quality_rows_expected
        || checkpoint_rows.len() != checkpoint_rows_expected
    {
        return Err("suite row counts do not match requested validation counts".into());
    }
    let timing_required = [
        "seed",
        "drafter",
        "prompt_len",
        "batch",
        "draft_len",
        "median_ms",
        "tokens_per_second",
        "continuation_sha256",
        "prompt_bucket_sha256",
    ];
    require_csv_columns(&timing, &timing_required, "timing")?;
    let mut actual_timing = BTreeSet::new();
    for row in &timing.rows {
        let key = (
            csv_u32(&timing, row, "seed")?,
            csv_field(&timing, row, "drafter")?.to_owned(),
            csv_usize(&timing, row, "prompt_len")?,
            csv_usize(&timing, row, "batch")?,
            csv_usize(&timing, row, "draft_len")?,
        );
        if !actual_timing.insert(key.clone()) {
            return Err(format!("duplicate timing cell {key:?}"));
        }
        let median = csv_f64(&timing, row, "median_ms")?;
        let throughput = csv_f64(&timing, row, "tokens_per_second")?;
        if median <= 0.0 || throughput <= 0.0 {
            return Err("timing values must be positive".into());
        }
        let continuation = csv_field(&timing, row, "continuation_sha256")?;
        let prompt_hash = csv_field(&timing, row, "prompt_bucket_sha256")?;
        if continuation != continuation_hash
            || !is_sha256(continuation)
            || !is_sha256(prompt_hash)
            || prompt_hashes.get(&(key.0, key.2)).map(String::as_str) != Some(prompt_hash)
        {
            return Err(format!("timing corpus signature mismatch for {key:?}"));
        }
    }
    if actual_timing != expected_timing {
        return Err(describe_set_mismatch(
            "timing",
            &expected_timing,
            &actual_timing,
        ));
    }

    let mut expected_quality = BTreeSet::new();
    for seed in &config.seeds {
        for prompt in &config.prompt_lengths {
            for arm in &config.arms {
                expected_quality.insert((*seed, *prompt, arm.name.clone()));
            }
        }
    }
    let actual_quality_json = validate_quality_json(quality_rows, &config, &train, &valid)?;
    let actual_quality_csv = validate_quality_csv(&quality_csv, &config, &train, &valid)?;
    if actual_quality_json != expected_quality
        || actual_quality_csv != expected_quality
        || actual_quality_json != actual_quality_csv
    {
        return Err(describe_set_mismatch(
            "quality",
            &expected_quality,
            &actual_quality_json,
        ));
    }

    let mut expected_checkpoints = BTreeSet::new();
    for seed in &config.seeds {
        for prompt in &config.prompt_lengths {
            for arm in &config.arms {
                for step in &config.checkpoint_steps {
                    expected_checkpoints.insert((*seed, *prompt, arm.name.clone(), *step));
                }
            }
        }
    }
    let actual_checkpoint_json =
        validate_checkpoint_json(checkpoint_rows, &output, &config, &train, &valid)?;
    let actual_checkpoint_csv = validate_checkpoint_csv(&checkpoint_csv)?;
    let checkpoint_json_keys = actual_checkpoint_json
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let checkpoint_csv_keys = actual_checkpoint_csv
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    if checkpoint_json_keys != expected_checkpoints
        || checkpoint_csv_keys != expected_checkpoints
        || actual_checkpoint_json != actual_checkpoint_csv
    {
        return Err(describe_set_mismatch(
            "checkpoint",
            &expected_checkpoints,
            &checkpoint_json_keys,
        ));
    }
    let seeds = &config.seeds;
    let qn = quality_rows.len();
    let cn = checkpoint_rows.len();
    validation_io::write_json(
        output.join("suite_validation.json"),
        root,
        &json!({"status":"complete","profile":config.profile,"timing_rows":timing.rows.len(),"quality_rows":qn,"checkpoint_rows":cn,"seeds":seeds,"prompt_lengths":config.prompt_lengths,"draft_lengths":benchmark.draft_lengths,"batch_sizes":benchmark.batch_sizes,"continuation_sha256":continuation_hash,"train_sha256":train.sha256,"validation_sha256":valid.sha256}),
    )
}

#[derive(Default)]
struct SpecObservation {
    elapsed_ms: f64,
    accepted: usize,
    proposed: usize,
    batch_rounds: usize,
    sequence_rounds: usize,
    prefix_sum: usize,
    accepted_by_position: Vec<usize>,
    proposed_by_position: Vec<usize>,
}

struct SpecRunConfig {
    prompt_len: usize,
    output_len: usize,
    draft_len: usize,
    temperature: f32,
    top_k: usize,
    seed: u32,
}

const DRAFTER_PROGRESS_SCHEMA: &str = "ncm-drafter-progress-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkProgress {
    schema: String,
    fingerprint: String,
    cells: BTreeMap<String, BenchmarkProgressCell>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkProgressCell {
    status: String,
    metrics: Map<String, Value>,
}

fn benchmark_cell_id(
    seed: u32,
    prompt_len: usize,
    drafter: &str,
    batch_size: usize,
    draft_len: usize,
) -> String {
    let quoted_drafter =
        serde_json::to_string(drafter).expect("string serialization is infallible");
    format!(
        "seed={seed};prompt={prompt_len};drafter={quoted_drafter};batch={batch_size};draft={draft_len}"
    )
}

fn benchmark_cell_id_from_metrics(metrics: &Map<String, Value>) -> Result<String, String> {
    let seed = u32::try_from(
        metrics
            .get("seed")
            .and_then(Value::as_u64)
            .ok_or("completed progress cell is missing an unsigned seed")?,
    )
    .map_err(|error| error.to_string())?;
    let prompt_len = metrics
        .get("prompt_len")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("completed progress cell is missing prompt_len")?;
    let drafter = metrics
        .get("drafter")
        .and_then(Value::as_str)
        .ok_or("completed progress cell is missing drafter")?;
    let batch_size = metrics
        .get("batch")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("completed progress cell is missing batch")?;
    let draft_len = metrics
        .get("draft_len")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("completed progress cell is missing draft_len")?;
    Ok(benchmark_cell_id(
        seed, prompt_len, drafter, batch_size, draft_len,
    ))
}

fn persist_benchmark_progress(path: &Path, progress: &BenchmarkProgress) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(progress).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("tmp");
    let mut file = fs::File::create(&temporary).map_err(|error| error.to_string())?;
    file.write_all(&bytes).map_err(|error| error.to_string())?;
    file.write_all(b"\n").map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    fs::rename(&temporary, path).map_err(|error| error.to_string())
}

fn load_benchmark_progress(path: &Path, fingerprint: String) -> Result<BenchmarkProgress, String> {
    if path.exists() {
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        let progress: BenchmarkProgress = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid drafter progress file: {error}"))?;
        if progress.schema != DRAFTER_PROGRESS_SCHEMA || progress.fingerprint != fingerprint {
            return Err("drafter progress file does not match this benchmark protocol or inputs; move it aside before restarting".into());
        }
        for (cell_id, cell) in &progress.cells {
            if cell.status != "completed" {
                return Err(
                    "drafter progress file contains a cell with an unsupported status".into(),
                );
            }
            if benchmark_cell_id_from_metrics(&cell.metrics)? != *cell_id {
                return Err(format!(
                    "drafter progress cell key does not match its metrics: {cell_id}"
                ));
            }
        }
        Ok(progress)
    } else {
        let progress = BenchmarkProgress {
            schema: DRAFTER_PROGRESS_SCHEMA.into(),
            fingerprint,
            cells: BTreeMap::new(),
        };
        persist_benchmark_progress(path, &progress)?;
        Ok(progress)
    }
}

fn complete_benchmark_cell(
    path: &Path,
    progress: &mut BenchmarkProgress,
    cell_id: String,
    metrics: Map<String, Value>,
) -> Result<(), String> {
    progress.cells.insert(
        cell_id,
        BenchmarkProgressCell {
            status: "completed".into(),
            metrics,
        },
    );
    persist_benchmark_progress(path, progress)
}

fn checkpoint_fingerprint(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| {
        format!("could not read checkpoint for drafter progress fingerprint: {error}")
    })?;
    Ok(corpus::sha256(&bytes))
}

struct BenchmarkProgressInputs<'a> {
    root: &'a Path,
    sweep_dir: &'a Path,
    config: &'a SweepConfig,
    checkpoints: &'a [CheckpointRow],
    target_arm: &'a str,
    drafters: &'a [(String, SweepArm)],
    options: &'a DrafterTimingOptions,
    validation_sha256: &'a str,
    strict_protocol: bool,
}

fn benchmark_progress_fingerprint(input: BenchmarkProgressInputs<'_>) -> Result<String, String> {
    let BenchmarkProgressInputs {
        root,
        sweep_dir,
        config,
        checkpoints,
        target_arm,
        drafters,
        options,
        validation_sha256,
        strict_protocol,
    } = input;
    let mut checkpoint_hashes = Vec::new();
    for seed in &config.seeds {
        if target_arm == "teacher" {
            let relative = teacher_checkpoint_path(root, sweep_dir, *seed)?;
            let absolute = resolve(root, sweep_dir).join(relative);
            checkpoint_hashes.push(json!({
                "seed": seed,
                "prompt_len": Value::Null,
                "arm": "teacher",
                "sha256": checkpoint_fingerprint(&absolute)?,
            }));
        } else {
            for prompt_len in &config.prompt_lengths {
                let checkpoint = checkpoints
                    .iter()
                    .find(|checkpoint| {
                        checkpoint.seed == *seed
                            && checkpoint.prompt_len == *prompt_len
                            && checkpoint.arm == target_arm
                            && checkpoint.step == config.steps
                    })
                    .ok_or_else(|| {
                        format!(
                            "target final checkpoint missing for seed {seed}, prompt {prompt_len}"
                        )
                    })?;
                checkpoint_hashes.push(json!({
                    "seed": seed,
                    "prompt_len": prompt_len,
                    "arm": target_arm,
                    "sha256": checkpoint_fingerprint(&resolve(root, sweep_dir).join(&checkpoint.path))?,
                }));
            }
        }
        for prompt_len in &config.prompt_lengths {
            for (_, spec) in drafters {
                let checkpoint = checkpoints
                    .iter()
                    .find(|checkpoint| {
                        checkpoint.seed == *seed
                            && checkpoint.prompt_len == *prompt_len
                            && checkpoint.arm == spec.name
                            && checkpoint.step == config.steps
                    })
                    .ok_or_else(|| format!("final checkpoint missing for {}", spec.name))?;
                checkpoint_hashes.push(json!({
                    "seed": seed,
                    "prompt_len": prompt_len,
                    "arm": spec.name,
                    "sha256": checkpoint_fingerprint(&resolve(root, sweep_dir).join(&checkpoint.path))?,
                }));
            }
        }
    }
    let payload = json!({
        "schema": DRAFTER_PROGRESS_SCHEMA,
        "implementation": validation_io::identity(root)?,
        "config": config,
        "target_arm": target_arm,
        "drafters": drafters,
        "options": options,
        "validation_sha256": validation_sha256,
        "strict_protocol": strict_protocol,
        "checkpoint_hashes": checkpoint_hashes,
    });
    let canonical = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    Ok(corpus::sha256(&canonical))
}

fn derive_seed(
    seed: u32,
    prompt_len: usize,
    batch: usize,
    draft: usize,
    repetition: usize,
    round: usize,
    stream: u32,
) -> u32 {
    let mut value = u64::from(seed)
        ^ (prompt_len as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (batch as u64).rotate_left(11)
        ^ (draft as u64).rotate_left(23)
        ^ (repetition as u64).wrapping_mul(0xd6e8feb86659fd93)
        ^ (round as u64).wrapping_mul(0xa0761d6478bd642f)
        ^ u64::from(stream);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    value as u32
}

fn run_speculative_observation(
    engine: &mut EngineClient,
    target: &Session,
    drafter: &Session,
    prompts: &[Vec<i32>],
    run: &SpecRunConfig,
) -> Result<SpecObservation, String> {
    let started = Instant::now();
    let result = engine.generate_and_verify_atomic(&GenerateAndVerifyAtomicRequest {
        target_engine_id: target.engine_id.clone(),
        drafter_engine_id: drafter.engine_id.clone(),
        prompts: prompts.to_vec(),
        prompt_len: run.prompt_len,
        output_len: run.output_len,
        draft_len: run.draft_len,
        temperature: run.temperature,
        top_k: run.top_k,
        seed: run.seed,
    })?;
    if result.tokens.len() != prompts.len()
        || result.tokens.iter().any(|row| row.len() != run.output_len)
        || result.accepted_by_position.len() != run.draft_len
        || result.proposed_by_position.len() != run.draft_len
    {
        return Err("atomic speculative worker returned invalid output shapes".into());
    }
    Ok(SpecObservation {
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        accepted: result.accepted_tokens,
        proposed: result.proposed_tokens,
        batch_rounds: result.batch_rounds,
        sequence_rounds: result.sequence_rounds,
        prefix_sum: result.prefix_sum,
        accepted_by_position: result.accepted_by_position,
        proposed_by_position: result.proposed_by_position,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn benchmark_rows_hash(data: &Dataset, rows: &[usize], start: usize, length: usize) -> String {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend(
            data.tokens[*row][start..start + length]
                .iter()
                .map(|x| *x as u8),
        );
        bytes.extend(
            data.masks[*row][start..start + length]
                .iter()
                .map(|x| u8::from(*x)),
        );
    }
    corpus::sha256(&bytes)
}

fn benchmark_windows(
    data: &Dataset,
    config: &SweepConfig,
    strict: bool,
) -> Result<Vec<usize>, String> {
    let eligible = data
        .masks
        .iter()
        .enumerate()
        .filter_map(|(i, mask)| {
            mask.get(..config.max_prompt_len + 1)
                .is_some_and(|prefix| prefix.iter().all(|valid| *valid))
                .then_some(i)
        })
        .collect::<Vec<_>>();
    if strict && eligible.len() < 32 {
        return Err(format!(
            "real128 drafter sweep requires 32 eligible validation windows, found {}",
            eligible.len()
        ));
    }
    if eligible.len() >= 32 {
        let mut selected = eligible[..31].to_vec();
        selected.push(
            *eligible
                .last()
                .ok_or("eligible validation split is empty")?,
        );
        Ok(selected)
    } else if !eligible.is_empty() {
        Ok(eligible)
    } else {
        Err("validation split has no window with the required prompt context".into())
    }
}

fn teacher_checkpoint_path(root: &Path, sweep_dir: &Path, seed: u32) -> Result<PathBuf, String> {
    let manifest: Value = serde_json::from_slice(
        &fs::read(resolve(root, sweep_dir).join("sweep.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    manifest
        .get("teacher_checkpoints")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row["seed"].as_u64() == Some(u64::from(seed)))
        })
        .and_then(|row| row["path"].as_str())
        .map(PathBuf::from)
        .ok_or_else(|| format!("teacher checkpoint missing for seed {seed}"))
}

/// Run the 3-seed × 3-prompt × reference/drafter/draft/batch timing grid.
/// The standard options and three drafter arms produce exactly 351 rows.
pub fn benchmark_drafters(
    root: &Path,
    sweep_dir: &Path,
    output_dir: &Path,
    target_arm: &str,
    drafters: &[(String, String)],
    options: &DrafterTimingOptions,
    strict_protocol: bool,
) -> Result<usize, String> {
    let (config, checkpoints) = load_manifest(root, sweep_dir)?;
    validate_output_location(root, &config, output_dir)?;
    if options.repetitions == 0
        || options.draft_lengths.is_empty()
        || options.batch_sizes.is_empty()
        || options
            .draft_lengths
            .iter()
            .any(|x| *x == 0 || *x > config.sequence_length)
        || options.batch_sizes.contains(&0)
        || !options.temperature.is_finite()
        || options.temperature < 0.0
        || options.top_k > 256
    {
        return Err("invalid drafter timing schedule or sampling controls".into());
    }
    if options
        .draft_lengths
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != options.draft_lengths.len()
        || options
            .batch_sizes
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != options.batch_sizes.len()
    {
        return Err("draft lengths and batch sizes must be unique".into());
    }
    if drafters.is_empty()
        || drafters
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != drafters.len()
    {
        return Err("drafter labels must be nonempty and unique".into());
    }
    if strict_protocol
        && (config.seeds != [200, 201, 202]
            || config.prompt_lengths != [0, 16, 64]
            || config.sequence_length != 128
            || config.max_prompt_len != 64
            || options.draft_lengths != [2, 4, 8, 16]
            || options.batch_sizes != [1, 16, 128]
            || drafters.len() != 3
            || options.warmups != 5
            || options.repetitions != 30)
    {
        return Err("benchmark-drafters strict protocol requires real128 seeds/buckets/sweeps, 5 warmups, 30 repetitions, and three drafter arms".into());
    }
    if strict_protocol && target_arm != "teacher" {
        return Err("strict protocol target arm must be named teacher".into());
    }
    if strict_protocol {
        let protocol = config
            .benchmark
            .as_ref()
            .ok_or("strict protocol config has no benchmark section")?;
        let configured_drafters = protocol
            .drafters
            .iter()
            .map(|item| (item.label.clone(), item.arm.clone()))
            .collect::<Vec<_>>();
        if target_arm != protocol.target_arm
            || drafters != configured_drafters
            || options.draft_lengths != protocol.draft_lengths
            || options.batch_sizes != protocol.batch_sizes
            || options.warmups != protocol.warmups
            || options.repetitions != protocol.repetitions
            || options.temperature != protocol.temperature
            || options.top_k != protocol.top_k
        {
            return Err(
                "strict timing invocation differs from the versioned real128 protocol".into(),
            );
        }
    }
    let target_spec = config.arms.iter().find(|a| a.name == target_arm);
    let target_config = if target_arm == "teacher" {
        config
            .teacher
            .as_ref()
            .ok_or("teacher config missing from sweep")?
    } else {
        match &target_spec.ok_or("target arm not found in sweep")?.model {
            SweepModel::CausalAr(ar) => ar,
            _ => return Err("target arm must use causal_ar".into()),
        }
    };
    let max_draft = options.draft_lengths.iter().copied().max().unwrap_or(0);
    if target_config.max_length < config.max_prompt_len + config.sequence_length + max_draft + 1 {
        return Err("target AR capacity must cover prompt, continuation, draft, and bonus".into());
    }
    let mut drafter_specs = Vec::new();
    for (label, arm_name) in drafters {
        let spec = config
            .arms
            .iter()
            .find(|a| a.name == *arm_name)
            .ok_or_else(|| format!("drafter arm {arm_name} not found"))?;
        if let SweepModel::CausalAr(ar) = &spec.model
            && ar.max_length < config.max_prompt_len + config.sequence_length + max_draft + 1
        {
            return Err(format!(
                "AR drafter {arm_name} capacity is too small for speculative continuation"
            ));
        }
        drafter_specs.push((label.clone(), spec.clone()));
    }
    let output = resolve(root, output_dir);
    if config.profile.as_deref() == Some("real128") && resolve(root, sweep_dir) != output {
        return Err(
            "real128 timing, sweep, quality, and audit artifacts must share the isolated directory"
                .into(),
        );
    }
    if !output.is_dir() {
        return Err(format!(
            "output/sweep directory must exist: {}",
            output.display()
        ));
    }
    if [
        "drafter_comparison.csv",
        "drafter_comparison.json",
        "drafter_timing_run.json",
    ]
    .iter()
    .any(|name| output.join(name).exists())
    {
        return Err(
            "drafter timing artifacts already exist; use a clean Rust-native output directory"
                .into(),
        );
    }
    let valid = select_windows(
        &load_dataset(
            &resolve(root, &config.validation_file),
            config.sequence_length + config.max_prompt_len,
        )?,
        &config.validation_windows,
        "validation",
    )?;
    validate_profile_windows(&config, &valid, "validation")?;
    let selected = benchmark_windows(&valid, &config, strict_protocol)?;
    let continuation_hash = benchmark_rows_hash(
        &valid,
        &selected,
        config.max_prompt_len,
        config.sequence_length,
    );
    let fingerprint = benchmark_progress_fingerprint(BenchmarkProgressInputs {
        root,
        sweep_dir,
        config: &config,
        checkpoints: &checkpoints,
        target_arm,
        drafters: &drafter_specs,
        options,
        validation_sha256: &valid.sha256,
        strict_protocol,
    })?;
    let progress_path = output.join("sweep_progress.json");
    let mut progress = load_benchmark_progress(&progress_path, fingerprint)?;
    let mut expected_cells = BTreeSet::new();
    for seed in &config.seeds {
        for prompt_len in &config.prompt_lengths {
            for batch_size in &options.batch_sizes {
                expected_cells.insert(benchmark_cell_id(
                    *seed,
                    *prompt_len,
                    "cached_teacher_reference",
                    *batch_size,
                    0,
                ));
                for (label, _) in &drafter_specs {
                    for draft_len in &options.draft_lengths {
                        expected_cells.insert(benchmark_cell_id(
                            *seed,
                            *prompt_len,
                            label,
                            *batch_size,
                            *draft_len,
                        ));
                    }
                }
            }
        }
    }
    if progress
        .cells
        .keys()
        .any(|cell_id| !expected_cells.contains(cell_id))
    {
        return Err(
            "drafter progress file contains cells outside the configured benchmark grid".into(),
        );
    }
    let progress_bar = ProgressBar::new(expected_cells.len() as u64);
    progress_bar.set_style(
        ProgressStyle::with_template("{wide_bar} {pos}/{len} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    progress_bar.set_message("benchmark-drafters");
    let mut rows = Vec::<Map<String, Value>>::new();
    let mut contexts_total = 0usize;
    for seed in &config.seeds {
        for prompt_len in &config.prompt_lengths {
            let mut context_cell_ids = Vec::new();
            for batch_size in &options.batch_sizes {
                context_cell_ids.push(benchmark_cell_id(
                    *seed,
                    *prompt_len,
                    "cached_teacher_reference",
                    *batch_size,
                    0,
                ));
            }
            for (label, _) in &drafter_specs {
                for draft_len in &options.draft_lengths {
                    for batch_size in &options.batch_sizes {
                        context_cell_ids.push(benchmark_cell_id(
                            *seed,
                            *prompt_len,
                            label,
                            *batch_size,
                            *draft_len,
                        ));
                    }
                }
            }
            if context_cell_ids
                .iter()
                .all(|cell_id| progress.cells.contains_key(cell_id))
            {
                for cell_id in context_cell_ids {
                    let cell = progress
                        .cells
                        .get(&cell_id)
                        .ok_or("completed benchmark cell disappeared")?;
                    rows.push(cell.metrics.clone());
                    progress_bar.inc(1);
                    contexts_total += 1;
                }
                continue;
            }
            let mut engine = EngineClient::start(root)?;
            let target = if target_arm == "teacher" {
                let teacher_config = config.teacher.as_ref().ok_or("teacher config missing")?;
                let handle = engine.causal_ar_init(&CausalArInitRequest {
                    config: teacher_config.clone(),
                    seed: *seed,
                    tokens: vec![vec![0]],
                    mask: None,
                })?;
                let path = teacher_checkpoint_path(root, sweep_dir, *seed)?;
                engine.load_checkpoint(&CheckpointRequest {
                    engine_id: handle.engine_id.clone(),
                    path: resolve(root, sweep_dir)
                        .join(path)
                        .to_string_lossy()
                        .into_owned(),
                })?;
                Session {
                    engine_id: handle.engine_id,
                    model: SweepModel::CausalAr(teacher_config.clone()),
                }
            } else {
                let spec = target_spec.ok_or("target arm not found")?;
                let target = session(&mut engine, spec, *seed, *prompt_len, config.batch_size)?;
                let checkpoint = checkpoints
                    .iter()
                    .find(|c| {
                        c.seed == *seed
                            && c.prompt_len == *prompt_len
                            && c.arm == target_arm
                            && c.step == config.steps
                    })
                    .ok_or("target final checkpoint missing")?;
                engine.load_checkpoint(&CheckpointRequest {
                    engine_id: target.engine_id.clone(),
                    path: resolve(root, sweep_dir)
                        .join(&checkpoint.path)
                        .to_string_lossy()
                        .into_owned(),
                })?;
                target
            };
            let mut loaded_drafters = Vec::new();
            for (label, spec) in &drafter_specs {
                let draft = session(&mut engine, spec, *seed, *prompt_len, config.batch_size)?;
                let checkpoint = checkpoints
                    .iter()
                    .find(|c| {
                        c.seed == *seed
                            && c.prompt_len == *prompt_len
                            && c.arm == spec.name
                            && c.step == config.steps
                    })
                    .ok_or_else(|| format!("final checkpoint missing for {}", spec.name))?;
                engine.load_checkpoint(&CheckpointRequest {
                    engine_id: draft.engine_id.clone(),
                    path: resolve(root, sweep_dir)
                        .join(&checkpoint.path)
                        .to_string_lossy()
                        .into_owned(),
                })?;
                loaded_drafters.push((label.clone(), draft));
            }
            let prompt_start = config.max_prompt_len - *prompt_len;
            let prompt_hash = benchmark_rows_hash(&valid, &selected, prompt_start, *prompt_len);
            let mut reference_by_batch = std::collections::BTreeMap::new();
            for batch_size in &options.batch_sizes {
                let cell_id = benchmark_cell_id(
                    *seed,
                    *prompt_len,
                    "cached_teacher_reference",
                    *batch_size,
                    0,
                );
                if let Some(cell) = progress.cells.get(&cell_id) {
                    let median_ms = cell.metrics["median_ms"]
                        .as_f64()
                        .ok_or("completed reference cell has no numeric median_ms")?;
                    reference_by_batch.insert(*batch_size, median_ms);
                    rows.push(cell.metrics.clone());
                    progress_bar.inc(1);
                    contexts_total += 1;
                    continue;
                }
                let row_ids = (0..*batch_size)
                    .map(|i| selected[i % selected.len()])
                    .collect::<Vec<_>>();
                let prompts = row_ids
                    .iter()
                    .map(|i| valid.tokens[*i][prompt_start..config.max_prompt_len].to_vec())
                    .collect::<Vec<_>>();
                let mut times = Vec::new();
                let mut last_tokens = Vec::new();
                for repetition in 0..(options.warmups + options.repetitions) {
                    let started = Instant::now();
                    let generated = engine.generate_tokens(&GenerateRequest {
                        engine_id: target.engine_id.clone(),
                        prompts: prompts.clone(),
                        max_new_tokens: config.sequence_length,
                        temperature: options.temperature,
                        top_k: options.top_k,
                        seed: derive_seed(*seed, *prompt_len, *batch_size, 0, repetition, 0, 3),
                    })?;
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    if repetition >= options.warmups {
                        times.push(elapsed);
                        last_tokens = generated.tokens;
                    }
                }
                let median_ms = median(&mut times);
                let seconds = median_ms / 1000.0;
                reference_by_batch.insert(*batch_size, median_ms);
                let mut row = Map::new();
                row.insert("seed".into(), json!(seed));
                row.insert("drafter".into(), json!("cached_teacher_reference"));
                row.insert(
                    "prompt_id".into(),
                    json!(format!("seed-{seed}-p{prompt_len}")),
                );
                row.insert("prompt_len".into(), json!(prompt_len));
                row.insert("length".into(), json!(config.sequence_length));
                row.insert("batch".into(), json!(batch_size));
                row.insert("draft_len".into(), json!(0));
                row.insert("acceptance".into(), json!(1.0));
                row.insert("acceptance_rate".into(), json!(1.0));
                row.insert("tokens_per_pass".into(), json!(1.0));
                row.insert("per_position_acceptance".into(), json!([]));
                row.insert("rounds".into(), json!(config.sequence_length));
                row.insert("measured_rounds".into(), json!(config.sequence_length));
                row.insert("median_ms".into(), json!(median_ms));
                row.insert("elapsed_seconds".into(), json!(seconds));
                row.insert(
                    "tokens_per_second".into(),
                    json!((*batch_size * config.sequence_length) as f64 / seconds.max(1e-12)),
                );
                row.insert("speedup_vs_cached_teacher".into(), json!(1.0));
                row.insert("timing_observations".into(), json!(options.repetitions));
                row.insert("bpb_kind".into(), json!("exact_teacher"));
                row.insert("continuation_sha256".into(), json!(continuation_hash));
                row.insert("prompt_bucket_sha256".into(), json!(prompt_hash));
                row.insert("tokens".into(), json!(last_tokens));
                complete_benchmark_cell(&progress_path, &mut progress, cell_id, row.clone())?;
                progress_bar.inc(1);
                rows.push(row);
                contexts_total += 1;
            }
            for (label, draft) in &loaded_drafters {
                for draft_len in &options.draft_lengths {
                    for batch_size in &options.batch_sizes {
                        let cell_id =
                            benchmark_cell_id(*seed, *prompt_len, label, *batch_size, *draft_len);
                        if let Some(cell) = progress.cells.get(&cell_id) {
                            rows.push(cell.metrics.clone());
                            progress_bar.inc(1);
                            contexts_total += 1;
                            continue;
                        }
                        let row_ids = (0..*batch_size)
                            .map(|i| selected[i % selected.len()])
                            .collect::<Vec<_>>();
                        let prompts = row_ids
                            .iter()
                            .map(|i| valid.tokens[*i][prompt_start..config.max_prompt_len].to_vec())
                            .collect::<Vec<_>>();
                        for repetition in 0..options.warmups {
                            let run = SpecRunConfig {
                                prompt_len: *prompt_len,
                                output_len: config.sequence_length,
                                draft_len: *draft_len,
                                temperature: options.temperature,
                                top_k: options.top_k,
                                seed: derive_seed(
                                    *seed,
                                    *prompt_len,
                                    *batch_size,
                                    *draft_len,
                                    repetition,
                                    0,
                                    5,
                                ),
                            };
                            let _ = run_speculative_observation(
                                &mut engine,
                                &target,
                                draft,
                                &prompts,
                                &run,
                            )?;
                        }
                        let mut times = Vec::new();
                        let mut accepted = 0usize;
                        let mut proposed = 0usize;
                        let mut batch_rounds = 0usize;
                        let mut sequence_rounds = 0usize;
                        let mut prefix_sum = 0usize;
                        let mut accepted_positions = vec![0usize; *draft_len];
                        let mut proposed_positions = vec![0usize; *draft_len];
                        for repetition in 0..options.repetitions {
                            let run = SpecRunConfig {
                                prompt_len: *prompt_len,
                                output_len: config.sequence_length,
                                draft_len: *draft_len,
                                temperature: options.temperature,
                                top_k: options.top_k,
                                seed: derive_seed(
                                    *seed,
                                    *prompt_len,
                                    *batch_size,
                                    *draft_len,
                                    repetition,
                                    0,
                                    6,
                                ),
                            };
                            let observation = run_speculative_observation(
                                &mut engine,
                                &target,
                                draft,
                                &prompts,
                                &run,
                            )?;
                            times.push(observation.elapsed_ms);
                            accepted += observation.accepted;
                            proposed += observation.proposed;
                            batch_rounds += observation.batch_rounds;
                            sequence_rounds += observation.sequence_rounds;
                            prefix_sum += observation.prefix_sum;
                            for pos in 0..*draft_len {
                                accepted_positions[pos] += observation.accepted_by_position[pos];
                                proposed_positions[pos] += observation.proposed_by_position[pos];
                            }
                        }
                        let median_ms = median(&mut times);
                        let seconds = median_ms / 1000.0;
                        let acceptance = accepted as f64 / proposed.max(1) as f64;
                        let mean_rounds =
                            sequence_rounds as f64 / (*batch_size * options.repetitions) as f64;
                        let mean_batch_rounds = batch_rounds as f64 / options.repetitions as f64;
                        let reference_ms = *reference_by_batch
                            .get(batch_size)
                            .ok_or("reference timing missing for batch size")?;
                        let mut row = Map::new();
                        row.insert("seed".into(), json!(seed));
                        row.insert("drafter".into(), json!(label));
                        row.insert(
                            "prompt_id".into(),
                            json!(format!("seed-{seed}-p{prompt_len}")),
                        );
                        row.insert("prompt_len".into(), json!(prompt_len));
                        row.insert("length".into(), json!(config.sequence_length));
                        row.insert("batch".into(), json!(batch_size));
                        row.insert("draft_len".into(), json!(draft_len));
                        row.insert(
                            "accepted_tokens".into(),
                            json!(accepted as f64 / options.repetitions as f64),
                        );
                        row.insert(
                            "proposed_tokens".into(),
                            json!(proposed as f64 / options.repetitions as f64),
                        );
                        row.insert("acceptance".into(), json!(acceptance));
                        row.insert("acceptance_rate".into(), json!(acceptance));
                        row.insert(
                            "per_position_acceptance".into(),
                            json!(
                                accepted_positions
                                    .iter()
                                    .zip(&proposed_positions)
                                    .map(|(a, p)| *a as f64 / (*p).max(1) as f64)
                                    .collect::<Vec<_>>()
                            ),
                        );
                        row.insert(
                            "tokens_per_pass".into(),
                            json!(
                                (*batch_size * config.sequence_length) as f64
                                    / (mean_batch_rounds * *batch_size as f64).max(1.0)
                            ),
                        );
                        row.insert("rounds".into(), json!(mean_batch_rounds));
                        row.insert("measured_rounds".into(), json!(mean_rounds));
                        row.insert("mean_sequence_rounds".into(), json!(mean_rounds));
                        row.insert(
                            "mean_prefix_length".into(),
                            json!(prefix_sum as f64 / sequence_rounds.max(1) as f64),
                        );
                        row.insert("median_ms".into(), json!(median_ms));
                        row.insert("elapsed_seconds".into(), json!(seconds));
                        row.insert(
                            "tokens_per_second".into(),
                            json!(
                                (*batch_size * config.sequence_length) as f64 / seconds.max(1e-12)
                            ),
                        );
                        row.insert(
                            "speedup_vs_cached_teacher".into(),
                            json!(reference_ms / median_ms.max(1e-12)),
                        );
                        row.insert("timing_observations".into(), json!(options.repetitions));
                        row.insert("measured_rounds_kind".into(), json!("mean per-sequence speculative rounds across synchronized observations"));
                        row.insert(
                            "acceptance_kind".into(),
                            json!("accepted-prefix survival fraction per eligible draft position"),
                        );
                        row.insert("bpb_kind".into(), json!("exact_teacher"));
                        row.insert("continuation_sha256".into(), json!(continuation_hash));
                        row.insert("prompt_bucket_sha256".into(), json!(prompt_hash));
                        complete_benchmark_cell(
                            &progress_path,
                            &mut progress,
                            cell_id,
                            row.clone(),
                        )?;
                        progress_bar.inc(1);
                        rows.push(row);
                        contexts_total += 1;
                    }
                }
            }
        }
    }
    let expected = config.seeds.len()
        * config.prompt_lengths.len()
        * options.batch_sizes.len()
        * (1 + drafters.len() * options.draft_lengths.len());
    if rows.len() != expected {
        return Err(format!(
            "drafter timing row count mismatch: got {}, expected {expected}",
            rows.len()
        ));
    }
    if strict_protocol && rows.len() != 351 {
        return Err(format!(
            "strict real128 timing grid generated {} rows instead of 351",
            rows.len()
        ));
    }
    if progress.cells.len() != expected {
        return Err(format!(
            "drafter progress has {} completed cells, expected {expected}",
            progress.cells.len()
        ));
    }
    progress_bar.finish_with_message("benchmark-drafters complete");
    let rows_json = rows.iter().cloned().map(Value::Object).collect::<Vec<_>>();
    validation_io::write_json(
        output.join("drafter_comparison.json"),
        root,
        &json!({"schema":"ncm-rust-drafter-timing-v1","seeds":config.seeds,"prompt_lengths":config.prompt_lengths,"draft_lengths":options.draft_lengths,"batch_sizes":options.batch_sizes,"warmups":options.warmups,"repetitions":options.repetitions,"continuation_sha256":continuation_hash,"row_count":rows.len(),"rows":rows_json}),
    )?;
    write_rows(&output.join("drafter_comparison.csv"), root, &rows)?;
    validation_io::write_json(
        output.join("drafter_timing_run.json"),
        root,
        &json!({"status":"complete","rows":rows.len(),"target_arm":target_arm,"drafters":drafters.iter().map(|(label,arm)| json!({"label":label,"arm":arm})).collect::<Vec<_>>(),"strict_protocol":strict_protocol,"repetitions":options.repetitions,"warmups":options.warmups,"batch_contexts":contexts_total,"continuation_sha256":continuation_hash}),
    )?;
    Ok(rows.len())
}

fn parse_csv_record(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            _ => field.push(ch),
        }
    }
    fields.push(field);
    fields
}

struct CsvTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn read_csv_table(path: &Path) -> Result<CsvTable, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut lines = text.lines();
    let headers = parse_csv_record(
        lines
            .next()
            .ok_or_else(|| format!("CSV is empty: {}", path.display()))?,
    );
    let mut rows = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let row = parse_csv_record(line);
        if row.len() != headers.len() {
            return Err(format!(
                "{} row {} has {} columns; expected {}",
                path.display(),
                index + 2,
                row.len(),
                headers.len()
            ));
        }
        rows.push(row);
    }
    Ok(CsvTable { headers, rows })
}

fn require_csv_columns(table: &CsvTable, columns: &[&str], label: &str) -> Result<(), String> {
    for column in columns {
        if !table.headers.iter().any(|header| header == column) {
            return Err(format!("{label} CSV is missing required column {column}"));
        }
    }
    Ok(())
}

fn csv_field<'a>(table: &'a CsvTable, row: &'a [String], name: &str) -> Result<&'a str, String> {
    let index = table
        .headers
        .iter()
        .position(|field| field == name)
        .ok_or_else(|| format!("CSV has no {name} column"))?;
    row.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("CSV row has no {name} value"))
}

fn csv_u32(table: &CsvTable, row: &[String], name: &str) -> Result<u32, String> {
    csv_field(table, row, name)?
        .parse()
        .map_err(|_| format!("CSV {name} must be a u32"))
}
fn csv_usize(table: &CsvTable, row: &[String], name: &str) -> Result<usize, String> {
    csv_field(table, row, name)?
        .parse()
        .map_err(|_| format!("CSV {name} must be a usize"))
}
fn csv_f64(table: &CsvTable, row: &[String], name: &str) -> Result<f64, String> {
    let value: f64 = csv_field(table, row, name)?
        .parse()
        .map_err(|_| format!("CSV {name} must be numeric"))?;
    if !value.is_finite() {
        return Err(format!("CSV {name} must be finite"));
    }
    Ok(value)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn json_u32(row: &Value, name: &str) -> Result<u32, String> {
    row[name]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!("JSON {name} must be a u32"))
}
fn json_usize(row: &Value, name: &str) -> Result<usize, String> {
    row[name]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("JSON {name} must be a usize"))
}
fn json_string<'a>(row: &'a Value, name: &str) -> Result<&'a str, String> {
    row[name]
        .as_str()
        .ok_or_else(|| format!("JSON {name} must be a string"))
}

fn validate_quality_json(
    rows: &[Value],
    config: &SweepConfig,
    train: &Dataset,
    valid: &Dataset,
) -> Result<BTreeSet<(u32, usize, String)>, String> {
    let mut keys = BTreeSet::new();
    for row in rows {
        let key = (
            json_u32(row, "seed")?,
            json_usize(row, "prompt_len")?,
            json_string(row, "arm")?.to_owned(),
        );
        if !keys.insert(key.clone()) {
            return Err(format!("duplicate quality cell {key:?}"));
        }
        let score = row["heldout_bpb"]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| format!("quality score is invalid for {key:?}"))?;
        let _ = score;
        if json_string(row, "train_sha256")? != train.sha256
            || json_string(row, "validation_sha256")? != valid.sha256
        {
            return Err(format!("quality corpus signature mismatch for {key:?}"));
        }
        if !config.seeds.contains(&key.0)
            || !config.prompt_lengths.contains(&key.1)
            || !config.arms.iter().any(|arm| arm.name == key.2)
        {
            return Err(format!("unexpected quality cell {key:?}"));
        }
    }
    Ok(keys)
}

fn validate_quality_csv(
    table: &CsvTable,
    config: &SweepConfig,
    train: &Dataset,
    valid: &Dataset,
) -> Result<BTreeSet<(u32, usize, String)>, String> {
    require_csv_columns(
        table,
        &[
            "seed",
            "prompt_len",
            "arm",
            "heldout_bpb",
            "train_sha256",
            "validation_sha256",
        ],
        "quality",
    )?;
    let mut keys = BTreeSet::new();
    for row in &table.rows {
        let key = (
            csv_u32(table, row, "seed")?,
            csv_usize(table, row, "prompt_len")?,
            csv_field(table, row, "arm")?.to_owned(),
        );
        if !keys.insert(key.clone()) {
            return Err(format!("duplicate quality CSV cell {key:?}"));
        }
        if csv_f64(table, row, "heldout_bpb")? <= 0.0
            || csv_field(table, row, "train_sha256")? != train.sha256
            || csv_field(table, row, "validation_sha256")? != valid.sha256
        {
            return Err(format!(
                "quality CSV signature or score mismatch for {key:?}"
            ));
        }
        if !config.seeds.contains(&key.0)
            || !config.prompt_lengths.contains(&key.1)
            || !config.arms.iter().any(|arm| arm.name == key.2)
        {
            return Err(format!("unexpected quality CSV cell {key:?}"));
        }
    }
    Ok(keys)
}

type CheckpointKey = (u32, usize, String, usize);

fn validate_checkpoint_json(
    rows: &[Value],
    output: &Path,
    config: &SweepConfig,
    train: &Dataset,
    valid: &Dataset,
) -> Result<BTreeMap<CheckpointKey, String>, String> {
    let mut signatures = BTreeMap::new();
    for row in rows {
        let key = (
            json_u32(row, "seed")?,
            json_usize(row, "prompt_len")?,
            json_string(row, "arm")?.to_owned(),
            json_usize(row, "checkpoint")?,
        );
        let hash = json_string(row, "checkpoint_sha256")?.to_owned();
        if !is_sha256(&hash) || signatures.insert(key.clone(), hash.clone()).is_some() {
            return Err(format!(
                "invalid or duplicate checkpoint signature for {key:?}"
            ));
        }
        if !config.seeds.contains(&key.0)
            || !config.prompt_lengths.contains(&key.1)
            || !config.arms.iter().any(|arm| arm.name == key.2)
            || !config.checkpoint_steps.contains(&key.3)
            || json_string(row, "train_sha256")? != train.sha256
            || json_string(row, "validation_sha256")? != valid.sha256
        {
            return Err(format!("unexpected checkpoint observation {key:?}"));
        }
        let relative = json_string(row, "checkpoint_path")?;
        let path = Path::new(relative);
        if path.is_absolute()
            || path
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err(format!("unsafe checkpoint path in observation {key:?}"));
        }
        let expected_path = format!(
            "seed-{}/prompt-{}/{}/step-{}.msgpack",
            key.0, key.1, key.2, key.3
        );
        if relative != expected_path {
            return Err(format!("checkpoint path does not match grid cell {key:?}"));
        }
        let bytes = fs::read(output.join(path))
            .map_err(|e| format!("checkpoint file missing for {key:?}: {e}"))?;
        if corpus::sha256(&bytes) != hash
            || row["checkpoint_bytes"].as_u64() != Some(bytes.len() as u64)
            || json_usize(row, "engine_step")? != key.3
            || row["heldout_bpb"]
                .as_f64()
                .is_none_or(|score| !score.is_finite() || score <= 0.0)
        {
            return Err(format!(
                "checkpoint contents or telemetry mismatch for {key:?}"
            ));
        }
    }
    Ok(signatures)
}

fn validate_checkpoint_csv(table: &CsvTable) -> Result<BTreeMap<CheckpointKey, String>, String> {
    require_csv_columns(
        table,
        &[
            "seed",
            "prompt_len",
            "arm",
            "checkpoint",
            "checkpoint_sha256",
        ],
        "checkpoint",
    )?;
    let mut signatures = BTreeMap::new();
    for row in &table.rows {
        let key = (
            csv_u32(table, row, "seed")?,
            csv_usize(table, row, "prompt_len")?,
            csv_field(table, row, "arm")?.to_owned(),
            csv_usize(table, row, "checkpoint")?,
        );
        let hash = csv_field(table, row, "checkpoint_sha256")?.to_owned();
        if !is_sha256(&hash) || signatures.insert(key.clone(), hash).is_some() {
            return Err(format!(
                "invalid or duplicate checkpoint CSV signature for {key:?}"
            ));
        }
    }
    Ok(signatures)
}

fn describe_set_mismatch<T: Ord + std::fmt::Debug>(
    label: &str,
    expected: &BTreeSet<T>,
    actual: &BTreeSet<T>,
) -> String {
    let missing = expected.difference(actual).take(5).collect::<Vec<_>>();
    let extra = actual.difference(expected).take(5).collect::<Vec<_>>();
    format!("{label} grid mismatch; missing {missing:?}, unexpected {extra:?}")
}

/// Evaluate the final trained codec checkpoints on document-separated validation windows.
pub fn run_reconstruction_pilot(
    output_dir: &Path,
    selected_length: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = if output_dir.is_absolute() {
        output_dir.to_owned()
    } else {
        root.join(output_dir)
    };
    fs::create_dir_all(&output)?;
    let validation = root.join("runs/real-text-pilot/validation.jsonl");
    let mut engine = EngineClient::start(&root).map_err(std::io::Error::other)?;
    let mut rows = Vec::new();
    let lengths: Vec<usize> = selected_length.map_or_else(|| vec![32, 128, 512], |x| vec![x]);
    if lengths.iter().any(|x| ![32, 128, 512].contains(x)) {
        return Err("length must be 32, 128, or 512".into());
    }
    for length in lengths {
        let config_path = root.join(format!(
            "runs/real-text-pilot/length-{length}-seed-0/config.json"
        ));
        let raw: Value = serde_json::from_slice(&fs::read(&config_path)?)?;
        let codebook = raw["codebook_size"].as_u64().unwrap_or(16) as usize;
        let dim = raw["dim"].as_u64().unwrap_or(16) as usize;
        let engine_config: EngineConfig = serde_json::from_value(json!({
            "rvq":{"lengths":raw["lengths"],"codebook_size":codebook,"dim":dim},
            "transformer":{"width":raw["width"],"heads":raw["heads"],"layers":raw["layers"]},
            "losses":{"beta":0.25,"entropy":0.0,"concept":0.0,"prior":0.0},
            "learning_rate":raw["learning_rate"],"init_batch_size":1,"token_vocab_size":256
        }))?;
        let handle = engine
            .init_model(&InitModelRequest {
                config: engine_config,
                seed: 0,
            })
            .map_err(std::io::Error::other)?;
        let checkpoint = root.join(format!(
            "runs/real-text-pilot/length-{length}-seed-0/checkpoint.msgpack"
        ));
        engine
            .load_codec(&CheckpointRequest {
                engine_id: handle.engine_id.clone(),
                path: checkpoint.to_string_lossy().into_owned(),
            })
            .map_err(std::io::Error::other)?;
        let windows = corpus::document_byte_windows(&validation, length)?;
        let (mut ber, mut em, mut n, mut valid_bytes) = (0.0, 0.0, 0usize, 0u64);
        let mut codebook_counts = Vec::<Vec<u64>>::new();
        for chunk in windows.rows.chunks(32) {
            let req = ForwardRequest {
                engine_id: handle.engine_id.clone(),
                batch: TensorBatch {
                    tokens: chunk
                        .iter()
                        .map(|r| r.iter().map(|b| i32::from(*b)).collect())
                        .collect(),
                    mask: None,
                    fallback_bytes: None,
                    fallback_mask: None,
                    prompt_ids: None,
                    prompt_mask: None,
                },
            };
            let m = engine
                .evaluate_reconstruction(&req)
                .map_err(std::io::Error::other)?;
            let count = chunk.len();
            ber += m.ber * m.valid_bytes as f64;
            valid_bytes += m.valid_bytes;
            em += m.exact_match * count as f64;
            n += count;
            if codebook_counts.is_empty() {
                codebook_counts = vec![
                    vec![0; m.codebook_counts.first().map_or(0, Vec::len)];
                    m.codebook_counts.len()
                ];
            }
            for (sum, batch) in codebook_counts.iter_mut().zip(m.codebook_counts) {
                for (total, value) in sum.iter_mut().zip(batch) {
                    *total += value;
                }
            }
        }
        let perplexity = codebook_counts
            .iter()
            .map(|counts| {
                let total: u64 = counts.iter().sum();
                if total == 0 {
                    return 0.0;
                }
                let entropy: f64 = counts
                    .iter()
                    .filter(|&&c| c > 0)
                    .map(|&c| {
                        let p = c as f64 / total as f64;
                        -p * p.ln()
                    })
                    .sum();
                entropy.exp()
            })
            .collect::<Vec<_>>();
        rows.push(json!({"length_bytes":length,"windows":n,"valid_bytes":valid_bytes,"document_count":windows.document_ids.iter().collect::<BTreeSet<_>>().len(),"corpus_sha256":windows.sha256,"byte_error_rate":ber/valid_bytes as f64,"exact_match_rate":em/n as f64,"codebook_assignment_counts":codebook_counts,"codebook_perplexity":perplexity }));
    }
    fs::write(
        output.join("reconstruction_metrics.json"),
        serde_json::to_vec_pretty(
            &json!({"protocol":"document-separated validation windows; final seed-0 checkpoints","results":rows}),
        )?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_rejects_duplicate_seeds_and_incomplete_checkpoint_schedule() {
        let config: SweepConfig = serde_json::from_value(json!({
            "train_file":"train.jsonl","validation_file":"valid.jsonl","sequence_length":8,"max_prompt_len":2,
            "seeds":[200,200],"prompt_lengths":[0,2],"steps":2,"checkpoint_steps":[1],"batch_size":1,
            "arms":[{"name":"ncm","kind":"ncm","config":{"rvq":{"lengths":[2,8],"codebook_size":2,"dim":2}}}]
        })).unwrap();
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn code_position_count_ignores_invalid_sentinels() {
        assert_eq!(count_valid_codes(&json!([[0, -1, 2], [3, -1, -1]])), 3);
    }

    #[test]
    fn checked_in_real128_config_pins_the_historical_grid_and_windows() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config: SweepConfig =
            serde_json::from_slice(&fs::read(root.join("configs/real128-sweep.json")).unwrap())
                .unwrap();
        validate_config(&config).unwrap();
        let mut expected_train = (0..31).collect::<Vec<_>>();
        expected_train.push(786);
        let mut expected_valid = (0..31).collect::<Vec<_>>();
        expected_valid.push(899);
        assert_eq!(config.train_windows, expected_train);
        assert_eq!(config.validation_windows, expected_valid);

        // Exercise selection without the ignored, locally prepared real-text corpus.
        // Row identities make index or ordering errors observable in this fixture.
        let width = config.max_prompt_len + config.sequence_length;
        let fixture = Dataset {
            tokens: (0..900).map(|index| vec![index; width]).collect(),
            masks: vec![vec![true; width]; 900],
            sha256: "synthetic-window-selection-fixture".into(),
        };
        let train = select_windows(&fixture, &config.train_windows, "train").unwrap();
        let valid = select_windows(&fixture, &config.validation_windows, "validation").unwrap();
        for (selected, indices) in [(&train, &expected_train), (&valid, &expected_valid)] {
            for (row, index) in selected.tokens.iter().zip(indices) {
                assert_eq!(row, &vec![*index as i32; width]);
            }
        }
        validate_profile_windows(&config, &train, "training").unwrap();
        validate_profile_windows(&config, &valid, "validation").unwrap();
        assert_eq!(config.seeds, [200, 201, 202]);
        assert_eq!(train.tokens.len(), 32);
        assert_eq!(valid.tokens.len(), 32);
        assert_eq!(config.checkpoint_steps, [2, 4, 8]);
    }

    #[test]
    fn suite_validator_requires_every_unique_cell_and_valid_signatures() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dir = std::env::temp_dir().join(format!("ncm suite validator {}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("train.txt"), b"abcdef").unwrap();
        fs::write(dir.join("valid.txt"), b"ghijkl").unwrap();
        let config = SweepConfig {
            profile: None,
            train_file: dir.join("train.txt"),
            validation_file: dir.join("valid.txt"),
            sequence_length: 2,
            max_prompt_len: 1,
            seeds: vec![200, 201, 202],
            prompt_lengths: vec![0, 1],
            steps: 1,
            checkpoint_steps: vec![1],
            batch_size: 1,
            train_windows: vec![0, 1],
            validation_windows: vec![0, 1],
            codec_pretrain: None,
            teacher: None,
            benchmark: Some(DrafterBenchmarkConfig {
                target_arm: "ar".into(),
                drafters: vec![DrafterArm {
                    label: "draft".into(),
                    arm: "ar".into(),
                }],
                draft_lengths: vec![1],
                batch_sizes: vec![1],
                warmups: 0,
                repetitions: 1,
                temperature: 1.,
                top_k: 0,
            }),
            arms: vec![SweepArm {
                name: "ar".into(),
                distillation_weight: 0.,
                training_mode: TrainingMode::Joint,
                model: SweepModel::CausalAr(CausalArConfig {
                    transformer: crate::rpc::TransformerConfig {
                        width: 4,
                        heads: 2,
                        layers: 1,
                        mlp_ratio: 2,
                        dropout: 0.,
                    },
                    max_length: 5,
                    learning_rate: 0.001,
                }),
            }],
        };
        validate_config(&config).unwrap();
        let raw_train = load_dataset(&config.train_file, 3).unwrap();
        let raw_valid = load_dataset(&config.validation_file, 3).unwrap();
        let train = select_windows(&raw_train, &config.train_windows, "train").unwrap();
        let valid = select_windows(&raw_valid, &config.validation_windows, "valid").unwrap();
        let mut timing_rows = Vec::new();
        let mut quality_rows = Vec::new();
        let mut checkpoint_rows = Vec::new();
        let timing_windows = benchmark_windows(&valid, &config, false).unwrap();
        let continuation_hash = benchmark_rows_hash(&valid, &timing_windows, 1, 2);
        for seed in &config.seeds {
            for prompt in &config.prompt_lengths {
                let prompt_hash = benchmark_rows_hash(&valid, &timing_windows, 1 - prompt, *prompt);
                for drafter in ["cached_teacher_reference", "draft"] {
                    let draft_len = if drafter == "cached_teacher_reference" {
                        0
                    } else {
                        1
                    };
                    timing_rows.push(json!({"seed":seed,"drafter":drafter,"prompt_len":prompt,"batch":1,"draft_len":draft_len,"median_ms":1.0,"tokens_per_second":2.0,"continuation_sha256":continuation_hash,"prompt_bucket_sha256":prompt_hash}));
                }
                quality_rows.push(json!({"seed":seed,"prompt_len":prompt,"arm":"ar","heldout_bpb":1.25,"train_sha256":train.sha256,"validation_sha256":valid.sha256}));
                let relative = format!("seed-{seed}/prompt-{prompt}/ar/step-1.msgpack");
                let path = dir.join(&relative);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                let contents = format!("checkpoint-{seed}-{prompt}").into_bytes();
                fs::write(&path, &contents).unwrap();
                let hash = corpus::sha256(&contents);
                checkpoint_rows.push(json!({"seed":seed,"prompt_len":prompt,"arm":"ar","checkpoint":1,"engine_step":1,"heldout_bpb":1.25,"checkpoint_bytes":contents.len(),"checkpoint_sha256":hash,"train_sha256":train.sha256,"validation_sha256":valid.sha256,"checkpoint_path":relative}));
            }
        }
        let checkpoint_index = checkpoint_rows.iter().map(|row| json!({"seed":row["seed"],"prompt_len":row["prompt_len"],"arm":row["arm"],"kind":"causal_ar","step":row["checkpoint"],"path":row["checkpoint_path"]})).collect::<Vec<_>>();
        validation_io::write_json(dir.join("sweep.json"), &root, &json!({"schema":"ncm-rust-sweep-v1","config":config,"train_sha256":raw_train.sha256,"validation_sha256":raw_valid.sha256,"checkpoints":checkpoint_index})).unwrap();
        let mut timing_maps = timing_rows
            .iter()
            .map(|row| row.as_object().unwrap().clone())
            .collect::<Vec<_>>();
        write_rows(&dir.join("drafter_comparison.csv"), &root, &timing_maps).unwrap();
        rows_to_csv(&dir.join("quality_rows.csv"), &root, &quality_rows).unwrap();
        rows_to_csv(&dir.join("checkpoint_audit.csv"), &root, &checkpoint_rows).unwrap();
        validation_io::write_json(
            dir.join("quality_rows.json"),
            &root,
            &json!({"rows":quality_rows}),
        )
        .unwrap();
        validation_io::write_json(
            dir.join("checkpoint_audit.json"),
            &root,
            &json!({"rows":checkpoint_rows}),
        )
        .unwrap();
        validate_suite(&root, &dir, 12, 6, 6).unwrap();
        let report: Value =
            serde_json::from_slice(&fs::read(dir.join("suite_validation.json")).unwrap()).unwrap();
        assert_eq!(report["status"], "complete");
        assert_eq!(report["timing_rows"], 12);
        assert_eq!(report["quality_rows"], 6);
        assert_eq!(report["checkpoint_rows"], 6);
        timing_maps[0] = timing_maps[1].clone();
        write_rows(&dir.join("drafter_comparison.csv"), &root, &timing_maps).unwrap();
        assert!(
            validate_suite(&root, &dir, 12, 6, 6)
                .unwrap_err()
                .contains("duplicate timing cell")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn drafter_progress_persists_metrics_and_rejects_stale_protocols() {
        let dir = std::env::temp_dir().join(format!(
            "ncm drafter progress {} {}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sweep_progress.json");
        let fingerprint = "protocol-hash-a".to_owned();
        let mut progress = load_benchmark_progress(&path, fingerprint.clone()).unwrap();
        let cell_id = benchmark_cell_id(200, 16, "ncm", 128, 8);
        let metrics = json!({"seed":200,"prompt_len":16,"drafter":"ncm","batch":128,"draft_len":8,"median_ms":12.5})
            .as_object()
            .unwrap()
            .clone();
        complete_benchmark_cell(&path, &mut progress, cell_id.clone(), metrics).unwrap();

        let resumed = load_benchmark_progress(&path, fingerprint).unwrap();
        assert_eq!(resumed.cells[&cell_id].status, "completed");
        assert_eq!(resumed.cells[&cell_id].metrics["median_ms"], 12.5);
        assert!(load_benchmark_progress(&path, "different-protocol".into()).is_err());
        assert_ne!(
            benchmark_cell_id(200, 16, "ncm", 128, 8),
            benchmark_cell_id(200, 16, "ncm", 16, 8)
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
