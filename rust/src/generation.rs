//! Isolated fixed-prompt generation pilot orchestration.
use crate::rpc::{
    CausalArConfig, CausalArInitRequest, CheckpointRequest, EngineClient, GenerateSequenceRequest,
    GenerationConfig, RvqPriorGenerationRequest,
};
use serde_json::json;
use std::{fs, path::Path};

pub fn run_generation_pilot(
    root: &Path,
    model_config_path: &Path,
    checkpoint_path: &Path,
    prompt: &str,
    config: GenerationConfig,
) -> Result<(), String> {
    config.validate().map_err(str::to_owned)?;
    if prompt.is_empty() {
        return Err("prompt must not be empty".into());
    }
    let model_config: CausalArConfig =
        serde_json::from_slice(&fs::read(model_config_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid causal AR config: {e}"))?;
    let prompt_bytes = prompt
        .as_bytes()
        .iter()
        .map(|&b| i32::from(b))
        .collect::<Vec<_>>();
    if prompt_bytes.len() + config.max_length > model_config.max_length {
        return Err("prompt plus generation length exceeds model max_length".into());
    }
    let mut engine = EngineClient::start(root)?;
    let initialized = engine.causal_ar_init(&CausalArInitRequest {
        config: model_config,
        seed: config.seed,
        tokens: vec![vec![0]],
        mask: None,
    })?;
    engine.load_checkpoint(&CheckpointRequest {
        engine_id: initialized.engine_id.clone(),
        path: checkpoint_path.to_string_lossy().into_owned(),
    })?;
    let response = engine.generate_sequence(&GenerateSequenceRequest {
        engine_id: initialized.engine_id,
        prompts: vec![prompt_bytes],
        config: config.clone(),
    })?;
    let output = root.join("reports/generation-pilot");
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    fs::write(
        output.join("generated_samples.json"),
        serde_json::to_vec_pretty(&json!({
            "seed": config.seed,
            "prompt": prompt,
            "samples": response.tokens,
            "log_probabilities": response.log_probabilities,
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        output.join("generation_metrics.json"),
        serde_json::to_vec_pretty(&json!({
            "config": config,
            "metrics": response.metrics,
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn run_rvq_prior_generation_pilot(
    root: &Path,
    model_config_path: &Path,
    codec_checkpoint_path: &Path,
    prior_checkpoint_path: &Path,
    prompt: &str,
    config: GenerationConfig,
    output_dir: &Path,
) -> Result<(), String> {
    config.validate().map_err(str::to_owned)?;
    if prompt.is_empty() {
        return Err("prompt must not be empty".into());
    }
    let model_config: serde_json::Value =
        serde_json::from_slice(&fs::read(model_config_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid RVQ model config: {e}"))?;
    let prompt_bytes = prompt
        .as_bytes()
        .iter()
        .map(|&byte| i32::from(byte))
        .collect();
    let mut engine = EngineClient::start(root)?;
    let response = engine.generate_rvq_prior(&RvqPriorGenerationRequest {
        model_config,
        codec_checkpoint: codec_checkpoint_path.to_string_lossy().into_owned(),
        prior_checkpoint: prior_checkpoint_path.to_string_lossy().into_owned(),
        prompt: prompt_bytes,
        config: config.clone(),
    })?;
    if response.tokens.is_empty()
        || response
            .tokens
            .iter()
            .flatten()
            .any(|&token| !(0..=255).contains(&token))
        || response
            .log_probabilities
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        || !response.metrics.mean_log_probability.is_finite()
        || !response.metrics.mean_entropy_nats.is_finite()
        || !response.metrics.generation_seconds.is_finite()
    {
        return Err("RVQ generation returned invalid bytes or non-finite metrics".into());
    }
    let decoded = response
        .tokens
        .iter()
        .map(|tokens| {
            String::from_utf8_lossy(&tokens.iter().map(|&b| b as u8).collect::<Vec<_>>())
                .to_string()
        })
        .collect::<Vec<_>>();
    let output = if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        root.join(output_dir)
    };
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    fs::write(
        output.join("generated_samples.json"),
        serde_json::to_vec_pretty(&json!({
            "mode": "rvq_categorical_prior",
            "seed": config.seed,
            "prompt": prompt,
            "samples": decoded,
            "sample_bytes": response.tokens,
            "code_log_probabilities": response.log_probabilities,
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        output.join("generation_metrics.json"),
        serde_json::to_vec_pretty(&json!({
            "mode": "rvq_categorical_prior",
            "config": config,
            "metrics": response.metrics,
        }))
        .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::rpc::GenerationConfig;
    #[test]
    fn generation_config_checks_sampling_domain() {
        let valid = GenerationConfig {
            max_length: 8,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.95,
            seed: 4,
        };
        assert!(valid.validate().is_ok());
        assert!(
            GenerationConfig {
                max_length: 0,
                ..valid.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            GenerationConfig {
                temperature: f32::NAN,
                ..valid.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            GenerationConfig {
                top_p: 0.0,
                ..valid.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            GenerationConfig {
                top_k: 257,
                ..valid
            }
            .validate()
            .is_err()
        );
    }
}
