//! Rust-owned evaluation and speculative-drafter orchestration over JAX RPC.
use crate::{
    corpus,
    rpc::{
        CausalArConfig, CausalArEvaluateRequest, CausalArInitRequest, CausalArTrainStepRequest,
        EngineClient, GenerateRequest, SpeculativeVerifyRequest,
    },
    validation_io,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug)]
struct Rows {
    tokens: Vec<Vec<i32>>,
    mask: Vec<Vec<bool>>,
}

fn read_rows(path: &Path, length: usize) -> Result<Rows, String> {
    if length == 0 {
        return Err("sequence length must be positive".into());
    }
    if path.extension().is_some_and(|x| x == "jsonl") {
        let windows =
            corpus::document_masked_byte_windows(path, length).map_err(|e| e.to_string())?;
        return Ok(Rows {
            tokens: windows
                .rows
                .into_iter()
                .map(|r| r.into_iter().map(i32::from).collect())
                .collect(),
            mask: windows.masks,
        });
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    if bytes.is_empty() {
        return Err(format!("empty input file: {}", path.display()));
    }
    let mut tokens = Vec::new();
    let mut mask = Vec::new();
    for chunk in bytes.chunks(length) {
        let mut row = vec![0; length];
        let mut valid = vec![false; length];
        for (i, b) in chunk.iter().enumerate() {
            row[i] = i32::from(*b);
            valid[i] = true;
        }
        tokens.push(row);
        mask.push(valid);
    }
    Ok(Rows { tokens, mask })
}
fn resolve(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}
fn initial_request(config: CausalArConfig, seed: u32) -> CausalArInitRequest {
    CausalArInitRequest {
        tokens: vec![vec![0]],
        mask: None,
        config,
        seed,
    }
}
fn train(
    engine: &mut EngineClient,
    id: &str,
    config: &CausalArConfig,
    data: &Rows,
    steps: usize,
    batch_size: usize,
    seed: u64,
) -> Result<(), String> {
    if data.tokens.is_empty() {
        return Err("training corpus has no rows".into());
    }
    for step in 0..steps {
        let mut state = seed ^ 0xa0761d6478bd642f ^ (step as u64).wrapping_mul(0xe7037ed1a0b428db);
        let indices = (0..batch_size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as usize) % data.tokens.len()
            })
            .collect::<Vec<_>>();
        engine.causal_ar_train_step(&CausalArTrainStepRequest {
            engine_id: id.into(),
            tokens: indices.iter().map(|i| data.tokens[*i].clone()).collect(),
            mask: Some(indices.iter().map(|i| data.mask[*i].clone()).collect()),
            continuation_start: 0,
            teacher_engine_id: None,
            distillation_weight: 0.,
        })?;
    }
    let _ = config;
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct LossSummary {
    pub tokens: usize,
    pub sequences: usize,
    pub mean_nll: f64,
    pub variance: f64,
    pub perplexity: f64,
    pub sequence_nll: Vec<f64>,
}
fn loss_summary(
    rows: &Rows,
    log_likelihoods: &[Vec<f64>],
    prompt_len: usize,
) -> Result<LossSummary, String> {
    if rows.tokens.len() != log_likelihoods.len() {
        return Err("evaluation response batch count mismatch".into());
    }
    let mut sum = 0.;
    let mut squares = 0.;
    let mut n = 0;
    let mut sequence_nll = Vec::new();
    for (mask, ll) in rows.mask.iter().zip(log_likelihoods) {
        if mask.len() != ll.len() {
            return Err("token likelihood width mismatch".into());
        }
        let mut seq = 0.;
        for (i, (valid, logp)) in mask.iter().zip(ll).enumerate() {
            if *valid && i >= prompt_len {
                let loss = -logp;
                if !loss.is_finite() {
                    return Err("non-finite token loss".into());
                }
                sum += loss;
                squares += loss * loss;
                seq += loss;
                n += 1;
            }
        }
        if seq > 0. {
            sequence_nll.push(seq);
        }
    }
    if n == 0 {
        return Err("evaluation split has no scored tokens".into());
    }
    let mean = sum / n as f64;
    Ok(LossSummary {
        tokens: n,
        sequences: sequence_nll.len(),
        mean_nll: mean,
        variance: (squares / n as f64 - mean * mean).max(0.),
        perplexity: mean.exp(),
        sequence_nll,
    })
}
fn eval_split(
    engine: &mut EngineClient,
    id: &str,
    data: &Rows,
    batch_size: usize,
    prompt_len: usize,
) -> Result<LossSummary, String> {
    if batch_size == 0 {
        return Err("batch_size must be positive".into());
    }
    let mut all = Vec::new();
    for start in (0..data.tokens.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.tokens.len());
        let batch = Rows {
            tokens: data.tokens[start..end].to_vec(),
            mask: data.mask[start..end].to_vec(),
        };
        let response = engine.causal_ar_evaluate(&CausalArEvaluateRequest {
            engine_id: id.into(),
            tokens: batch.tokens.clone(),
            mask: Some(batch.mask.clone()),
            prompt_len,
        })?;
        if response.log_likelihoods.len() != batch.tokens.len() {
            return Err("worker likelihood response row count mismatch".into());
        }
        all.extend(response.log_likelihoods);
    }
    loss_summary(data, &all, prompt_len)
}

pub struct DecisiveBenchmarkRunner;
impl DecisiveBenchmarkRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        root: &Path,
        config_path: &Path,
        heldout_path: &Path,
        train_path: Option<&Path>,
        output: &Path,
        steps: usize,
        batch_size: usize,
        prompt_len: usize,
        seed: u32,
    ) -> Result<(), String> {
        let config: CausalArConfig = serde_json::from_slice(
            &fs::read(resolve(root, config_path)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("invalid AR config: {e}"))?;
        if prompt_len >= config.max_length || batch_size == 0 {
            return Err("prompt_len must be below max_length and batch_size positive".into());
        }
        let held = read_rows(&resolve(root, heldout_path), config.max_length)?;
        let train_data = train_path
            .map(|p| read_rows(&resolve(root, p), config.max_length))
            .transpose()?;
        if steps > 0 && train_data.is_none() {
            return Err("--train is required when --steps is positive".into());
        }
        let mut engine = EngineClient::start(root)?;
        let handle = engine.causal_ar_init(&initial_request(config.clone(), seed))?;
        if let Some(data) = train_data.as_ref() {
            train(
                &mut engine,
                &handle.engine_id,
                &config,
                data,
                steps,
                batch_size,
                u64::from(seed),
            )?;
        }
        let summary = eval_split(
            &mut engine,
            &handle.engine_id,
            &held,
            batch_size,
            prompt_len,
        )?;
        let report = json!({"protocol":"rust-causal-ar-evaluation-v1","model_config":config,"seed":seed,"training_steps":steps,"batch_size":batch_size,"prompt_len":prompt_len,"heldout":{"path":heldout_path,"metrics":{"tokens":summary.tokens,"sequences":summary.sequences,"mean_nll_nats":summary.mean_nll,"variance_nll_nats2":summary.variance,"perplexity":summary.perplexity,"sequence_nll_nats":summary.sequence_nll}}});
        let output = resolve(root, output);
        fs::create_dir_all(&output).map_err(|e| e.to_string())?;
        validation_io::write_json(output.join("results.json"), root, &report)?;
        let md = format!(
            "# Decisive benchmark\n\nModel: causal AR\n\nTraining steps: {steps}\n\nHeld-out tokens: {}\n\nMean NLL: {:.8} nats/token\n\nNLL variance: {:.8} nats²/token\n\nPerplexity: {:.8}\n",
            summary.tokens, summary.mean_nll, summary.variance, summary.perplexity
        );
        fs::write(output.join("summary.md"), md).map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[derive(Deserialize)]
struct PromptRecord {
    id: Option<String>,
    prompt: String,
}
fn prompt_suite(path: &Path) -> Result<Vec<(String, Vec<i32>)>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut prompts = Vec::new();
    if path.extension().is_some_and(|x| x == "jsonl") {
        for (i, line) in text
            .lines()
            .enumerate()
            .filter(|(_, x)| !x.trim().is_empty())
        {
            let row: PromptRecord = serde_json::from_str(line)
                .map_err(|e| format!("prompt JSONL line {}: {e}", i + 1))?;
            prompts.push((
                row.id.unwrap_or_else(|| format!("prompt-{}", i + 1)),
                row.prompt
                    .into_bytes()
                    .into_iter()
                    .map(i32::from)
                    .collect::<Vec<i32>>(),
            ));
        }
    } else {
        for (i, line) in text.lines().enumerate().filter(|(_, x)| !x.is_empty()) {
            prompts.push((
                format!("prompt-{}", i + 1),
                line.as_bytes()
                    .iter()
                    .copied()
                    .map(i32::from)
                    .collect::<Vec<i32>>(),
            ));
        }
    }
    if prompts.is_empty() || prompts.iter().any(|(_, p)| p.is_empty()) {
        return Err("prompt suite must contain nonempty prompts".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    if prompts.iter().any(|(id, _)| !ids.insert(id.clone())) {
        return Err("prompt suite contains duplicate prompt IDs".into());
    }
    Ok(prompts)
}

fn aggregate_drafter_rows(rows: &[Map<String, Value>]) -> Vec<Map<String, Value>> {
    let names = rows
        .iter()
        .filter_map(|r| r.get("drafter").and_then(Value::as_str))
        .filter(|name| *name != "cached_teacher_reference")
        .collect::<std::collections::BTreeSet<_>>();
    names
        .into_iter()
        .map(|name| {
            let selected = rows
                .iter()
                .filter(|r| r.get("drafter").and_then(Value::as_str) == Some(name))
                .collect::<Vec<_>>();
            let values = |key: &str| {
                selected
                    .iter()
                    .filter_map(|r| r.get(key).and_then(Value::as_f64))
                    .collect::<Vec<_>>()
            };
            let mean = |xs: &[f64]| {
                if xs.is_empty() {
                    0.
                } else {
                    xs.iter().sum::<f64>() / xs.len() as f64
                }
            };
            let acceptance = values("acceptance");
            let throughput = values("tokens_per_second");
            let latency = values("elapsed_seconds");
            let speedup = values("speedup_vs_cached_teacher");
            let mut times = latency.clone();
            times.sort_by(f64::total_cmp);
            let median = if times.is_empty() {
                0.
            } else if times.len() % 2 == 0 {
                (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.
            } else {
                times[times.len() / 2]
            };
            let mu = mean(&acceptance);
            let variance = if acceptance.is_empty() {
                0.
            } else {
                acceptance.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / acceptance.len() as f64
            };
            let mut row = Map::new();
            row.insert("drafter".into(), json!(name));
            row.insert("prompt_count".into(), json!(selected.len()));
            row.insert("mean_acceptance".into(), json!(mu));
            row.insert("acceptance_variance".into(), json!(variance));
            row.insert("mean_tokens_per_second".into(), json!(mean(&throughput)));
            row.insert("median_elapsed_seconds".into(), json!(median));
            row.insert(
                "mean_speedup_vs_cached_teacher".into(),
                json!(mean(&speedup)),
            );
            row
        })
        .collect()
}

fn paired_drafter_statistics(rows: &[Map<String, Value>], seed: u32) -> Result<Value, String> {
    let references = rows
        .iter()
        .filter(|row| {
            row.get("drafter").and_then(Value::as_str) == Some("cached_teacher_reference")
        })
        .filter_map(|row| {
            Some((
                row.get("prompt_id")?.as_str()?.to_owned(),
                row.get("tokens_per_second")?.as_f64()?,
            ))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let names = rows
        .iter()
        .filter_map(|row| row.get("drafter").and_then(Value::as_str))
        .filter(|name| *name != "cached_teacher_reference")
        .collect::<std::collections::BTreeSet<_>>();
    let mut statistics = Map::new();
    for name in names {
        let mut pairs = rows
            .iter()
            .filter(|row| row.get("drafter").and_then(Value::as_str) == Some(name))
            .filter_map(|row| {
                let id = row.get("prompt_id")?.as_str()?;
                Some((
                    id.to_owned(),
                    row.get("tokens_per_second")?.as_f64()?,
                    *references.get(id)?,
                ))
            })
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        let (draft, teacher): (Vec<_>, Vec<_>) = pairs
            .iter()
            .map(|(_, draft, teacher)| (*draft, *teacher))
            .unzip();
        let summary = validation_io::paired_bootstrap(&draft, &teacher, u64::from(seed), 2000)?;
        statistics.insert(name.to_owned(), validation_io::summary_json(&summary));
    }
    Ok(Value::Object(statistics))
}

pub struct DrafterComparisonRunner;
impl DrafterComparisonRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        root: &Path,
        target_config_path: &Path,
        drafter_configs: &[(String, PathBuf)],
        prompts_path: &Path,
        output: &Path,
        max_new_tokens: usize,
        draft_len: usize,
        temperature: f32,
        top_k: usize,
        seed: u32,
    ) -> Result<(), String> {
        if max_new_tokens == 0 || draft_len == 0 {
            return Err("max_new_tokens and draft_len must be positive".into());
        }
        let target_config: CausalArConfig = serde_json::from_slice(
            &fs::read(resolve(root, target_config_path)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let configs = drafter_configs
            .iter()
            .map(|(name, path)| {
                Ok((
                    name.clone(),
                    serde_json::from_slice::<CausalArConfig>(
                        &fs::read(resolve(root, path)).map_err(|e| e.to_string())?,
                    )
                    .map_err(|e| format!("invalid drafter config {name}: {e}"))?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if configs.is_empty() {
            return Err("at least one --drafter NAME=CONFIG is required".into());
        }
        let prompts = prompt_suite(&resolve(root, prompts_path))?;
        let mut engine = EngineClient::start(root)?;
        let target = engine.causal_ar_init(&initial_request(target_config.clone(), seed))?;
        let mut drafters = Vec::new();
        for (i, (name, config)) in configs.iter().enumerate() {
            let h = engine.causal_ar_init(&initial_request(
                config.clone(),
                seed.wrapping_add(i as u32 + 1),
            ))?;
            drafters.push((name.clone(), config.clone(), h.engine_id));
        }
        let mut output_rows = Vec::<Map<String, Value>>::new();
        for (pi, (prompt_id, prompt)) in prompts.iter().enumerate() {
            let reference_started = Instant::now();
            let reference = engine.generate(&GenerateRequest {
                engine_id: target.engine_id.clone(),
                prompts: vec![prompt.clone()],
                max_new_tokens,
                temperature,
                top_k,
                seed: seed.wrapping_add((pi * 131_071) as u32),
            })?;
            let reference_elapsed = reference_started.elapsed().as_secs_f64();
            let mut reference_row = Map::new();
            reference_row.insert("seed".into(), json!(seed));
            reference_row.insert("prompt_id".into(), json!(prompt_id));
            reference_row.insert("drafter".into(), json!("cached_teacher_reference"));
            reference_row.insert("draft_len".into(), json!(0));
            reference_row.insert("prompt_len".into(), json!(prompt.len()));
            reference_row.insert("length".into(), json!(max_new_tokens));
            reference_row.insert("batch".into(), json!(1));
            reference_row.insert("acceptance".into(), json!(1.));
            reference_row.insert("acceptance_rate".into(), json!(1.));
            reference_row.insert("tokens_per_pass".into(), json!(1.));
            reference_row.insert("median_ms".into(), json!(reference_elapsed * 1000.));
            reference_row.insert("elapsed_seconds".into(), json!(reference_elapsed));
            reference_row.insert(
                "tokens_per_second".into(),
                json!(max_new_tokens as f64 / reference_elapsed.max(1e-12)),
            );
            reference_row.insert("rounds".into(), json!(max_new_tokens));
            reference_row.insert("speedup_vs_cached_teacher".into(), json!(1.));
            reference_row.insert("measured_rounds".into(), json!(max_new_tokens));
            reference_row.insert("compiler_flops".into(), Value::Null);
            reference_row.insert("analytic_flops_per_round".into(), Value::Null);
            reference_row.insert("analytic_flops".into(), Value::Null);
            reference_row.insert("analytic_label".into(), json!("not reported"));
            reference_row.insert("compiler_label".into(), json!("not reported"));
            reference_row.insert("bpb_kind".into(), json!("exact_teacher"));
            reference_row.insert("teacher_bpb".into(), Value::Null);
            reference_row.insert("teacher_warning".into(), json!(false));
            reference_row.insert(
                "measured_rounds_kind".into(),
                json!("single direct autoregressive generation"),
            );
            reference_row.insert("per_position_acceptance".into(), json!([]));
            reference_row.insert(
                "acceptance_kind".into(),
                json!("accepted-prefix survival fraction per eligible draft position"),
            );
            reference_row.insert(
                "latency_note".into(),
                json!("single JSON-RPC generation; includes compilation on first shape"),
            );
            reference_row.insert("tokens".into(), json!(reference.tokens.first()));
            output_rows.push(reference_row);
            for (di, (name, config, draft_id)) in drafters.iter().enumerate() {
                if prompt.len() + max_new_tokens + 1 > target_config.max_length
                    || prompt.len() + max_new_tokens > config.max_length
                {
                    return Err(format!(
                        "prompt {prompt_id} exceeds configured model capacity"
                    ));
                }
                let mut context = prompt.clone();
                let mut accepted = 0usize;
                let mut proposed = 0usize;
                let mut rounds = 0usize;
                let mut accepted_by_position = vec![0usize; draft_len];
                let mut proposed_by_position = vec![0usize; draft_len];
                let mut prefix_sum = 0usize;
                let started = Instant::now();
                while context.len() - prompt.len() < max_new_tokens {
                    prefix_sum += context.len();
                    let remaining = max_new_tokens - (context.len() - prompt.len());
                    let d = remaining.min(draft_len);
                    let generated = engine.generate(&GenerateRequest {
                        engine_id: draft_id.clone(),
                        prompts: vec![context.clone()],
                        max_new_tokens: d,
                        temperature,
                        top_k,
                        seed: seed.wrapping_add((pi * 100_003 + di * 1_009 + rounds) as u32),
                    })?;
                    let drafts = generated
                        .tokens
                        .into_iter()
                        .next()
                        .ok_or("drafter returned no rows")?;
                    let probs = generated
                        .probabilities
                        .into_iter()
                        .next()
                        .ok_or("drafter returned no probability rows")?;
                    if drafts.len() != d || probs.len() != d {
                        return Err("drafter returned inconsistent proposal shape".into());
                    }
                    let verification = engine.speculative_verify(&SpeculativeVerifyRequest {
                        engine_id: target.engine_id.clone(),
                        prefixes: vec![context.clone()],
                        draft_tokens: vec![drafts.clone()],
                        proposal_probs: vec![probs],
                        temperature,
                        top_k,
                        seed: seed.wrapping_add((pi * 97_409 + di * 887 + rounds) as u32),
                    })?;
                    let n = verification
                        .accepted_lengths
                        .first()
                        .copied()
                        .ok_or("verifier returned no acceptance count")?;
                    if n > d {
                        return Err("verifier accepted more tokens than proposed".into());
                    }
                    let returned = verification
                        .tokens
                        .first()
                        .ok_or("verifier returned no token row")?;
                    context.extend_from_slice(&drafts[..n]);
                    accepted += n;
                    proposed += d;
                    for position in 0..d {
                        proposed_by_position[position] += 1;
                        if position < n {
                            accepted_by_position[position] += 1;
                        }
                    }
                    if context.len() - prompt.len() < max_new_tokens {
                        context.push(
                            *returned
                                .get(d)
                                .ok_or("verifier omitted correction/bonus token")?,
                        );
                    }
                    rounds += 1;
                }
                let elapsed = started.elapsed().as_secs_f64();
                let new_tokens = context.len() - prompt.len();
                let mut row = Map::new();
                row.insert("prompt_id".into(), json!(prompt_id));
                row.insert("seed".into(), json!(seed));
                row.insert("drafter".into(), json!(name));
                row.insert("prompt_len".into(), json!(prompt.len()));
                row.insert("length".into(), json!(max_new_tokens));
                row.insert("batch".into(), json!(1));
                row.insert("max_new_tokens".into(), json!(max_new_tokens));
                row.insert("draft_len".into(), json!(draft_len));
                row.insert("rounds".into(), json!(rounds));
                row.insert("proposed_tokens".into(), json!(proposed));
                row.insert("accepted_tokens".into(), json!(accepted));
                row.insert(
                    "acceptance_rate".into(),
                    json!(accepted as f64 / proposed.max(1) as f64),
                );
                row.insert(
                    "acceptance".into(),
                    json!(accepted as f64 / proposed.max(1) as f64),
                );
                row.insert(
                    "per_position_acceptance".into(),
                    json!(
                        accepted_by_position
                            .iter()
                            .zip(&proposed_by_position)
                            .map(|(a, p)| *a as f64 / (*p).max(1) as f64)
                            .collect::<Vec<_>>()
                    ),
                );
                row.insert(
                    "acceptance_kind".into(),
                    json!("accepted-prefix survival fraction per eligible draft position"),
                );
                row.insert(
                    "tokens_per_pass".into(),
                    json!(new_tokens as f64 / rounds.max(1) as f64),
                );
                row.insert("median_ms".into(), json!(elapsed * 1000.));
                row.insert(
                    "speedup_vs_cached_teacher".into(),
                    json!(reference_elapsed / elapsed.max(1e-12)),
                );
                row.insert("measured_rounds".into(), json!(rounds));
                row.insert("mean_sequence_rounds".into(), json!(rounds));
                row.insert("compiler_flops".into(), Value::Null);
                row.insert("analytic_flops_per_round".into(), Value::Null);
                row.insert("analytic_flops".into(), Value::Null);
                row.insert("analytic_label".into(), json!("not reported"));
                row.insert("compiler_label".into(), json!("not reported"));
                row.insert("bpb_kind".into(), json!("exact_teacher"));
                row.insert("teacher_bpb".into(), Value::Null);
                row.insert("teacher_warning".into(), json!(false));
                row.insert(
                    "measured_rounds_kind".into(),
                    json!("single sequence measured round count"),
                );
                row.insert(
                    "mean_prefix_length".into(),
                    json!(prefix_sum as f64 / rounds.max(1) as f64),
                );
                row.insert("timing_observations".into(), json!(1));
                row.insert(
                    "latency_note".into(),
                    json!("single JSON-RPC speculative run; includes compilation on first shape"),
                );
                row.insert("elapsed_seconds".into(), json!(elapsed));
                row.insert(
                    "tokens_per_second".into(),
                    json!(new_tokens as f64 / elapsed.max(1e-12)),
                );
                row.insert("tokens".into(), json!(context[prompt.len()..].to_vec()));
                output_rows.push(row);
            }
        }
        let output = resolve(root, output);
        fs::create_dir_all(&output).map_err(|e| e.to_string())?;
        let aggregates = aggregate_drafter_rows(&output_rows);
        let paired_statistics = paired_drafter_statistics(&output_rows, seed)?;
        let rows = output_rows
            .iter()
            .cloned()
            .map(Value::Object)
            .collect::<Vec<_>>();
        validation_io::write_json(
            output.join("results.json"),
            root,
            &json!({"protocol":"rust-speculative-drafter-comparison-v1","target_config":target_config,"max_new_tokens":max_new_tokens,"draft_len":draft_len,"rows":rows,"aggregates":aggregates,"paired_throughput_vs_teacher":paired_statistics}),
        )?;
        validation_io::write_csv(output.join("drafter_comparison.csv"), root, &output_rows)?;
        validation_io::write_csv(output.join("drafter_summary.csv"), root, &aggregates)?;
        validation_io::write_json(
            output.join("paired_statistics.json"),
            root,
            &paired_statistics,
        )?;
        let mut md = String::from(
            "# Drafter comparison\n\n| Prompt | Drafter | Acceptance | Tokens/s | Rounds |\n|---|---|---:|---:|---:|\n",
        );
        md.push_str("\n## Aggregate comparison\n\n| Drafter | Prompts | Mean acceptance | Acceptance variance | Mean tokens/s | Median elapsed (s) | Mean speedup |\n|---|---:|---:|---:|---:|---:|---:|\n");
        for row in &aggregates {
            md.push_str(&format!(
                "| {} | {} | {:.4} | {:.6} | {:.2} | {:.4} | {:.3} |\n",
                row["drafter"].as_str().unwrap_or(""),
                row["prompt_count"].as_u64().unwrap_or(0),
                row["mean_acceptance"].as_f64().unwrap_or(0.),
                row["acceptance_variance"].as_f64().unwrap_or(0.),
                row["mean_tokens_per_second"].as_f64().unwrap_or(0.),
                row["median_elapsed_seconds"].as_f64().unwrap_or(0.),
                row["mean_speedup_vs_cached_teacher"].as_f64().unwrap_or(0.)
            ));
        }
        md.push_str("\nPaired bootstrap mean throughput difference versus direct target generation (draft minus target, tokens/s):\n\n| Drafter | Mean difference | 95% CI | Prompts |\n|---|---:|---:|---:|\n");
        if let Some(stats) = paired_statistics.as_object() {
            for (name, value) in stats {
                md.push_str(&format!(
                    "| {name} | {:.2} | [{:.2}, {:.2}] | {} ({}) |\n",
                    value["mean_difference"].as_f64().unwrap_or(0.),
                    value["ci95"][0].as_f64().unwrap_or(0.),
                    value["ci95"][1].as_f64().unwrap_or(0.),
                    value["seeds"].as_u64().unwrap_or(0),
                    value["label"].as_str().unwrap_or("")
                ));
            }
        }
        for row in &output_rows {
            md.push_str(&format!(
                "| {} | {} | {:.4} | {:.2} | {} |\n",
                row["prompt_id"].as_str().unwrap_or(""),
                row["drafter"].as_str().unwrap_or(""),
                row["acceptance_rate"].as_f64().unwrap_or(0.),
                row["tokens_per_second"].as_f64().unwrap_or(0.),
                row["rounds"].as_u64().unwrap_or(0)
            ));
        }
        fs::write(output.join("summary.md"), md).map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_loss_aggregation_matches_hand_computed_baseline() {
        let rows = Rows {
            tokens: vec![vec![1, 2, 3], vec![4, 5, 6]],
            mask: vec![vec![true; 3], vec![true, true, false]],
        };
        let ll = vec![vec![-1., -2., -3.], vec![-4., -5., 0.]];
        let x = loss_summary(&rows, &ll, 1).unwrap();
        assert_eq!(x.tokens, 3);
        assert_eq!(x.sequences, 2);
        assert!((x.mean_nll - 10. / 3.).abs() < 1e-12);
        assert!((x.variance - 14. / 9.).abs() < 1e-12);
    }

    #[test]
    fn drafter_statistics_match_a_paired_hand_computed_baseline() {
        let reference = serde_json::from_value::<Map<String, Value>>(json!({
            "prompt_id":"p1", "drafter":"cached_teacher_reference", "tokens_per_second":10.0
        }))
        .unwrap();
        let drafter = serde_json::from_value::<Map<String, Value>>(json!({
            "prompt_id":"p1", "drafter":"draft", "tokens_per_second":13.0,
            "acceptance":0.5, "elapsed_seconds":2.0, "speedup_vs_cached_teacher":1.3
        }))
        .unwrap();
        let stats = paired_drafter_statistics(&[reference, drafter.clone()], 7).unwrap();
        assert_eq!(stats["draft"]["mean_difference"], 3.0);
        assert_eq!(stats["draft"]["ci95"], json!([3.0, 3.0]));
        let aggregate = aggregate_drafter_rows(&[drafter]);
        assert_eq!(aggregate.len(), 1);
        assert_eq!(aggregate[0]["mean_acceptance"], 0.5);
    }
}
