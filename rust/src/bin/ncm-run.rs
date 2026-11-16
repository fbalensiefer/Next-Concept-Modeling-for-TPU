//! Command dispatch and paper-build orchestration. Model computation stays in JAX.
use ncm::rpc::{Request, Response};
use serde_json::{Value, json};
use std::{
    env,
    ffi::OsString,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
};
fn engine_call(root: &Path, method: &str) -> Result<Value, String> {
    let python = env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into());
    let mut paths = vec![root.join("src")];
    if let Some(old) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&old));
    }
    let mut child = Command::new(python)
        .args(["-u", "-m", "ncm.engine_rpc"])
        .current_dir(root)
        .env(
            "PYTHONPATH",
            env::join_paths(paths).map_err(|e| e.to_string())?,
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("could not start JAX engine: {e}"))?;
    let request = Request::new(json!(1), method, json!({}));
    request.validate().map_err(str::to_owned)?;
    let mut input = child.stdin.take().ok_or("JAX engine stdin unavailable")?;
    serde_json::to_writer(&mut input, &request).map_err(|e| e.to_string())?;
    input.write_all(b"\n").map_err(|e| e.to_string())?;
    drop(input);
    let mut output = BufReader::new(child.stdout.take().ok_or("JAX engine stdout unavailable")?);
    let mut line = String::new();
    output.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut extra = String::new();
    output.read_line(&mut extra).map_err(|e| e.to_string())?;
    if !extra.is_empty() {
        return Err("JAX engine emitted multiple responses for one request".into());
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("JAX engine exited with {status}"));
    }
    let response: Response =
        serde_json::from_str(&line).map_err(|e| format!("invalid JAX engine response: {e}"))?;
    if response.jsonrpc != "2.0" || response.id != json!(1) {
        return Err("JAX engine response version or ID mismatch".into());
    }
    if let Some(error) = response.error {
        return Err(format!("JAX engine RPC {}: {}", error.code, error.message));
    }
    response
        .result
        .ok_or_else(|| "JAX engine response has no result".into())
}
fn audit_papers(root: &Path, args: &[OsString]) -> Result<(), String> {
    let mut build_dir = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--build-dir") if index + 1 < args.len() => {
                build_dir = Some(PathBuf::from(&args[index + 1]));
                index += 1;
            }
            _ => return Err("audit-papers accepts --build-dir PATH".into()),
        }
        index += 1;
    }
    ncm::paper::generate_bibliographies(root, true)?;
    ncm::paper::audit_papers(root, build_dir.as_deref())
}
fn build_papers(root: &Path, args: &[OsString]) -> Result<(), String> {
    let mut folder = root.join("build/papers");
    let mut offline = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].to_str() {
            Some("--build-dir") if i+1<args.len() => { folder=PathBuf::from(&args[i+1]); i+=1; },
            Some("--offline") => offline=true,
            _ => return Err("build-papers accepts --build-dir PATH and --offline; set NCM_TECTONIC to choose Tectonic".into())
        }
        i += 1;
    }
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    let folder = folder.canonicalize().map_err(|e| e.to_string())?;
    ncm::paper::generate_results(root, false)?;
    ncm::paper::generate_recent_results(root, false)?;
    ncm::paper::generate_bibliographies(root, false)?;
    for stem in ["ncm_systems_paper", "SCALE_research_paper"] {
        let mut c = Command::new(env::var_os("NCM_TECTONIC").unwrap_or_else(|| "tectonic".into()));
        c.args(["--keep-logs", "--keep-intermediates", "--outdir"])
            .arg(&folder);
        if offline {
            c.arg("--only-cached");
        }
        let status = c
            .arg(format!("{stem}.tex"))
            .current_dir(root.join("paper"))
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("paper compilation failed: {status}"));
        }
    }
    audit_papers(
        root,
        &["--build-dir".into(), folder.clone().into_os_string()],
    )?;
    for stem in ["ncm_systems_paper", "SCALE_research_paper"] {
        std::fs::copy(
            folder.join(format!("{stem}.pdf")),
            root.join(format!("output/pdf/{stem}.pdf")),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn train(root: &Path, args: &[OsString]) -> Result<(), String> {
    let mut config = None;
    let mut output = None;
    let mut resume = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].to_str() {
            Some("--config") if i + 1 < args.len() => {
                config = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            Some("--output") if i + 1 < args.len() => {
                output = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            Some("--resume") => resume = true,
            _ => return Err("train accepts --config PATH --output DIR [--resume]".into()),
        }
        i += 1;
    }
    let config = config.ok_or("train requires --config PATH")?;
    let output = output.ok_or("train requires --output DIR")?;
    ncm::workloads::train_codec(root, &config, &output, resume)
}
fn generate_bibliographies(root: &Path, args: &[OsString]) -> Result<(), String> {
    let check = match args {
        [] => false,
        [flag] if flag == "--check" => true,
        _ => return Err("generate-bibliographies accepts only --check".into()),
    };
    ncm::paper::generate_bibliographies(root, check)
}
fn update_manifests(root: &Path, args: &[OsString]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("update-manifests takes no arguments".into());
    }
    ncm::paper::update_manifests(root)?;
    println!("Implementation and document manifests refreshed; verify with shasum -a 256 -c.");
    Ok(())
}
fn generate_pilot(root: &Path, args: &[OsString]) -> Result<(), String> {
    use ncm::rpc::GenerationConfig;
    let (mut model, mut checkpoint, mut codec_checkpoint, mut prior_checkpoint, mut prompt) =
        (None, None, None, None, None);
    let (mut mode, mut output) = (None, PathBuf::from("reports/generation-pilot"));
    let mut config = GenerationConfig {
        max_length: 32,
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        seed: 0,
    };
    let mut i = 0;
    while i < args.len() {
        if i + 1 >= args.len() {
            return Err(format!("missing value for {}", args[i].to_string_lossy()));
        }
        let key = args[i].to_str().unwrap_or("");
        let value = args[i + 1].to_str().ok_or("arguments must be UTF-8")?;
        match key {
            "--model-config" | "--config" => model = Some(PathBuf::from(value)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--codec-checkpoint" => codec_checkpoint = Some(PathBuf::from(value)),
            "--prior-checkpoint" => prior_checkpoint = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value.to_owned()),
            "--mode" => mode = Some(value.to_owned()),
            "--output" => output = PathBuf::from(value),
            "--max-length" => config.max_length = parse_usize_arg(&args[i + 1], key)?,
            "--top-k" => config.top_k = parse_usize_arg(&args[i + 1], key)?,
            "--temperature" => {
                config.temperature = value.parse().map_err(|_| "temperature must be numeric")?
            }
            "--top-p" => config.top_p = value.parse().map_err(|_| "top-p must be numeric")?,
            "--seed" => {
                config.seed = parse_usize_arg(&args[i + 1], key)?
                    .try_into()
                    .map_err(|_| "seed must fit uint32")?
            }
            _ => return Err(format!("unknown generate-pilot option: {key}")),
        }
        i += 2;
    }
    if mode.as_deref() == Some("rvq-prior")
        || codec_checkpoint.is_some()
        || prior_checkpoint.is_some()
    {
        if mode.as_deref().is_some_and(|value| value != "rvq-prior") {
            return Err("RVQ checkpoint arguments require --mode rvq-prior".into());
        }
        return ncm::generation::run_rvq_prior_generation_pilot(
            root,
            &model.ok_or("--config/--model-config is required")?,
            &codec_checkpoint.ok_or("--codec-checkpoint is required for RVQ prior mode")?,
            &prior_checkpoint.ok_or("--prior-checkpoint is required for RVQ prior mode")?,
            &prompt.ok_or("--prompt is required")?,
            config,
            &output,
        );
    }
    if mode.as_deref().is_some_and(|value| value != "byte-ar") {
        return Err("--mode must be byte-ar or rvq-prior".into());
    }
    ncm::generation::run_generation_pilot(
        root,
        &model.ok_or("--model-config is required")?,
        &checkpoint.ok_or("--checkpoint is required")?,
        &prompt.ok_or("--prompt is required")?,
        config,
    )
}
fn parse_usize_arg(value: &std::ffi::OsStr, name: &str) -> Result<usize, String> {
    value
        .to_str()
        .and_then(|x| x.parse::<usize>().ok())
        .ok_or_else(|| format!("{name} must be a nonnegative integer"))
}
fn decisive_benchmark(root: &Path, args: &[OsString]) -> Result<(), String> {
    let (mut config, mut heldout, mut training, mut output) = (None, None, None, None);
    let (mut steps, mut batch, mut prompt, mut seed) = (0usize, 16usize, 0usize, 0u32);
    let mut i = 0;
    while i < args.len() {
        let key = args[i].to_str().unwrap_or("");
        if i + 1 >= args.len() {
            return Err(format!("missing value for {key}"));
        }
        let value = &args[i + 1];
        match key {
            "--model-config" => config = Some(PathBuf::from(value)),
            "--heldout" => heldout = Some(PathBuf::from(value)),
            "--train" => training = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--steps" => steps = parse_usize_arg(value, key)?,
            "--batch-size" => batch = parse_usize_arg(value, key)?,
            "--prompt-len" => prompt = parse_usize_arg(value, key)?,
            "--seed" => {
                seed = parse_usize_arg(value, key)?
                    .try_into()
                    .map_err(|_| "--seed must fit uint32")?
            }
            _ => return Err(format!("unknown run-decisive-benchmark option: {key}")),
        }
        i += 2;
    }
    ncm::model_benchmarks::DecisiveBenchmarkRunner::run(
        root,
        &config.ok_or("--model-config is required")?,
        &heldout.ok_or("--heldout is required")?,
        training.as_deref(),
        &output.ok_or("--output is required")?,
        steps,
        batch,
        prompt,
        seed,
    )
}
fn drafter_comparison(root: &Path, args: &[OsString]) -> Result<(), String> {
    let (mut target, mut prompts, mut output) = (None, None, None);
    let (mut max_new, mut draft_len, mut top_k, mut seed) = (None, None, 0usize, 0u32);
    let mut temperature = 1f32;
    let mut drafters = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].to_str().unwrap_or("");
        if i + 1 >= args.len() {
            return Err(format!("missing value for {key}"));
        }
        let value = &args[i + 1];
        match key {
            "--target-config" => target = Some(PathBuf::from(value)),
            "--prompts" => prompts = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--drafter" => {
                let spec = value.to_str().ok_or("drafter spec must be UTF-8")?;
                let (name, path) = spec
                    .split_once('=')
                    .ok_or("--drafter expects NAME=CONFIG")?;
                if name.is_empty() || path.is_empty() {
                    return Err("--drafter expects NAME=CONFIG".into());
                }
                drafters.push((name.to_owned(), PathBuf::from(path)));
            }
            "--max-new-tokens" => max_new = Some(parse_usize_arg(value, key)?),
            "--draft-len" => draft_len = Some(parse_usize_arg(value, key)?),
            "--top-k" => top_k = parse_usize_arg(value, key)?,
            "--seed" => {
                seed = parse_usize_arg(value, key)?
                    .try_into()
                    .map_err(|_| "--seed must fit uint32")?
            }
            "--temperature" => {
                temperature = value
                    .to_str()
                    .and_then(|x| x.parse().ok())
                    .ok_or("--temperature must be numeric")?
            }
            _ => return Err(format!("unknown run-drafter-comparison option: {key}")),
        }
        i += 2;
    }
    ncm::model_benchmarks::DrafterComparisonRunner::run(
        root,
        &target.ok_or("--target-config is required")?,
        &drafters,
        &prompts.ok_or("--prompts is required")?,
        &output.ok_or("--output is required")?,
        max_new.ok_or("--max-new-tokens is required")?,
        draft_len.ok_or("--draft-len is required")?,
        temperature,
        top_k,
        seed,
    )
}
fn sweep_command(root: &Path, command: &str, args: &[OsString]) -> Result<(), String> {
    let mut config = None;
    let mut output = None;
    let mut i = 0;
    while i < args.len() {
        let key = args[i].to_str().unwrap_or("");
        if i + 1 >= args.len() {
            return Err(format!("missing value for {key}"));
        }
        let value = PathBuf::from(&args[i + 1]);
        match key {
            "--config" if command == "train-sweep" => config = Some(value),
            "--output" | "--output-dir" if command == "train-sweep" => output = Some(value),
            "--sweep" | "--dir" if command != "train-sweep" => output = Some(value),
            _ => return Err(format!("unknown {command} option: {key}")),
        }
        i += 2;
    }
    match command {
        "train-sweep" => ncm::sweep::train_sweep(
            root,
            &config.ok_or("--config is required")?,
            &output.ok_or("--output is required")?,
        ),
        "quality-eval" => ncm::sweep::quality_eval(root, &output.ok_or("--sweep is required")?),
        "checkpoint-audit" => {
            ncm::sweep::checkpoint_audit(root, &output.ok_or("--sweep is required")?)
        }
        "validate-benchmark-suite" => {
            ncm::sweep::validate_suite(root, &output.ok_or("--sweep is required")?, 351, 45, 135)
        }
        _ => Err(format!("unknown sweep command {command}")),
    }
}
fn benchmark_drafters(root: &Path, args: &[OsString]) -> Result<(), String> {
    let mut sweep = PathBuf::from("reports/validation-controls/real128-rust-native");
    let mut target_arm = None;
    let mut drafters = Vec::new();
    let mut warmups = None;
    let mut repetitions = None;
    let mut i = 0;
    while i < args.len() {
        let key = args[i].to_str().unwrap_or("");
        if i + 1 >= args.len() {
            return Err(format!("missing value for {key}"));
        }
        let value = args[i + 1]
            .to_str()
            .ok_or("benchmark arguments must be UTF-8")?;
        match key {
            "--sweep" => sweep = PathBuf::from(value),
            "--target-arm" => target_arm = Some(value.to_owned()),
            "--drafter" => {
                let (label, arm) = value.split_once('=').ok_or("--drafter expects LABEL=ARM")?;
                if label.is_empty() || arm.is_empty() {
                    return Err("--drafter expects LABEL=ARM".into());
                }
                drafters.push((label.to_owned(), arm.to_owned()));
            }
            "--warmups" => {
                warmups = Some(value.parse().map_err(|_| "--warmups must be an integer")?)
            }
            "--repetitions" => {
                repetitions = Some(
                    value
                        .parse()
                        .map_err(|_| "--repetitions must be an integer")?,
                )
            }
            _ => return Err(format!("unknown benchmark-drafters option: {key}")),
        }
        i += 2;
    }
    let manifest_path = if sweep.is_absolute() {
        sweep.join("sweep.json")
    } else {
        root.join(&sweep).join("sweep.json")
    };
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("invalid sweep manifest: {e}"))?;
    let config: ncm::sweep::SweepConfig = serde_json::from_value(
        manifest
            .get("config")
            .cloned()
            .ok_or("sweep manifest has no config")?,
    )
    .map_err(|e| e.to_string())?;
    let protocol = config
        .benchmark
        .ok_or("sweep config has no drafter benchmark protocol")?;
    let target_arm = target_arm.unwrap_or(protocol.target_arm);
    let drafters = if drafters.is_empty() {
        protocol
            .drafters
            .iter()
            .map(|d| (d.label.clone(), d.arm.clone()))
            .collect()
    } else {
        drafters
    };
    let options = ncm::sweep::DrafterTimingOptions {
        draft_lengths: protocol.draft_lengths,
        batch_sizes: protocol.batch_sizes,
        warmups: warmups.unwrap_or(protocol.warmups),
        repetitions: repetitions.unwrap_or(protocol.repetitions),
        temperature: protocol.temperature,
        top_k: protocol.top_k,
    };
    let output = PathBuf::from("reports/validation-controls/real128-rust-native");
    let rows = ncm::sweep::benchmark_drafters(
        root,
        &sweep,
        &output,
        &target_arm,
        &drafters,
        &options,
        true,
    )?;
    println!(
        "wrote {rows} Rust-native drafter/reference timing rows to {}",
        output.display()
    );
    Ok(())
}
fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "--list" {
        println!(
            "ncm-run <command> [arguments]\n\nCommands:\n  engine-info\n  build-papers\n  train\n  train-sweep --config PATH --output DIR\n  quality-eval --sweep DIR\n  checkpoint-audit --sweep DIR\n  benchmark-drafters --sweep DIR\n  validate-benchmark-suite --sweep DIR\n  reconstruct-pilot [--output DIR] [--length 32|128|512]\n  generate-pilot [--mode byte-ar|rvq-prior] [--model-config|--config PATH] [--checkpoint PATH | --codec-checkpoint PATH --prior-checkpoint PATH] --prompt TEXT [--max-length N] [--temperature F] [--top-k N] [--top-p F] [--seed N] [--output DIR]\n  real-text-reconstruction\n  run-tc-probe\n  run-decisive-benchmark\n  run-drafter-comparison\n  summarize-maintenance\n  summarize-domain-validation\n  summarize-dependency-results\n  domain-source-baselines\n  alignment-diagnosis\n  summarize-reconstruction-capacity\n  generate-results\n  generate-recent-results\n  generate-bibliographies\n  update-manifests\n  audit-papers"
        );
        return ExitCode::SUCCESS;
    }
    let root = env::current_dir().ok().and_then(|p| {
        p.ancestors()
            .find(|x| x.join("Cargo.toml").is_file() && x.join("src/ncm").is_dir())
            .map(Path::to_path_buf)
    });
    let Some(root) = root else {
        eprintln!("run ncm-run from the repository or a subdirectory");
        return ExitCode::from(2);
    };
    let name = args[0].to_string_lossy();
    let result = if name == "engine-info" {
        engine_call(&root, "engine.describe").map(|result| {
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        })
    } else if name == "build-papers" {
        build_papers(&root, &args[1..])
    } else if name == "train" {
        train(&root, &args[1..])
    } else if name == "run-decisive-benchmark" {
        decisive_benchmark(&root, &args[1..])
    } else if name == "run-drafter-comparison" {
        drafter_comparison(&root, &args[1..])
    } else if name == "train-sweep"
        || name == "quality-eval"
        || name == "checkpoint-audit"
        || name == "validate-benchmark-suite"
    {
        sweep_command(&root, &name, &args[1..])
    } else if name == "benchmark-drafters" {
        benchmark_drafters(&root, &args[1..])
    } else if name == "reconstruct-pilot" {
        let (mut output, mut length) = (PathBuf::from("reports/reconstruction-pilot"), None);
        let mut i = 1;
        while i < args.len() {
            match args[i].to_str() {
                Some("--output") if i + 1 < args.len() => {
                    output = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
                Some("--length") if i + 1 < args.len() => {
                    length = args[i + 1].to_str().and_then(|x| x.parse::<usize>().ok());
                    if length.is_none() {
                        eprintln!("--length must be 32, 128, or 512");
                        return ExitCode::from(2);
                    }
                    i += 1;
                }
                _ => {
                    eprintln!("reconstruct-pilot accepts --output DIR and --length 32|128|512");
                    return ExitCode::from(2);
                }
            }
            i += 1;
        }
        ncm::sweep::run_reconstruction_pilot(&output, length).map_err(|e| e.to_string())
    } else if name == "generate-pilot" {
        generate_pilot(&root, &args[1..])
    } else if name == "real-text-reconstruction" {
        let mut output = None;
        let mut steps = 100usize;
        let mut index = 1;
        while index < args.len() {
            match args[index].to_str() {
                Some("--output") if index + 1 < args.len() => {
                    output = Some(PathBuf::from(&args[index + 1]));
                    index += 1;
                }
                Some("--steps") if index + 1 < args.len() => {
                    steps = match args[index + 1].to_str().and_then(|x| x.parse().ok()) {
                        Some(x) => x,
                        None => {
                            return {
                                eprintln!("--steps must be a positive integer");
                                ExitCode::from(2)
                            };
                        }
                    };
                    index += 1;
                }
                _ => {
                    return {
                        eprintln!("real-text-reconstruction requires --output DIR [--steps N]");
                        ExitCode::from(2)
                    };
                }
            }
            index += 1;
        }
        match output {
            Some(path) => ncm::workloads::run_real_text_reconstruction(&root, &path, steps),
            None => Err("--output is required".into()),
        }
    } else if name == "run-tc-probe" {
        let mut output = PathBuf::from("reports/validation-controls/tc-probe-rust");
        let mut steps = 200usize;
        let mut index = 1;
        while index < args.len() {
            match args[index].to_str() {
                Some("--output") if index + 1 < args.len() => {
                    output = PathBuf::from(&args[index + 1]);
                    index += 1;
                }
                Some("--steps") if index + 1 < args.len() => {
                    steps = match args[index + 1].to_str().and_then(|x| x.parse().ok()) {
                        Some(x) => x,
                        None => {
                            eprintln!("--steps must be positive");
                            return ExitCode::from(2);
                        }
                    };
                    index += 1;
                }
                _ => {
                    eprintln!("run-tc-probe accepts [--output DIR] [--steps N]");
                    return ExitCode::from(2);
                }
            }
            index += 1;
        }
        ncm::probe::run_tc_probe(&root, &output, steps).map(|passed| {
            println!("real-text gate: {passed}");
        })
    } else if name == "summarize-maintenance" {
        ncm::reports::summarize_maintenance(&root).map(|_| ())
    } else if name == "summarize-domain-validation" {
        let from_reports = match args[1..].as_ref() {
            [] => false,
            [flag] if flag == "--from-reports" => true,
            _ => {
                return {
                    eprintln!("summarize-domain-validation accepts only --from-reports");
                    ExitCode::from(2)
                };
            }
        };
        ncm::research::summarize_domain_validation(&root, from_reports).map(|_| ())
    } else if name == "summarize-dependency-results" {
        if !args[1..].is_empty() {
            return {
                eprintln!("summarize-dependency-results takes no arguments");
                ExitCode::from(2)
            };
        }
        ncm::research::summarize_dependency_results(&root).map(|_| ())
    } else if name == "domain-source-baselines" {
        if !args[1..].is_empty() {
            return {
                eprintln!("domain-source-baselines takes no arguments");
                ExitCode::from(2)
            };
        }
        ncm::research::domain_source_baselines(&root).map(|_| ())
    } else if name == "alignment-diagnosis" {
        if !args[1..].is_empty() {
            return {
                eprintln!("alignment-diagnosis takes no arguments");
                ExitCode::from(2)
            };
        }
        ncm::research::diagnose_alignment(&root).map(|_| ())
    } else if name == "generate-recent-results" {
        let check = match args[1..].as_ref() {
            [] => false,
            [flag] if flag == "--check" => true,
            _ => {
                return {
                    eprintln!("generate-recent-results accepts only --check");
                    ExitCode::from(2)
                };
            }
        };
        ncm::paper::generate_recent_results(&root, check)
    } else if name == "generate-results" {
        let check = match args[1..].as_ref() {
            [] => false,
            [flag] if flag == "--check" => true,
            _ => {
                return {
                    eprintln!("generate-results accepts only --check");
                    ExitCode::from(2)
                };
            }
        };
        ncm::paper::generate_results(&root, check)
    } else if name == "summarize-reconstruction-capacity" {
        let mut run_dir = None;
        let mut output_dir = None;
        let mut index = 1;
        while index < args.len() {
            match args[index].to_str() {
                Some("--run") if index + 1 < args.len() => {
                    run_dir = Some(PathBuf::from(&args[index + 1]));
                    index += 1;
                }
                Some("--output") if index + 1 < args.len() => {
                    output_dir = Some(PathBuf::from(&args[index + 1]));
                    index += 1;
                }
                _ => {
                    return {
                        eprintln!(
                            "summarize-reconstruction-capacity requires --run DIR --output DIR"
                        );
                        ExitCode::from(2)
                    };
                }
            }
            index += 1;
        }
        match (run_dir, output_dir) {
            (Some(run), Some(output)) => {
                ncm::research::summarize_reconstruction_capacity(&run, &output)
            }
            (None, _) => Err("--run is required".into()),
            (_, None) => Err("--output is required".into()),
        }
    } else if name == "generate-bibliographies" {
        generate_bibliographies(&root, &args[1..])
    } else if name == "update-manifests" {
        update_manifests(&root, &args[1..])
    } else if name == "audit-papers" {
        audit_papers(&root, &args[1..])
    } else {
        eprintln!("unknown command: {name}; use --list");
        return ExitCode::from(2);
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
