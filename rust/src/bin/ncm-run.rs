//! Command dispatch and paper-build orchestration. Model computation stays in JAX.
use ncm::commands::COMMANDS;
use ncm::rpc::{Request, Response};
use serde_json::{Value, json};
use std::{
    env,
    ffi::OsString,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
};
fn worker(root: &Path, module: &str, args: &[OsString]) -> Result<(), String> {
    let python = env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into());
    let mut paths = vec![root.join("src")];
    if let Some(old) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&old));
    }
    let status = Command::new(python)
        .args(["-m", module])
        .args(args)
        .current_dir(root)
        .env(
            "PYTHONPATH",
            env::join_paths(paths).map_err(|e| e.to_string())?,
        )
        .status()
        .map_err(|e| format!("could not start worker: {e}"))?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}
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
fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "--list" {
        println!(
            "ncm-run <command> [arguments]\n\nCommands:\n  engine-info\n  build-papers\n  train\n  real-text-reconstruction\n  summarize-maintenance\n  summarize-domain-validation\n  summarize-dependency-results\n  domain-source-baselines\n  generate-bibliographies\n  update-manifests\n  audit-papers"
        );
        for (name, _) in COMMANDS {
            println!("  {name}");
        }
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
    } else if let Some((_, module)) = COMMANDS.iter().find(|(n, _)| *n == name) {
        worker(&root, module, &args[1..])
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
