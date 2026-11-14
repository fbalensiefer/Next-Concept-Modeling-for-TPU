//! Command dispatch and paper-build orchestration. Model computation stays in JAX.
use ncm::commands::COMMANDS;
use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
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
    for module in [
        "generate_results",
        "generate_recent_results",
        "generate_bibliographies",
    ] {
        worker(root, &format!("ncm.workloads.papers.{module}"), &[])?;
    }
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
    worker(
        root,
        "ncm.workloads.papers.audit_papers",
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
fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "--list" {
        println!("ncm-run <command> [arguments]\n\nCommands:\n  build-papers");
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
    let result = if name == "build-papers" {
        build_papers(&root, &args[1..])
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
