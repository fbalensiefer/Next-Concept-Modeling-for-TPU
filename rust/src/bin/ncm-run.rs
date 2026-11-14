//! Minimal Rust process orchestrator for existing Python/JAX workloads.

use std::{
    env,
    process::{Command, ExitCode},
};

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    let Some(script) = args.next() else {
        eprintln!("usage: ncm-run <python-script> [script arguments ...]");
        return ExitCode::from(2);
    };
    let path = std::path::Path::new(&script);
    if path.extension().and_then(|x| x.to_str()) != Some("py") || !path.is_file() {
        eprintln!("expected an existing .py entrypoint");
        return ExitCode::from(2);
    }
    let python = env::var_os("NCM_PYTHON").unwrap_or_else(|| "python3".into());
    match Command::new(python).arg(script).args(args).status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1) as u8),
        Err(error) => {
            eprintln!("could not start JAX worker: {error}");
            ExitCode::from(1)
        }
    }
}
