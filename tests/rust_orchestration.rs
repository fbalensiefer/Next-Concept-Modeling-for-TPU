#[cfg(unix)]
#[test]
fn named_worker_gets_module_arguments_source_path_and_exit_status() {
    use std::{fs, os::unix::fs::PermissionsExt, process::Command};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("ncm launcher {}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("fake python");
    fs::write(
        &fake,
        "#!/bin/sh\nprintf '%s\\n' \"$PWD\" \"$PYTHONPATH\" \"$@\"\nexit 33\n",
    )
    .unwrap();
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o700)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ncm-run"))
        .current_dir(root.join("paper"))
        .env("NCM_PYTHON", &fake)
        .args(["run-drafter-comparison", "--output", "a b; echo injected"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(33));
    let text = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<_> = text.lines().collect();
    assert_eq!(rows[0], root.to_str().unwrap());
    assert!(rows[1].starts_with(root.join("src").to_str().unwrap()));
    assert_eq!(
        &rows[2..],
        &[
            "-m",
            "ncm.workloads.validation.run_drafter_comparison",
            "--output",
            "a b; echo injected"
        ]
    );
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn unknown_command_fails_and_help_lists_commands() {
    use std::process::Command;
    let binary = env!("CARGO_BIN_EXE_ncm-run");
    assert_eq!(
        Command::new(binary)
            .arg("unregistered")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    let output = Command::new(binary).arg("--list").output().unwrap();
    assert!(output.status.success());
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("build-papers")
    );
}

#[cfg(unix)]
#[test]
fn rust_orders_paper_checks_and_stops_on_worker_failure() {
    use std::{fs, os::unix::fs::PermissionsExt, process::Command};
    let dir = std::env::temp_dir().join(format!("ncm audit {}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let python = dir.join("worker");
    let log = dir.join("calls");
    fs::write(
        &python,
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$NCM_AUDIT_LOG\"\nexit \"${NCM_FAKE_EXIT:-0}\"\n",
    )
    .unwrap();
    fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
    let invoke = |exit: &str| {
        Command::new(env!("CARGO_BIN_EXE_ncm-run"))
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("NCM_PYTHON", &python)
            .env("NCM_AUDIT_LOG", &log)
            .env("NCM_FAKE_EXIT", exit)
            .args(["audit-papers", "--build-dir", "example"])
            .output()
            .unwrap()
    };
    assert!(invoke("0").status.success());
    assert_eq!(
        fs::read_to_string(&log)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        vec![
            "-m ncm.workloads.papers.generate_results --check",
            "-m ncm.workloads.papers.generate_recent_results --check",
            "-m ncm.workloads.papers.generate_bibliographies --check",
            "-m ncm.workloads.papers.audit_papers --build-dir example"
        ]
    );
    fs::remove_file(&log).unwrap();
    assert_eq!(invoke("7").status.code(), Some(7));
    assert_eq!(fs::read_to_string(&log).unwrap().lines().count(), 1);
    fs::remove_dir_all(dir).unwrap();
}
