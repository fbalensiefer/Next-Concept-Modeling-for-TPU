#[cfg(unix)]
#[test]
fn native_drafter_command_does_not_invoke_python_launcher() {
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
    for (command, output_arg, expected_error) in [
        (
            "run-decisive-benchmark",
            "--output",
            "--model-config is required",
        ),
        (
            "run-drafter-comparison",
            "--output",
            "--target-config is required",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_ncm-run"))
            .current_dir(root.join("paper"))
            .env("NCM_PYTHON", &fake)
            .args([command, output_arg, "native-output"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected_error));
    }
    assert!(!dir.join("calls").exists());
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
    let listing = String::from_utf8(output.stdout).unwrap();
    assert!(listing.contains("build-papers"));
    assert!(listing.contains("summarize-maintenance"));
}

#[cfg(unix)]
#[test]
fn rust_paper_audit_uses_native_tools_without_starting_python() {
    use std::{fs, os::unix::fs::PermissionsExt, process::Command};
    let dir = std::env::temp_dir().join(format!("ncm audit {}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let python = dir.join("worker");
    fs::write(&python, "#!/bin/sh\nexit 71\n").unwrap();
    fs::set_permissions(&python, fs::Permissions::from_mode(0o700)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ncm-run"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("NCM_PYTHON", &python)
        .arg("audit-papers")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!dir.join("calls").exists());
    fs::remove_dir_all(dir).unwrap();
}
