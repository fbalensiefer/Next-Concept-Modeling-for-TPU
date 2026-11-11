import os
from pathlib import Path
import subprocess
import sys


def test_four_device_spmd_matches_global_reference():
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4")
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("spmd_check.py"))],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
