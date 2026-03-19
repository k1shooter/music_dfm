import subprocess
import sys
from pathlib import Path


def _run_help(script_name: str) -> str:
    script = Path("scripts") / script_name
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_script_entrypoints_help_smoke():
    scripts = [
        "download_pop909.py",
        "preprocess.py",
        "train.py",
        "train_editflow.py",
        "sample.py",
        "eval.py",
        "visualize.py",
    ]
    for script in scripts:
        out = _run_help(script)
        assert "usage:" in out.lower()
