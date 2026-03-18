import subprocess
import sys


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "music_graph_dfm", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "music-graph-dfm" in result.stdout


def test_cli_subcommand_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "music_graph_dfm", "preprocess", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--span-resolution" in result.stdout
