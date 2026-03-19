#!/usr/bin/env python
"""Download POP909 dataset repository."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone POP909 dataset repository")
    parser.add_argument("--target-dir", type=str, default="data/raw/POP909-Dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    repo = "https://github.com/music-x-lab/POP909-Dataset"

    if target.exists() and any(target.iterdir()):
        if not args.force:
            print(json.dumps({"status": "exists", "path": str(target)}, indent=2))
            return
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", repo, str(target)])
    print(json.dumps({"status": "downloaded", "path": str(target)}, indent=2))


if __name__ == "__main__":
    main()
