#!/usr/bin/env python
"""Train FSNTG graph DFM model (default velocity parameterization)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from music_graph_dfm.train_runner import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    metrics = run_training(cfg_dict, use_edit_ops=False)
    print(metrics)


if __name__ == "__main__":
    main()
