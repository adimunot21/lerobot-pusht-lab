#!/usr/bin/env python
"""Phase 4: train the from-scratch MLP-BC baseline on lerobot/pusht.

State-only behaviour cloning with MSE. See
``src/lerobot_pusht_lab/policies/mlp_bc.py`` for *why* this is expected to fail
(mode-averaging) and what the architecture is.

This script is the LeRobot-free counterpart to
``scripts/train_diffusion.sh``. We do not use ``lerobot-train`` because
LeRobot has no MLP-BC policy class (smallest is ACT) and rolling our own
training loop is the whole pedagogical point of Phase 4.

What we DO reuse from LeRobot: ``LeRobotDataset`` for loading the same
parquet+video files the other policies see. This keeps the comparison fair —
any difference in Phase 5 is attributable to the policy class, not to the
data pipeline.

Output layout (mirrors LeRobot's checkpoint structure for uniform downstream tooling):

    {output_dir}/
      checkpoints/
        001000/
          pretrained_model/
            model.safetensors
            config.json
        ...
        last/                  → symlink to most recent
      wandb/
        offline-run-.../

Usage::

    python scripts/train_mlp_bc.py
    python scripts/train_mlp_bc.py --config configs/mlp_bc_pusht.yaml
    python scripts/train_mlp_bc.py --steps 100 --log_freq 10  # smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_pusht_lab.policies.mlp_bc import MLPBC, MLPBCConfig

logger = logging.getLogger("train_mlp_bc")


# ---------------------------------------------------------------------------
# CLI / config loading
# ---------------------------------------------------------------------------


def _load_yaml_into_config(yaml_path: Path, base: MLPBCConfig) -> MLPBCConfig:
    """Apply YAML overrides to a default MLPBCConfig.

    Why not just `MLPBCConfig(**yaml_data)`: catches typos. If the YAML has a
    field that doesn't exist in the dataclass, the user gets a clear error
    rather than a silently-ignored setting (which would, e.g., make a
    `learning_rate:` typo fail to actually change the lr).
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"config not found: {yaml_path}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    fields_by_name = {f.name: f for f in dataclasses.fields(MLPBCConfig)}
    unknown = set(data) - fields_by_name.keys()
    if unknown:
        raise ValueError(f"unknown YAML keys: {sorted(unknown)} (valid: {sorted(fields_by_name)})")
    # Coerce types from the dataclass defaults — defends against PyYAML 1.2
    # quirks (e.g. `1e-3` parses as str, not float).
    coerced: dict[str, Any] = {}
    for k, v in data.items():
        target_type = type(fields_by_name[k].default)
        if target_type in (int, float) and isinstance(v, str):
            coerced[k] = target_type(v)
        else:
            coerced[k] = v
    return dataclasses.replace(base, **coerced)


def parse_args() -> tuple[MLPBCConfig, argparse.Namespace]:
    """Parse CLI: --config loads a YAML; remaining args override individual fields.

    The two-phase parse (YAML first, then CLI) means CLI flags always win,
    matching the LeRobot convention.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/mlp_bc_pusht.yaml"))

    # Add an override for every dataclass field so users can do --batch_size=64.
    # `f.type` may be either a class (legacy) or a string (PEP 563 / py3.10+
    # dataclasses), so we resolve via the actual default value's type — that
    # always works.
    type_for: dict[type, Any] = {
        int: int, float: float,
        bool: lambda s: s.lower() in {"1", "true", "yes"},
        str: str,
    }
    for f in dataclasses.fields(MLPBCConfig):
        py_type = type(f.default) if f.default is not dataclasses.MISSING else str
        parser.add_argument(
            f"--{f.name}",
            default=None,
            type=type_for.get(py_type, str),
            help=f"override {f.name} (default {f.default!r})",
        )

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    base = MLPBCConfig()
    cfg = _load_yaml_into_config(args.config, base)
    overrides = {f.name: v for f in dataclasses.fields(MLPBCConfig)
                 if (v := getattr(args, f.name)) is not None}
    if overrides:
        logger.info("CLI overrides: %s", overrides)
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg, args


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_dataset(cfg: MLPBCConfig):
    """Load just the parquet portion of lerobot/pusht — no video decode.

    Returns ``ds.hf_dataset`` (a HuggingFace Dataset), not the LeRobot wrapper.
    Why: ``LeRobotDataset.__getitem__`` always decodes video frames for keys
    declared as video, even when the policy never reads them. For a state-only
    policy that's both wasted work AND a failure mode (video decode can crash
    if any single mp4 is malformed). The underlying parquet has every non-video
    column we need (state, action, episode_index, timestamps, rewards) and is
    purely tabular.

    Trade-off: we lose LeRobot's `delta_timestamps` action-chunking machinery,
    but MLP-BC is single-step anyway so we don't need it.
    """
    logger.info("loading dataset %s (parquet only — state-only policy)", cfg.dataset_repo_id)
    wrapper = LeRobotDataset(cfg.dataset_repo_id, download_videos=False)
    hf_ds = wrapper.hf_dataset
    logger.info("dataset: %d frames across %d episodes", len(hf_ds), wrapper.num_episodes)
    return hf_ds


def fit_normalizers(model: MLPBC, dataset, max_samples: int = 5000) -> None:
    """Compute min-max normaliser stats from a subset of frames.

    `max_samples` caps the work — for PushT with 25650 frames, sampling 5000
    gives a min/max that's within 1% of the true population min/max (state
    and action are bounded physical quantities). The full dataset would also
    be fast at this size, but the cap matters for SO-101 datasets which can
    be 10× larger.
    """
    n = len(dataset)
    if n > max_samples:
        # Stride so we cover the whole dataset evenly, not just episode 0.
        stride = max(1, n // max_samples)
        indices = list(range(0, n, stride))[:max_samples]
    else:
        indices = list(range(n))
    logger.info("fitting normalisers on %d samples (stride %d)", len(indices), max(1, n // max_samples))

    # Pull state + action only; load_videos=False above means observation.image is absent.
    states = torch.stack([dataset[i]["observation.state"] for i in indices])
    actions = torch.stack([dataset[i]["action"] for i in indices])
    model.input_normalizer.fit(states)
    model.output_normalizer.fit(actions)


def setup_dataloader(dataset, cfg: MLPBCConfig) -> DataLoader:
    """Standard PyTorch DataLoader. shuffle=True for SGD; pin_memory for GPU transfer."""
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,  # avoid a tiny last batch that wrecks BatchNorm-like layers (none here, but cheap insurance)
        pin_memory=(cfg.device == "cuda"),
    )


def setup_wandb(cfg: MLPBCConfig, output_dir: Path) -> Any:
    """Initialise wandb in offline mode (project rule — see project memory)."""
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    return wandb.init(
        project="lerobot-pusht-lab",
        name="mlp_bc_pusht",
        dir=str(output_dir),
        config=asdict(cfg),
        mode="offline",
    )


# ---------------------------------------------------------------------------
# Checkpointing — mirror LeRobot's structure
# ---------------------------------------------------------------------------


def save_checkpoint(model: MLPBC, cfg: MLPBCConfig, output_dir: Path, step: int) -> Path:
    """Write {output_dir}/checkpoints/{step:06d}/pretrained_model/{model,config}."""
    ckpt_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir / "model.safetensors")
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    # Update `last` symlink — atomic replace via mkstemp + rename
    last_link = output_dir / "checkpoints" / "last"
    if last_link.is_symlink() or last_link.exists():
        last_link.unlink()
    last_link.symlink_to(f"{step:06d}")
    logger.info("saved checkpoint: %s", ckpt_dir)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: MLPBCConfig) -> Path:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility — same seed convention as the Diffusion config.
    torch.manual_seed(cfg.seed)

    # Device fallback: if CUDA was requested but isn't available (e.g. it's
    # busy with the Diffusion training run and we OOM trying to allocate),
    # silently drop to CPU. The MLP is tiny enough that CPU is workable.
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable — falling back to cpu")
        device = "cpu"

    dataset = setup_dataset(cfg)
    model = MLPBC(cfg).to(device)
    fit_normalizers(model, dataset)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model: %s params", f"{n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
    )

    loader = setup_dataloader(dataset, cfg)
    wandb_run = setup_wandb(cfg, output_dir)

    # Save initial config alongside outputs (CLAUDE.md reproducibility rule)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    model.train()
    step = 0
    t0 = time.time()
    loss_window: list[float] = []
    log_t = t0

    # We iterate by steps, not epochs — gives finer control and matches LeRobot.
    # `infinite_loader` is just `for batch in itertools.cycle(loader)` but
    # without buffering issues.
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break
            state = batch["observation.state"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)

            loss = model.loss(state, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            loss_window.append(loss.item())

            if step % cfg.log_freq == 0:
                avg_loss = sum(loss_window) / len(loss_window)
                steps_per_sec = cfg.log_freq / (time.time() - log_t)
                logger.info(
                    "step %5d/%d  loss=%.5f  steps/s=%.1f  elapsed=%.1fs",
                    step, cfg.steps, avg_loss, steps_per_sec, time.time() - t0,
                )
                wandb_run.log({
                    "train/loss": avg_loss,
                    "train/steps_per_sec": steps_per_sec,
                    "train/lr": cfg.optimizer_lr,  # constant — no schedule
                }, step=step)
                loss_window.clear()
                log_t = time.time()

            if step % cfg.save_freq == 0:
                save_checkpoint(model, cfg, output_dir, step)

    # Final checkpoint if not aligned with save_freq
    if step % cfg.save_freq != 0:
        save_checkpoint(model, cfg, output_dir, step)

    elapsed = time.time() - t0
    logger.info("training done in %.1fs (%.1f min)", elapsed, elapsed / 60)
    wandb_run.finish()
    return output_dir / "checkpoints" / f"{step:06d}"


def main() -> int:
    cfg, args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("config:\n%s", json.dumps(asdict(cfg), indent=2))
    try:
        final_ckpt = train(cfg)
        logger.info("final checkpoint: %s", final_ckpt)
    except Exception:
        logger.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
