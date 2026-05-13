"""
ImageNet normalization guard for LeRobot Diffusion Policy training on PushT.

BUG HISTORY AND DIAGNOSTIC EVIDENCE
=====================================
Symptom observed: Diffusion Policy trained on lerobot/pusht with a ResNet18 vision
backbone only achieved ~20% pc_success (~35% by max-overlap criterion), roughly half
the published LeRobot benchmark of 65%.

Hypothesis tested: LeRobot's make_pre_post_processors passes the dataset's actual pixel
distribution stats to the NormalizerProcessorStep for VISUAL features. For PushT (mostly
white background), those dataset stats are mean≈[0.97, 0.98, 0.98] / std≈[0.10, 0.07,
0.08]. ResNet18 with ImageNet pretrained weights requires inputs normalized with canonical
ImageNet stats: mean=[0.485, 0.456, 0.406] / std=[0.229, 0.224, 0.225]. If the wrong
stats are used, the encoder is fed inputs outside its pretraining distribution and the
policy learns a degraded representation.

Diagnostic verdict (lerobot==0.5.1, commit f90db58c, 2026-05-12):
  The bug does NOT exist in the installed version. Two pieces of evidence:
  1. lerobot/datasets/factory.py defines IMAGENET_STATS and make_dataset() overwrites
     camera feature stats with those values when use_imagenet_stats=True (the default).
  2. Inspecting the local 50K-step checkpoint's safetensors confirms:
       observation.image.mean = [0.485, 0.456, 0.406]  ← ImageNet ✓
       observation.image.std  = [0.229, 0.224, 0.225]  ← ImageNet ✓
     whereas the raw PushT dataset stats are mean≈[0.972, 0.981, 0.977].

The 20% failure was due to: (a) under-training (50K×batch8 = 400K samples vs 12.8M
needed), and (b) the stricter pc_success eval criterion (env's sustained-overlap signal)
vs the model card's max-overlap ≥ 0.95 criterion used during the 200K RunPod run.

WHAT THIS MODULE PROVIDES
===========================
Three utilities, used as a safety harness rather than a bug fix:

  preflight_check(config_path)
      Simulate what lerobot-train will do with image normalization stats before
      spending GPU time. Prints raw dataset stats, whether use_imagenet_stats will
      override them, and the final values the normalizer will receive. Exits non-zero
      if the final values are not ImageNet stats, so train_diffusion.sh can abort.

  verify_checkpoint_stats(checkpoint_dir)
      Inspect a saved checkpoint's preprocessor safetensors and print the image
      mean/std. Use after training to confirm the checkpoint was saved with the
      correct normalization. Also guards against loading old-lerobot checkpoints
      (before use_imagenet_stats was added) into the eval harness.

  apply_imagenet_override(preprocessor, input_features)
      Surgical in-place patch: iterate over VISUAL-typed features in the preprocessor's
      NormalizerProcessorStep and overwrite their mean/std with ImageNet values. This
      is the defensive fallback for loading checkpoints trained with older lerobot.
      Logs a WARNING if it has to patch anything, INFO if stats were already correct.
      Returns the (possibly-patched) preprocessor.

If apply_imagenet_override() emits a WARNING, one of these is true:
  - You loaded a checkpoint from a lerobot version that lacked use_imagenet_stats
  - Someone set use_imagenet_stats=false in the training config
  - There is a regression in a newer lerobot version
All three scenarios degrade model performance significantly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Canonical ImageNet normalization constants — required by ResNet/ViT backbones
# pretrained on ImageNet. Shape (3, 1, 1) to broadcast over CHW tensors.
IMAGENET_MEAN: list[list[list[float]]] = [[[0.485]], [[0.456]], [[0.406]]]
IMAGENET_STD: list[list[list[float]]] = [[[0.229]], [[0.224]], [[0.225]]]

_IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)  # (3,1,1)
_IMAGENET_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32)    # (3,1,1)
_ATOL = 1e-4  # tolerance for floating-point equality check


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_imagenet(mean: torch.Tensor, std: torch.Tensor) -> bool:
    """Return True iff mean/std tensors match ImageNet values within _ATOL.

    Flattens both sides before comparison so shapes (3,), (3,1,1), or any
    equivalent do not cause broadcasting mismatches.
    """
    return (
        torch.allclose(mean.cpu().float().flatten(), _IMAGENET_MEAN_T.flatten(), atol=_ATOL)
        and torch.allclose(std.cpu().float().flatten(), _IMAGENET_STD_T.flatten(), atol=_ATOL)
    )


def _find_normalizer_step(preprocessor: Any) -> Any | None:
    """Find the NormalizerProcessorStep inside a PolicyProcessorPipeline.

    Iterates over pipeline steps and returns the first step that is an instance
    of NormalizerProcessorStep (or its mixin). Returns None if not found.
    """
    try:
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
    except ImportError:
        logger.warning("lerobot.processor.normalize_processor not found; cannot inspect steps")
        return None

    for step in getattr(preprocessor, "steps", []):
        if isinstance(step, NormalizerProcessorStep):
            return step
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_imagenet_override(
    preprocessor: Any,
    input_features: dict[str, Any],
    device: str = "cpu",
) -> Any:
    """Ensure every VISUAL feature in the preprocessor uses ImageNet mean/std.

    Iterates over ``input_features``, finds any with ``type == FeatureType.VISUAL``,
    and checks the corresponding mean/std in the preprocessor's NormalizerProcessorStep.
    If they already match ImageNet values, logs INFO. If not, patches them in-place
    and logs WARNING.

    This mutates the preprocessor's internal ``_tensor_stats`` and ``stats`` dicts
    directly. Since the preprocessor's ``state_dict()`` reads from ``_tensor_stats``,
    any subsequent ``save_pretrained`` call will persist the corrected values.

    Args:
        preprocessor: A ``PolicyProcessorPipeline`` returned by make_pre_post_processors.
        input_features: Policy config's ``input_features`` dict (maps key → PolicyFeature).
        device: Target device for the patched tensors.

    Returns:
        The (possibly mutated) preprocessor. The return value is the same object.
    """
    try:
        from lerobot.configs.types import FeatureType
    except ImportError as exc:
        raise ImportError("lerobot is required for apply_imagenet_override") from exc

    normalizer = _find_normalizer_step(preprocessor)
    if normalizer is None:
        logger.warning("No NormalizerProcessorStep found; cannot apply ImageNet override")
        return preprocessor

    any_patched = False
    for feat_name, feat in input_features.items():
        if feat.type is not FeatureType.VISUAL:
            continue

        mean_t = _IMAGENET_MEAN_T.to(device)
        std_t = _IMAGENET_STD_T.to(device)

        current = normalizer._tensor_stats.get(feat_name, {})
        current_mean = current.get("mean")
        current_std = current.get("std")

        if (
            current_mean is None
            or current_std is None
            or not _is_imagenet(current_mean, current_std)
        ):
            logger.warning(
                "apply_imagenet_override: '%s' mean/std are NOT ImageNet — patching.\n"
                "  current mean = %s\n"
                "  current std  = %s\n"
                "  new mean     = %s\n"
                "  new std      = %s",
                feat_name,
                current_mean.squeeze().tolist() if current_mean is not None else None,
                current_std.squeeze().tolist() if current_std is not None else None,
                mean_t.squeeze().tolist(),
                std_t.squeeze().tolist(),
            )
            normalizer._tensor_stats.setdefault(feat_name, {})["mean"] = mean_t
            normalizer._tensor_stats.setdefault(feat_name, {})["std"] = std_t
            # Keep the plain-Python stats dict in sync so .to() calls work correctly.
            normalizer.stats.setdefault(feat_name, {})["mean"] = IMAGENET_MEAN
            normalizer.stats.setdefault(feat_name, {})["std"] = IMAGENET_STD
            any_patched = True
        else:
            logger.info(
                "apply_imagenet_override: '%s' already has correct ImageNet stats.", feat_name
            )

        # Always print the authoritative values for human confirmation.
        final_mean = normalizer._tensor_stats[feat_name]["mean"].squeeze().tolist()
        final_std = normalizer._tensor_stats[feat_name]["std"].squeeze().tolist()
        logger.info(
            "  %s — mean=%.4f %.4f %.4f  std=%.4f %.4f %.4f",
            feat_name, *final_mean, *final_std,
        )

    if not any_patched:
        logger.info("apply_imagenet_override: all VISUAL features confirmed ImageNet — no patch needed")
    return preprocessor


def verify_checkpoint_stats(checkpoint_dir: str | Path) -> dict[str, dict[str, list]]:
    """Read a saved checkpoint's preprocessor safetensors and print image norm stats.

    Finds the ``policy_preprocessor_step_*_normalizer_processor.safetensors`` file,
    extracts all keys containing ``image`` + (``mean`` or ``std``), and prints them.
    Returns a dict mapping ``"<feature>.<stat>"`` → value list.

    Args:
        checkpoint_dir: Path to the checkpoint directory. Accepts either the top-level
            step directory (e.g. ``checkpoints/diffusion_pusht/checkpoints/200000``) or
            the ``pretrained_model/`` subdirectory directly.

    Returns:
        Dict mapping ``"observation.image.mean"`` etc. → squeezed tensor as a list.

    Raises:
        FileNotFoundError: if no normalizer safetensors is found.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("safetensors package is required for verify_checkpoint_stats") from exc

    checkpoint_dir = Path(checkpoint_dir)
    pretrained = (
        checkpoint_dir / "pretrained_model"
        if (checkpoint_dir / "pretrained_model").exists()
        else checkpoint_dir
    )

    sf_files = sorted(pretrained.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not sf_files:
        raise FileNotFoundError(
            f"No normalizer safetensors found in {pretrained}\n"
            "Expected: policy_preprocessor_step_*_normalizer_processor.safetensors"
        )

    sf_path = sf_files[0]
    logger.info("Inspecting: %s", sf_path)

    image_stats: dict[str, dict[str, list]] = {}
    all_ok = True
    with safe_open(str(sf_path), framework="pt", device="cpu") as st:
        for key in st.keys():
            parts = key.rsplit(".", 1)
            feat_key, stat_name = parts[0], parts[1] if len(parts) == 2 else ""
            if "image" not in feat_key or stat_name not in ("mean", "std"):
                continue
            t = st.get_tensor(key)
            val = t.squeeze().tolist()
            image_stats.setdefault(feat_key, {})[stat_name] = val
            logger.info("  %s = %s", key, val)

    # Validate all found image features
    for feat_key, stats in image_stats.items():
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            logger.warning("%s: missing mean or std", feat_key)
            all_ok = False
            continue
        mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32)
        if not _is_imagenet(mean_t, std_t):
            logger.warning(
                "%s: stats are NOT ImageNet!\n  mean=%s\n  std=%s",
                feat_key, mean, std,
            )
            all_ok = False
        else:
            logger.info("%s: ImageNet stats confirmed ✓", feat_key)

    if not image_stats:
        logger.warning("No image features found in normalizer safetensors")
    elif all_ok:
        print("verify_checkpoint_stats: all image features have correct ImageNet stats ✓")
    else:
        print(
            "verify_checkpoint_stats: WARNING — one or more image features have non-ImageNet stats.\n"
            "  Load this checkpoint with apply_imagenet_override() before eval.",
            file=sys.stderr,
        )

    return image_stats


def preflight_check(config_path: str | Path) -> None:
    """Simulate lerobot-train's image normalization and print the stats it will use.

    Loads the dataset specified in the config, applies ``use_imagenet_stats`` logic
    (same as lerobot/datasets/factory.py:make_dataset), and prints the final image
    mean/std values. Exits with code 1 if the final values are not ImageNet stats, so
    train_diffusion.sh can abort before wasting GPU time.

    Does NOT download video data or build delta-timestamps — only the metadata is needed
    to read feature stats.

    Args:
        config_path: Path to the draccus YAML config (e.g. configs/diffusion_pusht.yaml).
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        raw_cfg: dict = yaml.safe_load(f) or {}

    repo_id: str | None = raw_cfg.get("dataset", {}).get("repo_id")
    if repo_id is None:
        logger.error("Config has no dataset.repo_id")
        sys.exit(1)

    # use_imagenet_stats defaults to True in lerobot/configs/default.py.
    # The YAML override file only sets it to False explicitly, so a missing key → True.
    use_imagenet_stats: bool = raw_cfg.get("dataset", {}).get("use_imagenet_stats", True)

    print(f"Pre-flight image normalization check for config: {config_path}")
    print(f"  dataset.repo_id        = {repo_id}")
    print(f"  use_imagenet_stats     = {use_imagenet_stats}")

    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    except ImportError as exc:
        logger.error("lerobot is required for preflight_check: %s", exc)
        sys.exit(1)

    print("  loading dataset metadata (no video download) …")
    meta = LeRobotDatasetMetadata(repo_id)

    all_ok = True
    for key in meta.camera_keys:
        raw_mean = torch.tensor(meta.stats[key]["mean"], dtype=torch.float32).squeeze().tolist()
        raw_std = torch.tensor(meta.stats[key]["std"], dtype=torch.float32).squeeze().tolist()

        if use_imagenet_stats:
            final_mean = [x[0][0] for x in IMAGENET_MEAN]   # [0.485, 0.456, 0.406]
            final_std = [x[0][0] for x in IMAGENET_STD]     # [0.229, 0.224, 0.225]
        else:
            final_mean = raw_mean
            final_std = raw_std

        is_ok = (
            use_imagenet_stats  # overriding to ImageNet → always correct
            or _is_imagenet(
                torch.tensor(final_mean, dtype=torch.float32),
                torch.tensor(final_std, dtype=torch.float32),
            )
        )
        status = "✓" if is_ok else "✗ NON-IMAGENET"
        if not is_ok:
            all_ok = False

        print(f"\n  Feature: {key}")
        print(f"    Raw dataset mean = {[round(v, 4) for v in raw_mean]}")
        print(f"    Raw dataset std  = {[round(v, 4) for v in raw_std]}")
        print(f"    Training will use mean = {[round(v, 4) for v in final_mean]}  {status}")
        print(f"    Training will use std  = {[round(v, 4) for v in final_std]}  {status}")

    if not all_ok:
        print(
            "\nERROR: One or more visual features will NOT use ImageNet stats.\n"
            "Set use_imagenet_stats=true in your dataset config (it is the default;\n"
            "check that no override sets it to false).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nPre-flight PASSED — image normalization is correct ✓")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Usage:
      python -m lerobot_pusht_lab.training.imagenet_norm_guard <config_path>
      python -m lerobot_pusht_lab.training.imagenet_norm_guard --verify <checkpoint_dir>
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Verify/enforce ImageNet image normalization stats"
    )
    parser.add_argument(
        "path",
        help="Config YAML (for preflight) or checkpoint directory (for --verify)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Inspect a saved checkpoint's preprocessor instead of doing a preflight check",
    )
    args = parser.parse_args()

    if args.verify:
        verify_checkpoint_stats(args.path)
    else:
        preflight_check(args.path)
