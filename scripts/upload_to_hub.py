#!/usr/bin/env python
"""Phase 6: publish trained checkpoints to HuggingFace Hub with model cards.

Discovers ``checkpoints/<policy>/checkpoints/last/pretrained_model/`` for
each trained policy AND ``outputs/eval/<policy>/eval_metrics.json`` from the
Phase 5 eval. Generates a model card and publishes to
``adimunot21/<policy>-lab`` (or whatever ``--repo-prefix`` is set to).

REQUIREMENTS:
  - HF_TOKEN in environment (or .env). Get one at
    https://huggingface.co/settings/tokens with WRITE scope.
  - Phase 5 eval must have run for any policy you want to publish (the model
    card needs the success_rate / avg_max_reward).

Usage::

    # Dry-run (renders cards, no Hub calls — verify before publishing for real):
    python scripts/upload_to_hub.py --dry-run

    # Publish all 3:
    python scripts/upload_to_hub.py

    # Publish just one:
    python scripts/upload_to_hub.py --only diffusion_pusht
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from lerobot_pusht_lab.hub.upload import UploadSpec, publish


# Default policy → publish-spec template. Update if checkpoint layout changes.
DEFAULT_POLICIES: list[dict] = [
    {
        "name": "diffusion_pusht",
        "policy_type": "diffusion",
        "checkpoint": "checkpoints/diffusion_pusht/checkpoints/last/pretrained_model",
        "eval_metrics": "outputs/eval/diffusion_pusht/eval_metrics.json",
    },
    {
        "name": "act_pusht",
        "policy_type": "act",
        "checkpoint": "checkpoints/act_pusht/checkpoints/last/pretrained_model",
        "eval_metrics": "outputs/eval/act_pusht/eval_metrics.json",
    },
    {
        "name": "mlp_bc_pusht",
        "policy_type": "mlp_bc",
        "checkpoint": "checkpoints/mlp_bc_pusht/checkpoints/last/pretrained_model",
        "eval_metrics": "outputs/eval/mlp_bc_pusht/eval_metrics.json",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--repo-prefix",
        default="adimunot21",
        help="HF Hub user/org name. Repos will be <prefix>/<policy>-lab.",
    )
    p.add_argument(
        "--only",
        action="append",
        help="Only publish these policy names (repeatable). Default: all.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Make repos private. Default: public.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Render cards but skip create_repo + upload. Useful for verification.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("upload_to_hub")

    # Load .env so HF_TOKEN propagates without manual export
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        logger.error("HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens "
                     "(WRITE scope), then add to .env")
        return 1

    specs_to_run = DEFAULT_POLICIES
    if args.only:
        specs_to_run = [s for s in DEFAULT_POLICIES if s["name"] in args.only]
        if not specs_to_run:
            logger.error("--only filter matched zero policies. Names: %s",
                         [s["name"] for s in DEFAULT_POLICIES])
            return 1

    train_config_paths = {
        s["name"]: Path(s["checkpoint"]) / "train_config.json"
        for s in specs_to_run
    }
    # MLP-BC saves config.json (no train_config.json) since it doesn't use lerobot-train
    for name in train_config_paths:
        if not train_config_paths[name].exists():
            alt = train_config_paths[name].parent / "config.json"
            if alt.exists():
                train_config_paths[name] = alt

    results: list[tuple[str, str | None]] = []
    for spec_dict in specs_to_run:
        ckpt_dir = Path(spec_dict["checkpoint"])
        if not ckpt_dir.exists():
            logger.warning("[skip] %s — checkpoint not found at %s", spec_dict["name"], ckpt_dir)
            results.append((spec_dict["name"], None))
            continue
        if not Path(spec_dict["eval_metrics"]).exists():
            logger.warning("[skip] %s — eval metrics not found at %s "
                           "(run scripts/eval_all.py first)",
                           spec_dict["name"], spec_dict["eval_metrics"])
            results.append((spec_dict["name"], None))
            continue

        spec = UploadSpec(
            pretrained_model_dir=ckpt_dir,
            eval_metrics_path=Path(spec_dict["eval_metrics"]),
            train_config_path=train_config_paths[spec_dict["name"]],
            repo_id=f"{args.repo_prefix}/{spec_dict['name'].replace('_', '-')}-lab",
            policy_type=spec_dict["policy_type"],
            private=args.private,
        )
        try:
            url = publish(spec, token=token, dry_run=args.dry_run)
            results.append((spec_dict["name"], url))
        except Exception as e:
            logger.exception("publish failed for %s: %s", spec_dict["name"], e)
            results.append((spec_dict["name"], None))

    print("\n=== Summary ===")
    for name, url in results:
        status = url or "FAILED/SKIPPED"
        print(f"  {name}: {status}")

    n_ok = sum(1 for _, url in results if url is not None)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
