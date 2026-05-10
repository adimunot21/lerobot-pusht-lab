#!/usr/bin/env python
"""Evaluate the trained MLP-BC checkpoint against gym-pusht.

Phase 5 deliverable for the MLP-BC slot. Diffusion + ACT use ``lerobot-eval``;
MLP-BC needs its own runner because we never wrapped it in the LeRobot policy
framework.

Output: ``outputs/eval/mlp_bc_pusht/eval_metrics.json`` — same schema the
comparison script (``compare_policies.py``) expects.

Usage::

    python scripts/eval_mlp_bc.py
    python scripts/eval_mlp_bc.py --checkpoint checkpoints/mlp_bc_pusht/checkpoints/last
    python scripts/eval_mlp_bc.py --n-episodes 10 --device cpu  # smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot_pusht_lab.eval.runner import MLPBCAdapter, run_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/mlp_bc_pusht/checkpoints/last"),
        help="Path to checkpoint directory (the one containing pretrained_model/).",
    )
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--base-seed", type=int, default=100000)
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--max-steps", type=int, default=300, help="Cap per episode (env default).")
    p.add_argument("--device", default="cpu", help="MLP-BC is tiny — CPU is fine.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eval/mlp_bc_pusht"),
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
    logger = logging.getLogger("eval_mlp_bc")

    if not args.checkpoint.exists():
        logger.error("checkpoint not found: %s", args.checkpoint)
        logger.error("train first: python scripts/train_mlp_bc.py")
        return 1

    logger.info("loading MLP-BC from %s", args.checkpoint)
    adapter = MLPBCAdapter.from_checkpoint(args.checkpoint, device=args.device)

    metrics = run_evaluation(
        policy=adapter,
        n_episodes=args.n_episodes,
        base_seed=args.base_seed,
        success_threshold=args.success_threshold,
        max_steps_per_episode=args.max_steps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.save(args.output_dir / "eval_metrics.json")

    print("\n=== Summary ===")
    print(f"policy:        {metrics.policy_name}")
    print(f"episodes:      {metrics.n_episodes}")
    print(f"success rate:  {100*metrics.success_rate:.1f}% "
          f"[{100*metrics.success_ci_low:.1f}%, {100*metrics.success_ci_high:.1f}%]")
    print(f"avg max_reward: {metrics.avg_max_reward:.3f}")
    print(f"wall time:     {metrics.wall_time_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
