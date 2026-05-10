#!/usr/bin/env python
"""Evaluate a trained LeRobot policy (Diffusion or ACT) against gym-pusht.

Phase 5 deliverable for the Diffusion + ACT slots. Uses our own runner with
the LeRobotPolicyAdapter — this gives exact-comparability with the MLP-BC
eval (same env, seeds, episode count, success threshold), at the cost of
re-running rollouts that ``lerobot-eval`` could also produce. The trade-off
is worth it: the comparison report needs guarantees that all three policies
saw identical initial states.

Output: ``outputs/eval/<policy_name>/eval_metrics.json``.

Usage::

    # Diffusion (final checkpoint):
    python scripts/eval_lerobot.py checkpoints/diffusion_pusht/checkpoints/last \\
        --policy-name diffusion_pusht

    # ACT (final checkpoint):
    python scripts/eval_lerobot.py checkpoints/act_pusht/checkpoints/last \\
        --policy-name act_pusht

    # Smoke (3 episodes, no GPU contention with training):
    python scripts/eval_lerobot.py <ckpt> --n-episodes 3 --device cpu

Note: ``--device cpu`` makes Diffusion eval ~10× slower than GPU due to the
100-step denoising loop in ``select_action``. Use only when GPU is busy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lerobot_pusht_lab.eval.lerobot_adapter import LeRobotPolicyAdapter
from lerobot_pusht_lab.eval.runner import run_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, help="Checkpoint dir (containing pretrained_model/) or pretrained_model/ itself.")
    p.add_argument("--policy-name", help="Override policy name (default: inferred from checkpoint path).")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--base-seed", type=int, default=100000)
    p.add_argument("--success-threshold", type=float, default=0.95)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Default: outputs/eval/<policy_name>/",
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
    logger = logging.getLogger("eval_lerobot")

    if not args.checkpoint.exists():
        logger.error("checkpoint not found: %s", args.checkpoint)
        return 1

    logger.info("loading LeRobot policy from %s (device=%s)", args.checkpoint, args.device)
    adapter = LeRobotPolicyAdapter.from_checkpoint(
        args.checkpoint, name=args.policy_name, device=args.device,
    )

    metrics = run_evaluation(
        policy=adapter,
        n_episodes=args.n_episodes,
        base_seed=args.base_seed,
        success_threshold=args.success_threshold,
        max_steps_per_episode=args.max_steps,
    )
    output_dir = args.output_dir or Path("outputs/eval") / adapter.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.save(output_dir / "eval_metrics.json")

    print("\n=== Summary ===")
    print(f"policy:        {metrics.policy_name}")
    print(f"episodes:      {metrics.n_episodes}")
    print(f"success rate:  {100*metrics.success_rate:.1f}% "
          f"[{100*metrics.success_ci_low:.1f}%, {100*metrics.success_ci_high:.1f}%]")
    print(f"avg max_reward: {metrics.avg_max_reward:.3f}")
    print(f"avg ep length: {metrics.avg_episode_length:.1f} steps")
    print(f"avg inference: {metrics.avg_inference_time_s:.3f} s/episode")
    print(f"wall time:     {metrics.wall_time_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
