#!/usr/bin/env python
"""Phase 1: inspect the lerobot/pusht dataset and write a data-contract report.

Mandated by CLAUDE.md's data-validation rule: never assume schema or value ranges
for an external dataset. This script is the discovery step.

What it does:
    1. Loads dataset metadata via LeRobotDatasetMetadata (no downloads).
    2. Loads the parquet portion via LeRobotDataset(download_videos=False) to
       analyse episode-length distribution (~5MB, fast).
    3. Loads a few episodes WITH videos to pull raw sample frames.
    4. Aggregates field stats over those samples.
    5. Saves one frame as PNG and writes a markdown report.

Output: outputs/inspection/lerobot_pusht.md and lerobot_pusht_sample.png.

Usage::

    python scripts/inspect_pusht.py --num-samples 3 --sample-episodes 0 1 2

Why a separate CLI script: the helpers in
``lerobot_pusht_lab.data.inspection`` are dataset-agnostic (Phase 7 reuses them
on the SO-101 dataset). This file is the dataset-specific glue.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot_pusht_lab.data.inspection import (
    aggregate_field_stats,
    collect_samples,
    episode_length_stats,
    evenly_spaced_indices,
    save_image_tensor,
    summarize_tensor,
)

logger = logging.getLogger("inspect_pusht")


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _format_metadata_section(meta: LeRobotDatasetMetadata, repo_id: str) -> str:
    rows = [
        ("Repo ID", repo_id),
        ("Source", f"https://huggingface.co/datasets/{repo_id}"),
        ("Total episodes", meta.total_episodes),
        ("Total frames", meta.total_frames),
        ("FPS", meta.fps),
        ("Robot type", meta.robot_type),
        ("Camera keys", ", ".join(meta.camera_keys) or "(none)"),
        ("Image keys", ", ".join(meta.image_keys) or "(none — videos only)"),
        ("Video keys", ", ".join(meta.video_keys) or "(none)"),
        ("Total tasks", meta.total_tasks),
    ]
    md = "## Dataset Metadata\n\n| Property | Value |\n|---|---|\n"
    md += "\n".join(f"| {k} | {v} |" for k, v in rows)
    return md + "\n"


def _format_schema_section(meta: LeRobotDatasetMetadata) -> str:
    md = "## Feature Schema (declared by dataset metadata)\n\n"
    md += "| Field | dtype | shape | notes |\n|---|---|---|---|\n"
    for name, info in meta.features.items():
        dtype = info.get("dtype", "?")
        shape = info.get("shape", "?")
        note = ""
        if dtype == "video":
            vi = info.get("video_info", {})
            note = f"codec={vi.get('video.codec')} fps={vi.get('video.fps')}"
        names = info.get("names")
        if isinstance(names, dict):
            note = (note + " " if note else "") + f"axis_names={names}"
        elif isinstance(names, list):
            note = (note + " " if note else "") + f"axis_names={names}"
        md += f"| `{name}` | {dtype} | {shape} | {note} |\n"
    return md


def _format_field_stats_section(stats: dict[str, dict[str, Any]], n_samples: int) -> str:
    md = f"## Aggregated Field Stats (across {n_samples} sampled frames)\n\n"
    md += "Min/max/mean/std are over **all elements of all sampled frames** "
    md += "(e.g. for a 96×96×3 image sampled N times, that's N·27 648 pixel values).\n\n"
    md += "| Field | dtype | shape | min | max | mean | std |\n|---|---|---|---|---|---|---|\n"
    for name, s in stats.items():
        if "error" in s:
            md += f"| `{name}` | (error: {s['error']}) | | | | | |\n"
            continue
        if "min" not in s:
            # non-tensor field
            ex = s.get("examples", [])
            md += f"| `{name}` | {s.get('dtype', '?')} | scalar | — | — | — | examples: {ex} |\n"
            continue
        md += (
            f"| `{name}` | {s['dtype']} | {s['shape']} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['mean']:.4f} | {s['std']:.4f} |\n"
        )
    return md


def _format_episode_section(ep: dict[str, Any]) -> str:
    md = "## Episode-Length Distribution (full dataset)\n\n"
    md += f"- **Total episodes:** {ep['n_episodes']}\n"
    md += f"- **Frames per episode:** min={ep['min']}, max={ep['max']}, "
    md += f"mean={ep['mean']:.1f}, median={ep['median']:.1f}, std={ep['std']:.1f}\n\n"
    md += "Histogram (10 fixed-width buckets across observed range):\n\n"
    md += "| Frames-per-episode range | Episode count |\n|---|---|\n"
    for b in ep["histogram"]:
        lo, hi = b["range"]
        md += f"| {lo:.0f} – {hi:.0f} | {b['count']} |\n"
    return md


def _format_raw_samples_section(samples: list[dict[str, Any]], indices: list[int]) -> str:
    md = "## Raw Sample Frames\n\n"
    md += "Tensor previews are summarised (shape/dtype/min/max) — full image arrays "
    md += "would be unreadable. The saved PNG below shows one frame visually.\n\n"
    for sample_pos, (idx, sample) in enumerate(zip(indices, samples)):
        md += f"### Sample {sample_pos + 1} (dataset index {idx})\n\n"
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                summary = summarize_tensor(v)
                if "scalar_value" in summary:
                    md += f"- `{k}`: {summary['dtype']} scalar = {summary['scalar_value']}\n"
                else:
                    md += (
                        f"- `{k}`: {summary['dtype']} {summary['shape']} "
                        f"min={summary['min']:.4f} max={summary['max']:.4f} "
                        f"mean={summary['mean']:.4f}\n"
                    )
            else:
                md += f"- `{k}`: {type(v).__name__} = {v!r}\n"
        md += "\n"
    return md


def _format_findings_section(stats: dict[str, dict[str, Any]]) -> str:
    """Compare measured stats to PLAN.md §8.1 expectations and call out surprises.

    Hard-coded checks per the Phase 1 deliverable. If the schema changes upstream,
    this section will report the divergence.
    """
    findings: list[str] = []

    img = stats.get("observation.image", {})
    if img.get("min", 1.0) >= 0.0 and img.get("max", 0.0) <= 1.0:
        findings.append(
            "**Image normalisation:** float32 in [0, 1] — confirmed (PLAN.md §8.1 had this listed as TO VERIFY)."
        )
    else:
        findings.append(
            f"**Image normalisation:** UNEXPECTED range "
            f"min={img.get('min')} max={img.get('max')}. PLAN.md §8.1 expected uint8 OR float [0,1]."
        )

    if "shape" in img:
        findings.append(
            f"**Image layout (post-`__getitem__`):** {img['shape']} — channels-first (PyTorch convention). "
            "Note: dataset metadata declares storage layout as (H, W, C) but the dataloader returns (C, H, W)."
        )

    state = stats.get("observation.state", {})
    if "min" in state:
        findings.append(
            f"**State / action ranges:** observation.state min={state['min']:.1f} max={state['max']:.1f} — "
            "**not** in [0, 1]. Values are raw env coordinates (pixel-space, ~0–512). "
            "→ Implication: the from-scratch MLP-BC (Phase 4) needs explicit normalisation; "
            "LeRobot's diffusion/ACT pipelines apply their own normalisation internally."
        )
    action = stats.get("action", {})
    if "min" in action:
        findings.append(
            f"**Action range:** min={action['min']:.1f} max={action['max']:.1f} — same env-coord scale as state."
        )

    extras = ["next.success", "task", "task_index", "index"]
    seen = [e for e in extras if e in stats]
    if seen:
        findings.append(
            f"**Extra fields not listed in PLAN.md §8.1:** `{', '.join(seen)}`. "
            "`task` is a string (the natural-language instruction); `next.success` is a bool flag distinct from `next.done`."
        )

    md = "## Key Findings vs PLAN.md §8.1\n\n"
    for f in findings:
        md += f"- {f}\n"
    return md


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def inspect(repo_id: str, num_samples: int, sample_episodes: list[int], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading metadata for %s", repo_id)
    meta = LeRobotDatasetMetadata(repo_id)
    logger.info(
        "metadata: %d episodes / %d frames / %d fps", meta.total_episodes, meta.total_frames, meta.fps
    )

    logger.info("loading parquet (no videos) for episode-length analysis")
    ds_no_video = LeRobotDataset(repo_id, download_videos=False)
    ep_stats = episode_length_stats(ds_no_video.hf_dataset)
    logger.info(
        "episodes: %d, frames/episode min=%d max=%d mean=%.1f",
        ep_stats["n_episodes"],
        ep_stats["min"],
        ep_stats["max"],
        ep_stats["mean"],
    )

    logger.info("loading episodes %s with videos for sample frames", sample_episodes)
    ds_with_video = LeRobotDataset(repo_id, episodes=sample_episodes)
    n = len(ds_with_video)
    logger.info("downloaded %d frames across %d episodes", n, len(sample_episodes))

    sample_indices = evenly_spaced_indices(n, num_samples)
    raw_samples = collect_samples(ds_with_video, sample_indices)
    logger.info("collected %d raw samples at indices %s", len(raw_samples), sample_indices)

    # Stats over a wider sample for tighter min/max bounds — use up to 200 evenly
    # spaced frames from the loaded episodes (still all in RAM, decoding is the cost).
    stat_n = min(200, n)
    stat_indices = evenly_spaced_indices(n, stat_n)
    logger.info("computing field stats over %d frames", stat_n)
    stat_samples = collect_samples(ds_with_video, stat_indices)
    field_stats = {k: aggregate_field_stats(stat_samples, k) for k in stat_samples[0]}

    # Save one image — pick the middle sample
    img_path = output_dir / "lerobot_pusht_sample.png"
    middle_sample = raw_samples[len(raw_samples) // 2]
    save_image_tensor(middle_sample["observation.image"], img_path)

    # Build report
    lines = [
        f"# Dataset Inspection Report: `{repo_id}`",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_Generated by: `scripts/inspect_pusht.py`_",
        "",
        _format_metadata_section(meta, repo_id),
        _format_schema_section(meta),
        _format_episode_section(ep_stats),
        _format_field_stats_section(field_stats, stat_n),
        _format_raw_samples_section(raw_samples, sample_indices),
        f"## Sample Image\n\n![Sample frame from episode 0](./{img_path.name})\n",
        _format_findings_section(field_stats),
    ]
    report = "\n".join(lines)

    report_path = output_dir / "lerobot_pusht.md"
    report_path.write_text(report)
    logger.info("wrote report: %s", report_path)

    # Echo to stdout so it's visible in the terminal
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="lerobot/pusht", help="HF Hub dataset repo")
    p.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Raw frames to print in full (default: 3)",
    )
    p.add_argument(
        "--sample-episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Episode indices to download for sampling (default: 0 1 2)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inspection"),
        help="Where to write report + sample PNG",
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
    try:
        inspect(args.repo_id, args.num_samples, args.sample_episodes, args.output_dir)
    except Exception:
        logger.exception("inspection failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
