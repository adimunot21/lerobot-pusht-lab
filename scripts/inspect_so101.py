#!/usr/bin/env python
"""Phase 7: inspect a community SO-101 dataset.

Why now: when the SO-101 arm arrives, the day-one workflow is "record dataset
→ run train". Doing a dry run on a community SO-101 dataset NOW means the
inspection step is debugged on real SO-101 schema before there's any pressure
from "the new hardware". Also surfaces any LeRobotDataset-version quirks we
haven't seen on PushT (multi-camera, joint-space state/action, unusual fps).

Default dataset: ``lerobot/svla_so101_pickplace`` (50 episodes, 30 fps,
official lerobot org, used in the SmolVLA examples). Override with --repo-id
to inspect a different community dataset.

Output: ``outputs/inspection/<dataset_slug>.md`` plus one sample frame per
camera at ``<dataset_slug>_<camera>_sample.png``.

Usage::

    # Default — official SmolVLA pickplace dataset
    python scripts/inspect_so101.py

    # Alternative community dataset
    python scripts/inspect_so101.py --repo-id ud-smart-city/lerobot-so-101-manipulations

The implementation reuses the Phase 1 helpers in
``lerobot_pusht_lab.data.inspection`` deliberately — proves the abstraction is
right by being usable on a totally different dataset/embodiment.
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

logger = logging.getLogger("inspect_so101")


def _format_metadata(meta: LeRobotDatasetMetadata, repo_id: str) -> str:
    rows = [
        ("Repo ID", repo_id),
        ("Source", f"https://huggingface.co/datasets/{repo_id}"),
        ("Total episodes", meta.total_episodes),
        ("Total frames", meta.total_frames),
        ("FPS", meta.fps),
        ("Robot type", meta.robot_type),
        ("Camera keys", ", ".join(meta.camera_keys) or "(none)"),
        ("Image keys", ", ".join(meta.image_keys) or "(none)"),
        ("Video keys", ", ".join(meta.video_keys) or "(none)"),
        ("Total tasks", meta.total_tasks),
    ]
    md = "## Dataset Metadata\n\n| Property | Value |\n|---|---|\n"
    md += "\n".join(f"| {k} | {v} |" for k, v in rows)
    return md + "\n"


def _format_schema(meta: LeRobotDatasetMetadata) -> str:
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
        if names:
            note = (note + " " if note else "") + f"axis_names={names}"
        md += f"| `{name}` | {dtype} | {shape} | {note} |\n"
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


def _format_field_stats(stats: dict[str, dict[str, Any]], n_samples: int) -> str:
    md = f"## Aggregated Field Stats (across {n_samples} sampled frames)\n\n"
    md += "| Field | dtype | shape | min | max | mean | std |\n|---|---|---|---|---|---|---|\n"
    for name, s in stats.items():
        if "error" in s:
            md += f"| `{name}` | (error) | | | | | |\n"
            continue
        if "min" not in s:
            md += f"| `{name}` | {s.get('dtype', '?')} | scalar | — | — | — | — |\n"
            continue
        md += (
            f"| `{name}` | {s['dtype']} | {s['shape']} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['mean']:.4f} | {s['std']:.4f} |\n"
        )
    return md


def _format_raw_samples(samples: list[dict[str, Any]], indices: list[int]) -> str:
    md = "## Raw Sample Frames\n\n"
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


def _format_findings(meta: LeRobotDatasetMetadata, stats: dict[str, dict[str, Any]]) -> str:
    """Phase 7 findings — cross-check vs PushT to highlight schema differences."""
    findings: list[str] = []

    # Multi-camera or single?
    n_cameras = len(meta.camera_keys)
    if n_cameras > 1:
        findings.append(
            f"**Multi-camera dataset:** {n_cameras} cameras "
            f"(`{', '.join(meta.camera_keys)}`). PushT had 1; SO-101 datasets "
            "typically have 2-3. Implication for image-based policies: vision "
            "encoder must handle multiple camera streams (LeRobot supports this "
            "via `use_separate_rgb_encoder_per_camera`)."
        )
    else:
        findings.append(
            f"**Single-camera dataset:** 1 camera (`{meta.camera_keys[0] if meta.camera_keys else '?'}`). "
            "Same shape as PushT in this respect."
        )

    # Action / state dimensionality — SO-101 is a 6-joint arm, so expect 6-dim
    state = stats.get("observation.state", {})
    action = stats.get("action", {})
    if "shape" in state:
        dim = state["shape"][0] if state["shape"] else 0
        findings.append(
            f"**State dimensionality:** {dim}D "
            f"(range {state.get('min', '?'):.2f} to {state.get('max', '?'):.2f}). "
            "SO-101 has 6 joints; if dim ≠ 6, the dataset includes extra channels "
            "(gripper open/close, end-effector pose, etc.)."
        )
    if "shape" in action:
        dim = action["shape"][0] if action["shape"] else 0
        findings.append(
            f"**Action dimensionality:** {dim}D "
            f"(range {action.get('min', '?'):.2f} to {action.get('max', '?'):.2f})."
        )

    # FPS check — SO-101 datasets are usually 30 fps vs PushT's 10
    findings.append(
        f"**Sampling rate:** {meta.fps} fps. "
        f"{'Higher than PushT (10 fps) — implies smoother trajectories and longer episodes for the same task duration.' if meta.fps > 15 else 'Comparable to PushT.'}"
    )

    md = "## Key Findings (vs PushT)\n\n"
    for f in findings:
        md += f"- {f}\n"
    return md


def inspect(repo_id: str, num_samples: int, sample_episodes: list[int], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = repo_id.replace("/", "_")

    logger.info("loading metadata for %s", repo_id)
    meta = LeRobotDatasetMetadata(repo_id)
    logger.info(
        "metadata: %d episodes / %d frames / %d fps / cameras=%s",
        meta.total_episodes, meta.total_frames, meta.fps, meta.camera_keys,
    )

    logger.info("loading parquet (no videos) for episode-length analysis")
    ds_no_video = LeRobotDataset(repo_id, download_videos=False)
    ep_stats = episode_length_stats(ds_no_video.hf_dataset)
    logger.info("episodes: %d, frames/episode min=%d max=%d mean=%.1f",
                ep_stats["n_episodes"], ep_stats["min"], ep_stats["max"], ep_stats["mean"])

    logger.info("loading episodes %s with videos for sample frames", sample_episodes)
    ds_with_video = LeRobotDataset(repo_id, episodes=sample_episodes)
    n = len(ds_with_video)
    logger.info("downloaded %d frames across %d episodes", n, len(sample_episodes))

    sample_indices = evenly_spaced_indices(n, num_samples)
    raw_samples = collect_samples(ds_with_video, sample_indices)

    stat_n = min(200, n)
    stat_indices = evenly_spaced_indices(n, stat_n)
    stat_samples = collect_samples(ds_with_video, stat_indices)
    field_stats = {k: aggregate_field_stats(stat_samples, k) for k in stat_samples[0]}

    # Save one image PER camera key (vs PushT which had only 1)
    middle_sample = raw_samples[len(raw_samples) // 2]
    image_paths: dict[str, Path] = {}
    for cam in meta.camera_keys:
        if cam not in middle_sample:
            logger.warning("camera %s not present in sample — skipping image save", cam)
            continue
        path = output_dir / f"{slug}_{cam.replace('.', '_')}_sample.png"
        save_image_tensor(middle_sample[cam], path)
        image_paths[cam] = path

    lines = [
        f"# Dataset Inspection Report: `{repo_id}`",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        f"_Generated by: `scripts/inspect_so101.py`_",
        "",
        _format_metadata(meta, repo_id),
        _format_schema(meta),
        _format_episode_section(ep_stats),
        _format_field_stats(field_stats, stat_n),
        _format_raw_samples(raw_samples, sample_indices),
    ]
    if image_paths:
        lines.append("## Sample Images\n")
        for cam, path in image_paths.items():
            lines.append(f"### `{cam}`\n\n![{cam}](./{path.name})\n")
    lines.append(_format_findings(meta, field_stats))

    report = "\n".join(lines)
    report_path = output_dir / f"{slug}.md"
    report_path.write_text(report)
    logger.info("wrote report: %s", report_path)
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default="lerobot/svla_so101_pickplace",
                   help="HF dataset repo. Default is the official SmolVLA pickplace dataset.")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--sample-episodes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/inspection"))
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
