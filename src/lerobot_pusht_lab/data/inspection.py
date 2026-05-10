"""Reusable dataset-inspection helpers.

These functions are dataset-agnostic. They take tensors and HuggingFace Datasets
and return summary statistics; dataset-specific orchestration lives in
``scripts/inspect_*.py``.

Used by:
    - scripts/inspect_pusht.py (Phase 1)
    - scripts/inspect_so101.py (Phase 7)

The CLAUDE.md "data validation" rule mandates that every external dataset is
inspected before processing code is written against it. These helpers exist so
that step is one short script, not a 100-line ad-hoc notebook each time.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torchvision.utils as tvu
from datasets import Dataset

logger = logging.getLogger(__name__)


def summarize_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Return dtype, shape, min, max, mean, std for a tensor.

    Handles float / int / bool. For 0-dim tensors, also returns the scalar value
    so the caller can show "this frame's reward = 0.19" verbatim.

    Equivalent manual implementation: just call .min(), .max(), .mean(), .std()
    yourself — but bool tensors don't support .mean() directly, hence the cast.
    """
    out: dict[str, Any] = {
        "dtype": str(t.dtype),
        "shape": tuple(t.shape),
    }
    if t.numel() == 0:
        return out

    out["min"] = float(t.min().item())
    out["max"] = float(t.max().item())

    floatv = t.float() if not t.dtype.is_floating_point else t
    out["mean"] = float(floatv.mean().item())
    out["std"] = float(floatv.std().item()) if t.numel() > 1 else 0.0

    if t.dim() == 0:
        out["scalar_value"] = bool(t.item()) if t.dtype == torch.bool else float(t.item())
    return out


def aggregate_field_stats(samples: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """Aggregate stats for one field across N sample dicts (e.g. dataset[i] outputs).

    Returns the union of dtype/shape (assumed consistent) plus min/max/mean/std
    computed across *all elements of all samples* — so for a 96x96 image sampled
    400 times, the stats are over ~3.7M pixels.

    Why aggregate across samples rather than per-sample: value-range claims like
    "image is in [0, 1]" are only meaningful at the population level, not from one
    frame.
    """
    values = [s[field] for s in samples if field in s]
    if not values:
        return {"error": f"field {field!r} missing from all samples"}

    first = values[0]
    if not isinstance(first, torch.Tensor):
        return {
            "dtype": type(first).__name__,
            "n_samples": len(values),
            "examples": values[:3],
        }

    # Promote 0-dim → 1-dim so torch.stack works uniformly
    stacked = torch.stack([v if v.dim() > 0 else v.unsqueeze(0) for v in values])
    floatv = stacked.float() if not stacked.dtype.is_floating_point else stacked
    return {
        "dtype": str(first.dtype),
        "shape": tuple(first.shape),
        "n_samples": len(values),
        "min": float(stacked.min().item()),
        "max": float(stacked.max().item()),
        "mean": float(floatv.mean().item()),
        "std": float(floatv.std().item()),
    }


def episode_length_stats(hf_dataset: Dataset, episode_field: str = "episode_index") -> dict[str, Any]:
    """Compute frames-per-episode distribution from the parquet-backed HF Dataset.

    Equivalent manual: ``df.groupby('episode_index').size()`` if you converted to
    pandas. Using torch.unique avoids the pandas detour.

    The histogram has 10 fixed-width buckets across the observed range — fine for
    the typical "are episodes roughly the same length?" question.
    """
    if episode_field not in hf_dataset.column_names:
        raise ValueError(
            f"column {episode_field!r} not found; available: {hf_dataset.column_names}"
        )

    ep_idx = torch.tensor(hf_dataset[episode_field], dtype=torch.long)
    _unique, counts = torch.unique(ep_idx, return_counts=True)
    counts_f = counts.float()

    n_buckets = 10
    lo, hi = counts.min().item(), counts.max().item()
    if lo == hi:
        # All episodes same length — degenerate histogram
        histogram = [{"range": (float(lo), float(hi)), "count": int(counts.numel())}]
    else:
        hist = torch.histc(counts_f, bins=n_buckets, min=float(lo), max=float(hi))
        edges = torch.linspace(float(lo), float(hi), n_buckets + 1)
        histogram = [
            {
                "range": (float(edges[i].item()), float(edges[i + 1].item())),
                "count": int(hist[i].item()),
            }
            for i in range(n_buckets)
        ]

    return {
        "n_episodes": int(counts.numel()),
        "min": int(lo),
        "max": int(hi),
        "mean": float(counts_f.mean().item()),
        "median": float(counts_f.median().item()),
        "std": float(counts_f.std().item()) if counts.numel() > 1 else 0.0,
        "histogram": histogram,
    }


def save_image_tensor(t: torch.Tensor, path: str | Path) -> Path:
    """Save a 3-D image tensor as PNG.

    Accepts CHW or HWC, float [0,1] or uint8 [0,255]. Detects layout from the
    smaller-of-shape-{0,-1} being in {1,3,4}.

    Equivalent manual: PIL.Image.fromarray(t.permute(1,2,0).mul(255).byte().numpy()).
    Using torchvision.utils.save_image because it already handles the dtype
    branches correctly. Alternative: imageio.imwrite — fine, one less dep here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if t.dim() != 3:
        raise ValueError(f"expected 3-D tensor, got shape {tuple(t.shape)}")

    # Detect layout: a "channel" axis has 1, 3, or 4 entries
    if t.shape[0] in (1, 3, 4):
        chw = t
    elif t.shape[-1] in (1, 3, 4):
        chw = t.permute(2, 0, 1)
    else:
        raise ValueError(f"cannot infer CHW vs HWC from shape {tuple(t.shape)}")

    chw_f = chw.float()
    if chw_f.max() > 1.5:  # heuristic: looks like 0-255
        chw_f = chw_f / 255.0
    tvu.save_image(chw_f, str(path))
    logger.info("saved image: %s", path)
    return path


def collect_samples(
    dataset: Any,
    indices: Iterable[int],
) -> list[dict[str, Any]]:
    """Pull samples at the given indices into a list of dicts.

    Kept as a tiny helper so the orchestrator can be ``samples =
    collect_samples(ds, evenly_spaced(len(ds), n))`` rather than a loop.
    """
    return [dataset[i] for i in indices]


def evenly_spaced_indices(n_total: int, n_samples: int) -> list[int]:
    """Return n_samples integer indices spanning [0, n_total)."""
    if n_samples <= 0:
        return []
    if n_samples == 1:
        # Only return a single point — pick the first frame. (Returning the
        # midpoint is also defensible; first is more intuitive for "show one sample".)
        return [0]
    if n_samples >= n_total:
        return list(range(n_total))
    return [int(i * (n_total - 1) / (n_samples - 1)) for i in range(n_samples)]


if __name__ == "__main__":
    # Smoke test — runnable standalone per CLAUDE.md.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print("summarize_tensor(1d):", summarize_tensor(t))

    img = torch.rand(3, 96, 96)
    print("summarize_tensor(image):", summarize_tensor(img))

    samples = [{"x": torch.rand(2)} for _ in range(5)]
    print("aggregate_field_stats:", aggregate_field_stats(samples, "x"))

    print("evenly_spaced_indices(100, 5):", evenly_spaced_indices(100, 5))
    print("smoke test OK")
