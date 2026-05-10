"""Unit tests for the inspection helpers (Phase 1's reusable building blocks)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from lerobot_pusht_lab.data.inspection import (
    aggregate_field_stats,
    episode_length_stats,
    evenly_spaced_indices,
    save_image_tensor,
    summarize_tensor,
)


# ---------------------------------------------------------------------------
# summarize_tensor
# ---------------------------------------------------------------------------


class TestSummarizeTensor:
    def test_float_1d(self) -> None:
        out = summarize_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert out["dtype"] == "torch.float32"
        assert out["shape"] == (4,)
        assert out["min"] == 1.0
        assert out["max"] == 4.0
        assert out["mean"] == pytest.approx(2.5)
        assert out["std"] == pytest.approx(1.2909944)

    def test_int_tensor_uses_float_for_mean_std(self) -> None:
        out = summarize_tensor(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
        assert out["dtype"] == "torch.int64"
        assert out["mean"] == pytest.approx(2.5)
        assert out["std"] == pytest.approx(1.2909944)

    def test_bool_tensor(self) -> None:
        out = summarize_tensor(torch.tensor([True, False, True, True]))
        assert out["dtype"] == "torch.bool"
        assert out["min"] == 0.0
        assert out["max"] == 1.0
        assert out["mean"] == pytest.approx(0.75)

    def test_zero_dim_includes_scalar_value(self) -> None:
        out = summarize_tensor(torch.tensor(3.14))
        assert out["shape"] == ()
        assert out["scalar_value"] == pytest.approx(3.14)

    def test_zero_dim_bool_returns_python_bool(self) -> None:
        out = summarize_tensor(torch.tensor(True))
        assert out["scalar_value"] is True
        assert isinstance(out["scalar_value"], bool)

    def test_empty_tensor(self) -> None:
        out = summarize_tensor(torch.empty(0))
        # min/max are absent — caller must handle this case
        assert out["shape"] == (0,)
        assert "min" not in out

    def test_single_element_zero_std(self) -> None:
        # std of one element is 0 (we return 0.0, not NaN — defensive default)
        out = summarize_tensor(torch.tensor([5.0]))
        assert out["std"] == 0.0


# ---------------------------------------------------------------------------
# aggregate_field_stats
# ---------------------------------------------------------------------------


class TestAggregateFieldStats:
    def test_aggregates_across_samples(self) -> None:
        samples = [{"x": torch.tensor([float(i), float(i + 1)])} for i in range(5)]
        out = aggregate_field_stats(samples, "x")
        assert out["n_samples"] == 5
        assert out["min"] == 0.0
        assert out["max"] == 5.0
        assert out["shape"] == (2,)

    def test_handles_zero_dim_via_unsqueeze(self) -> None:
        samples = [{"x": torch.tensor(float(i))} for i in range(3)]
        out = aggregate_field_stats(samples, "x")
        assert out["min"] == 0.0
        assert out["max"] == 2.0

    def test_missing_field_returns_error(self) -> None:
        out = aggregate_field_stats([{"y": torch.tensor(1.0)}], "x")
        assert "error" in out

    def test_non_tensor_field(self) -> None:
        samples = [{"label": "push"}, {"label": "pull"}, {"label": "push"}]
        out = aggregate_field_stats(samples, "label")
        assert out["dtype"] == "str"
        assert out["n_samples"] == 3
        assert "min" not in out


# ---------------------------------------------------------------------------
# episode_length_stats
# ---------------------------------------------------------------------------


class TestEpisodeLengthStats:
    def test_basic_distribution(self) -> None:
        # 3 episodes of lengths 100, 50, 200
        ep_idx = [0] * 100 + [1] * 50 + [2] * 200
        ds = Dataset.from_dict({"episode_index": ep_idx, "value": list(range(len(ep_idx)))})
        out = episode_length_stats(ds)
        assert out["n_episodes"] == 3
        assert out["min"] == 50
        assert out["max"] == 200
        assert out["mean"] == pytest.approx(116.6666, abs=1e-3)

    def test_constant_length_degenerate_histogram(self) -> None:
        ep_idx = [0] * 100 + [1] * 100  # both length 100
        ds = Dataset.from_dict({"episode_index": ep_idx})
        out = episode_length_stats(ds)
        assert out["n_episodes"] == 2
        assert out["min"] == 100 == out["max"]
        # Single histogram bucket — no division-by-zero
        assert len(out["histogram"]) == 1

    def test_missing_field_raises(self) -> None:
        ds = Dataset.from_dict({"foo": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            episode_length_stats(ds, episode_field="bar")


# ---------------------------------------------------------------------------
# evenly_spaced_indices
# ---------------------------------------------------------------------------


class TestEvenlySpacedIndices:
    def test_endpoints_included(self) -> None:
        idx = evenly_spaced_indices(100, 5)
        assert idx[0] == 0
        assert idx[-1] == 99
        assert len(idx) == 5

    def test_zero_samples(self) -> None:
        assert evenly_spaced_indices(100, 0) == []

    def test_more_samples_than_total(self) -> None:
        # Caller asks for more samples than exist → return everything once
        idx = evenly_spaced_indices(3, 10)
        assert idx == [0, 1, 2]

    def test_one_sample(self) -> None:
        # Single sample of n_total=100 returns just index 0 (degenerate)
        # because the formula is i*(n_total-1)/(n_samples-1) and i=0
        idx = evenly_spaced_indices(100, 1)
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# save_image_tensor
# ---------------------------------------------------------------------------


class TestSaveImageTensor:
    def test_chw_float_saves_png(self, tmp_path: Path) -> None:
        t = torch.rand(3, 16, 16)
        out = save_image_tensor(t, tmp_path / "img.png")
        assert out.exists()
        # PNG magic bytes
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_hwc_layout_detected_and_converted(self, tmp_path: Path) -> None:
        t = torch.rand(16, 16, 3)
        out = save_image_tensor(t, tmp_path / "img.png")
        assert out.exists()

    def test_uint8_in_0_255_normalised(self, tmp_path: Path) -> None:
        t = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
        out = save_image_tensor(t, tmp_path / "img.png")
        assert out.exists()

    def test_invalid_dim_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="3-D"):
            save_image_tensor(torch.rand(16, 16), tmp_path / "img.png")

    def test_ambiguous_layout_raises(self, tmp_path: Path) -> None:
        # Shape (2, 5, 7) has no axis in {1, 3, 4} so layout can't be inferred
        with pytest.raises(ValueError, match="cannot infer"):
            save_image_tensor(torch.rand(2, 5, 7), tmp_path / "img.png")
