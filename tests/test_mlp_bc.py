"""Unit tests for the MLP-BC policy module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lerobot_pusht_lab.policies.mlp_bc import (
    MLPBC,
    MLPBCConfig,
    StateActionNormalizer,
    _activation_module,
)


# ---------------------------------------------------------------------------
# StateActionNormalizer
# ---------------------------------------------------------------------------


class TestStateActionNormalizer:
    def test_round_trip(self) -> None:
        norm = StateActionNormalizer(dim=2)
        samples = torch.tensor([[10.0, 200.0], [50.0, 250.0], [100.0, 300.0]])
        norm.fit(samples)
        normalised = norm.normalize(samples)
        assert torch.allclose(normalised.min(0).values, torch.tensor([-1.0, -1.0]), atol=1e-5)
        assert torch.allclose(normalised.max(0).values, torch.tensor([1.0, 1.0]), atol=1e-5)
        denorm = norm.denormalize(normalised)
        assert torch.allclose(denorm, samples, atol=1e-5)

    def test_constant_feature_no_divide_by_zero(self) -> None:
        norm = StateActionNormalizer(dim=2)
        samples = torch.tensor([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])  # x is constant
        norm.fit(samples)  # shouldn't raise
        out = norm.normalize(samples)
        assert torch.isfinite(out).all()

    def test_unfit_normaliser_is_identity_to_minus1_1(self) -> None:
        # Initial state: min=-1, max=1 → normalize is identity, denormalize is identity
        norm = StateActionNormalizer(dim=2)
        x = torch.tensor([[0.5, -0.3]])
        assert torch.allclose(norm.normalize(x), x)
        assert torch.allclose(norm.denormalize(x), x)

    def test_wrong_input_shape_raises(self) -> None:
        norm = StateActionNormalizer(dim=2)
        with pytest.raises(ValueError, match="expected"):
            norm.fit(torch.tensor([1.0, 2.0, 3.0]))  # 1-D not 2-D
        with pytest.raises(ValueError, match="expected"):
            norm.fit(torch.tensor([[1.0, 2.0, 3.0]]))  # wrong dim

    def test_buffers_move_with_to(self) -> None:
        # The normaliser uses register_buffer, so .to() should move them
        norm = StateActionNormalizer(dim=2)
        norm.fit(torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
        # CPU → CPU is a no-op but shouldn't crash
        norm_cpu = norm.to("cpu")
        assert norm_cpu.min.device.type == "cpu"


# ---------------------------------------------------------------------------
# MLPBC model
# ---------------------------------------------------------------------------


class TestMLPBC:
    @pytest.fixture
    def model(self) -> MLPBC:
        cfg = MLPBCConfig(input_dim=2, output_dim=2, hidden_dim=64, num_hidden_layers=2)
        m = MLPBC(cfg)
        m.input_normalizer.fit(torch.tensor([[0.0, 0.0], [500.0, 500.0]]))
        m.output_normalizer.fit(torch.tensor([[0.0, 0.0], [500.0, 500.0]]))
        return m

    def test_forward_shape(self, model: MLPBC) -> None:
        x = torch.rand(8, 2) * 500
        out = model(x)
        assert out.shape == (8, 2)

    def test_predict_action_returns_env_coords(self, model: MLPBC) -> None:
        # After predict_action, output should be in [0, 500]-ish (env coords),
        # not [-1, 1].
        x = torch.rand(8, 2) * 500
        with torch.no_grad():
            out = model.predict_action(x)
        # Output range depends on what the network produces in [-1, 1] space, then
        # denormalised to [output_normalizer.min, output_normalizer.max] = [0, 500].
        # An untrained network typically outputs something near 0 in normalised space,
        # which denormalises to ~250 (the midpoint).
        assert (out >= 0).all() and (out <= 500).all()

    def test_loss_is_scalar(self, model: MLPBC) -> None:
        x = torch.rand(8, 2) * 500
        y = torch.rand(8, 2) * 500
        loss = model.loss(x, y)
        assert loss.shape == ()
        assert loss.item() > 0  # MSE is non-negative

    def test_gradient_flow(self, model: MLPBC) -> None:
        x = torch.rand(8, 2) * 500
        y = torch.rand(8, 2) * 500
        loss = model.loss(x, y)
        loss.backward()
        # Every learnable parameter should have a gradient
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"no grad for {name}"
            assert p.grad.abs().sum().item() > 0, f"zero grad for {name}"

    def test_deterministic_with_seed(self) -> None:
        torch.manual_seed(42)
        cfg = MLPBCConfig(input_dim=2, output_dim=2, dropout=0.0)  # disable dropout for determinism
        m1 = MLPBC(cfg)
        torch.manual_seed(42)
        m2 = MLPBC(cfg)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_save_load_round_trip(self, model: MLPBC, tmp_path: Path) -> None:
        x = torch.rand(8, 2) * 500
        model.eval()
        with torch.no_grad():
            out_orig = model(x)
        model.save(tmp_path / "model.safetensors")
        restored = MLPBC.load(tmp_path / "model.safetensors", model.config)
        restored.eval()
        with torch.no_grad():
            out_restored = restored(x)
        assert torch.allclose(out_orig, out_restored, atol=1e-6)

    def test_normalisers_persist_through_save_load(self, model: MLPBC, tmp_path: Path) -> None:
        # Verify the buffers (normaliser min/max) are restored, not just weights
        original_min = model.input_normalizer.min.clone()
        original_max = model.input_normalizer.max.clone()
        model.save(tmp_path / "m.safetensors")
        restored = MLPBC.load(tmp_path / "m.safetensors", model.config)
        assert torch.allclose(restored.input_normalizer.min, original_min)
        assert torch.allclose(restored.input_normalizer.max, original_max)


class TestActivationModule:
    @pytest.mark.parametrize("name", ["relu", "gelu", "tanh"])
    def test_supported(self, name: str) -> None:
        mod = _activation_module(name)
        # Should produce valid output for a sample input
        out = mod(torch.tensor([0.5]))
        assert out.shape == (1,)

    def test_case_insensitive(self) -> None:
        # _activation_module lowercases — "ReLU" should work
        mod = _activation_module("ReLU")
        assert isinstance(mod, torch.nn.ReLU)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            _activation_module("swish")
