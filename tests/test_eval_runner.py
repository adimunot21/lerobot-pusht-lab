"""Unit tests for the eval runner — Wilson CI, episode loop, adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gym_pusht  # noqa: F401  - side-effect: registers gym envs
import gymnasium as gym
import numpy as np
import pytest

from lerobot_pusht_lab.eval.runner import (
    EpisodeMetrics,
    EvalMetrics,
    MLPBCAdapter,
    PolicyAdapter,
    _run_one_episode,
    run_evaluation,
    wilson_score_interval,
)


# ---------------------------------------------------------------------------
# wilson_score_interval
# ---------------------------------------------------------------------------


class TestWilsonScoreInterval:
    def test_zero_n_returns_zero_zero(self) -> None:
        assert wilson_score_interval(0, 0) == (0.0, 0.0)

    def test_zero_successes_lower_bound_is_zero(self) -> None:
        lo, hi = wilson_score_interval(0, 100)
        assert lo == 0.0
        assert 0 < hi < 0.05  # one-sided upper for 0/100 is small but not zero

    def test_all_successes_upper_bound_is_one(self) -> None:
        lo, hi = wilson_score_interval(100, 100)
        # Float arithmetic gives 0.999...9, not exactly 1.0
        assert hi == pytest.approx(1.0, abs=1e-10)
        assert 0.95 < lo < 1.0

    def test_half_successes(self) -> None:
        # 50/100 → CI should be roughly (0.40, 0.60), centred on 0.5
        lo, hi = wilson_score_interval(50, 100)
        assert 0.39 < lo < 0.41
        assert 0.59 < hi < 0.61

    def test_small_sample_wide_ci(self) -> None:
        # 1/3 → very wide CI (small sample). Should still be valid bounds.
        lo, hi = wilson_score_interval(1, 3)
        assert 0.0 <= lo < 0.333 < hi <= 1.0

    def test_bounds_are_clipped_to_unit_interval(self) -> None:
        for s, n in [(0, 1), (1, 1), (0, 5), (5, 5), (1, 50)]:
            lo, hi = wilson_score_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Stub policy for testing the eval loop without a real model
# ---------------------------------------------------------------------------


@dataclass
class StubPolicy:
    """Policy that always emits the same action — useful for testing the loop."""

    name: str = "stub"
    action: tuple[float, float] = (256.0, 256.0)
    reset_count: int = 0
    predict_count: int = 0

    def predict(self, observation: dict[str, Any]) -> np.ndarray:
        self.predict_count += 1
        return np.array(self.action, dtype=np.float32)

    def reset(self) -> None:
        self.reset_count += 1


# ---------------------------------------------------------------------------
# _run_one_episode (private but worth covering — it's the core loop)
# ---------------------------------------------------------------------------


class TestRunOneEpisode:
    @pytest.fixture
    def env(self) -> gym.Env:
        e = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            observation_width=96,
            observation_height=96,
            render_mode="rgb_array",
        )
        yield e
        e.close()

    def test_returns_episode_metrics(self, env: gym.Env) -> None:
        policy = StubPolicy()
        ep = _run_one_episode(policy, env, seed=42, episode_index=0,
                              success_threshold=0.95, max_steps=20)
        assert isinstance(ep, EpisodeMetrics)
        assert ep.episode_index == 0
        assert ep.seed == 42
        assert ep.n_steps <= 20  # capped
        # PushT episodes can terminate early; this stub policy is unlikely to succeed
        # at the task in 20 steps, but it might via randomness — don't assert success/fail.
        assert isinstance(ep.success, bool)
        assert ep.max_reward >= 0.0
        assert ep.inference_time_s >= 0.0

    def test_max_steps_cap_enforced(self, env: gym.Env) -> None:
        policy = StubPolicy()
        ep = _run_one_episode(policy, env, seed=1, episode_index=0,
                              success_threshold=0.95, max_steps=3)
        # Either we got 3 steps (cap hit) or fewer if env terminated. Stub
        # policy on first step shouldn't terminate.
        assert ep.n_steps <= 3

    def test_reset_called_once_per_episode(self, env: gym.Env) -> None:
        policy = StubPolicy()
        _run_one_episode(policy, env, seed=1, episode_index=0,
                         success_threshold=0.95, max_steps=5)
        assert policy.reset_count == 1
        assert policy.predict_count >= 1


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def test_aggregates_across_episodes(self) -> None:
        policy = StubPolicy()
        metrics = run_evaluation(
            policy,
            n_episodes=3,
            base_seed=1000,
            max_steps_per_episode=10,
            progress_log_interval=1,
        )
        assert isinstance(metrics, EvalMetrics)
        assert metrics.policy_name == "stub"
        assert metrics.n_episodes == 3
        assert len(metrics.episodes) == 3
        # Per-episode seeds should be deterministic from base_seed
        assert [e.seed for e in metrics.episodes] == [1000, 1001, 1002]

    def test_seed_determinism_across_runs(self) -> None:
        # Two evals with the same policy + base_seed should hit identical
        # initial states. With our deterministic stub, every per-episode result
        # should match.
        m1 = run_evaluation(StubPolicy(), n_episodes=2, base_seed=999, max_steps_per_episode=10)
        m2 = run_evaluation(StubPolicy(), n_episodes=2, base_seed=999, max_steps_per_episode=10)
        for e1, e2 in zip(m1.episodes, m2.episodes):
            assert e1.seed == e2.seed
            assert e1.n_steps == e2.n_steps
            # max_reward is determined by env physics + actions, both deterministic
            assert e1.max_reward == pytest.approx(e2.max_reward)


# ---------------------------------------------------------------------------
# MLPBCAdapter.from_checkpoint round-trip
# ---------------------------------------------------------------------------


class TestMLPBCAdapterCheckpoint:
    def test_load_round_trip(self, tmp_path) -> None:
        from pathlib import Path
        import json
        from dataclasses import asdict

        from lerobot_pusht_lab.policies.mlp_bc import MLPBC, MLPBCConfig

        # Create a checkpoint with the exact layout train_mlp_bc.py writes
        cfg = MLPBCConfig(input_dim=2, output_dim=2, hidden_dim=32, num_hidden_layers=1)
        model = MLPBC(cfg)
        import torch
        model.input_normalizer.fit(torch.tensor([[0.0, 0.0], [500.0, 500.0]]))
        model.output_normalizer.fit(torch.tensor([[0.0, 0.0], [500.0, 500.0]]))

        ckpt_dir = tmp_path / "checkpoints" / "001000"
        pretrained_dir = ckpt_dir / "pretrained_model"
        pretrained_dir.mkdir(parents=True)
        model.save(pretrained_dir / "model.safetensors")
        (pretrained_dir / "config.json").write_text(json.dumps(asdict(cfg)))

        adapter = MLPBCAdapter.from_checkpoint(ckpt_dir, device="cpu")
        # Adapter name comes from the parent.parent.name — the policy name
        assert "001000" not in adapter.name  # name is the *policy* name not the step
        # Predict on a fake observation
        obs = {"agent_pos": np.array([100.0, 100.0], dtype=np.float32),
               "pixels": np.zeros((96, 96, 3), dtype=np.uint8)}
        action = adapter.predict(obs)
        assert action.shape == (2,)
        assert isinstance(action, np.ndarray)

    def test_missing_checkpoint_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            MLPBCAdapter.from_checkpoint(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# EvalMetrics serialisation
# ---------------------------------------------------------------------------


class TestEvalMetricsSerialisation:
    def test_round_trip_via_save(self, tmp_path) -> None:
        import json
        ep = EpisodeMetrics(
            episode_index=0, seed=1, success=True, max_reward=0.97,
            sum_reward=10.0, n_steps=200, inference_time_s=0.5,
        )
        m = EvalMetrics(
            policy_name="test", n_episodes=1, success_rate=1.0,
            success_ci_low=0.5, success_ci_high=1.0,
            avg_max_reward=0.97, avg_sum_reward=10.0,
            avg_episode_length=200, avg_inference_time_s=0.5,
            success_threshold=0.95, wall_time_s=1.0,
            episodes=[ep],
        )
        path = m.save(tmp_path / "m.json")
        # Round-trip via load_eval_metrics should preserve everything
        from lerobot_pusht_lab.eval.compare import load_eval_metrics
        loaded = load_eval_metrics(path)
        assert loaded.policy_name == "test"
        assert loaded.n_episodes == 1
        assert loaded.episodes[0].success is True
        assert loaded.episodes[0].max_reward == pytest.approx(0.97)
