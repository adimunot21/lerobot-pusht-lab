"""Phase 5 eval harness: run any policy through gym-pusht, compute metrics.

Goal: produce a uniform evaluation across all 3 policies (Diffusion, ACT, MLP-BC)
so the comparison report in Phase 5 is honest. "Uniform" means:
  - Same gym-pusht env config (96×96 obs, episode_length=300)
  - Same episode count + same seeds (so episode N for Diffusion is the same
    initial state as episode N for ACT and MLP-BC)
  - Same success criterion (max overlap ≥ 95% per LeRobot's PushT convention)
  - Same metric definitions (success_rate, avg_max_reward, avg_sum_reward)

Architecture: the env loop and the metrics computation are policy-agnostic. The
only per-policy code is the ``PolicyAdapter`` that translates between the env's
observation dict and the policy's expected input format.

What's here:
  - ``PolicyAdapter`` — protocol describing the minimum interface
  - ``MLPBCAdapter`` — wraps our from-scratch MLP-BC
  - ``run_evaluation`` — the loop. Returns ``EvalMetrics``.
  - ``EvalMetrics`` / ``EpisodeMetrics`` — typed dataclasses
  - ``wilson_score_interval`` — 95% CI for binomial success rate

LeRobot adapters (DiffusionPolicy, ACTPolicy) are deferred to a separate file
once those checkpoints exist — keeps this module testable without GPU.

What's NOT here:
  - The cross-policy comparison report (markdown + plots): ``compare.py``.
  - Top-level CLI (``scripts/eval_all.py``): TBD.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gym_pusht  # noqa: F401  - import registers gym envs
import gymnasium as gym
import numpy as np
import torch

from lerobot_pusht_lab.policies.mlp_bc import MLPBC, MLPBCConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class EpisodeMetrics:
    """Per-episode results."""

    episode_index: int
    seed: int
    success: bool                    # max_reward ≥ success_threshold (model-card criterion)
    env_terminated_success: bool     # env's own terminated=True signal (pc_success criterion)
    max_reward: float                # peak overlap fraction reached
    sum_reward: float                # accumulated step reward across episode
    n_steps: int                     # length of episode in env steps
    inference_time_s: float          # wall-clock policy inference time only


@dataclass
class EvalMetrics:
    """Aggregate stats across N episodes — the comparison row for one policy.

    Two success columns are reported to make the eval-criterion ambiguity explicit:

      success_rate / success_ci_*
          max_reward ≥ success_threshold (0.95) across the episode.
          This matches the LeRobot model card's definition ("max overlap criterion").
          More lenient: the block only needs to touch the target zone at peak.

      env_success_rate / env_success_ci_*
          The env's own terminated=True signal, equivalent to the lerobot-eval CLI's
          pc_success metric (sustained overlap). This is the stricter criterion.
          With gymnasium ≥ 1.0 the threshold is tighter — see LeRobot issue #470.

    Never conflate the two when comparing against published numbers.
    """

    policy_name: str
    n_episodes: int
    success_rate: float              # max_reward ≥ threshold (model-card criterion)
    success_ci_low: float            # Wilson 95% CI lower bound
    success_ci_high: float           # Wilson 95% CI upper bound
    env_success_rate: float          # env terminated=True rate (pc_success criterion)
    env_success_ci_low: float
    env_success_ci_high: float
    avg_max_reward: float
    avg_sum_reward: float
    avg_episode_length: float
    avg_inference_time_s: float
    success_threshold: float
    wall_time_s: float               # total eval wall time
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert nested EpisodeMetrics dataclasses cleanly
        d["episodes"] = [asdict(e) for e in self.episodes]
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("saved eval metrics: %s", path)
        return path


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


class PolicyAdapter(Protocol):
    """Minimum interface every policy must satisfy to be eval'd uniformly.

    Why a Protocol not an ABC: structural typing — any object with these two
    attributes works. Lets us wrap LeRobot's PreTrainedPolicy classes without
    inheritance.
    """

    name: str

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Map an env observation dict to an action.

        observation: dict from gym-pusht env. With obs_type='pixels_agent_pos':
          {'pixels': (H, W, 3) uint8, 'agent_pos': (2,) float32}

        Returns: action as a (action_dim,) numpy array in env coords.
        """
        ...

    def reset(self) -> None:
        """Called once per episode. Use to clear any internal policy state
        (e.g. action chunk buffers, observation history)."""
        ...


class MLPBCAdapter:
    """Wraps our from-scratch MLP-BC. State-only, single-step prediction."""

    def __init__(self, model: MLPBC, name: str = "mlp_bc", device: str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.name = name
        self.device = device

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        # PushT env returns 'agent_pos' as the state vector — same content as
        # the dataset's observation.state (verified during Phase 1 inspection).
        state = torch.from_numpy(observation["agent_pos"]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model.predict_action(state)
        return action.squeeze(0).cpu().numpy()

    def reset(self) -> None:
        # Stateless policy — nothing to reset.
        pass

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str | Path, device: str = "cpu") -> "MLPBCAdapter":
        """Load an MLP-BC checkpoint saved by ``scripts/train_mlp_bc.py``.

        Expects ``ckpt_dir/pretrained_model/{model.safetensors,config.json}``,
        the same layout the training script writes.
        """
        ckpt_dir = Path(ckpt_dir)
        pretrained = ckpt_dir if (ckpt_dir / "model.safetensors").exists() else ckpt_dir / "pretrained_model"
        cfg_path = pretrained / "config.json"
        model_path = pretrained / "model.safetensors"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing config.json in {pretrained}")
        if not model_path.exists():
            raise FileNotFoundError(f"missing model.safetensors in {pretrained}")
        cfg_dict = json.loads(cfg_path.read_text())
        # Filter to only fields the dataclass knows — robust to extra fields
        import dataclasses

        valid = {f.name for f in dataclasses.fields(MLPBCConfig)}
        cfg = MLPBCConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        model = MLPBC.load(model_path, cfg)
        return cls(model, name=ckpt_dir.parent.parent.name, device=device)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def wilson_score_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lower, upper) bounds of Wilson 95% CI for a binomial proportion.

    Why Wilson over a normal-approximation CI: handles edge cases (0/N or N/N)
    sensibly without producing impossible bounds. Standard for small-N binomial.
    Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _run_one_episode(
    policy: PolicyAdapter,
    env: gym.Env,
    seed: int,
    episode_index: int,
    success_threshold: float,
    max_steps: int | None = None,
) -> EpisodeMetrics:
    """Run one rollout, return metrics. Stops on env terminal or max_steps."""
    policy.reset()
    obs, _info = env.reset(seed=seed)
    sum_reward = 0.0
    max_reward = 0.0
    n_steps = 0
    inference_time_s = 0.0
    env_terminated_success = False  # True iff the env itself raised terminated=True

    while True:
        t0 = time.perf_counter()
        action = policy.predict(obs)
        inference_time_s += time.perf_counter() - t0

        obs, reward, terminated, truncated, _info = env.step(action)
        sum_reward += float(reward)
        # PushT reward is the block-target overlap fraction at this step (verified
        # in Phase 1 inspection — `next.reward` field). Max across the episode is
        # the deciding metric for success.
        max_reward = max(max_reward, float(reward))
        n_steps += 1

        if terminated:
            # gym-pusht raises terminated=True when block-target overlap has been
            # sustained above threshold — this is the pc_success criterion used by
            # `lerobot-eval`. It is stricter than max_reward ≥ success_threshold.
            env_terminated_success = True
            break
        if truncated:
            # Episode hit the time limit without success (env's view).
            break
        if max_steps is not None and n_steps >= max_steps:
            break

    return EpisodeMetrics(
        episode_index=episode_index,
        seed=seed,
        success=(max_reward >= success_threshold),
        env_terminated_success=env_terminated_success,
        max_reward=max_reward,
        sum_reward=sum_reward,
        n_steps=n_steps,
        inference_time_s=inference_time_s,
    )


def run_evaluation(
    policy: PolicyAdapter,
    n_episodes: int = 50,
    base_seed: int = 100000,
    success_threshold: float = 0.95,
    env_kwargs: dict[str, Any] | None = None,
    max_steps_per_episode: int | None = None,
    progress_log_interval: int = 5,
) -> EvalMetrics:
    """Run ``n_episodes`` rollouts, return aggregate metrics.

    Seeds are deterministic: episode i uses seed = base_seed + i. This means the
    same i-th episode for two different policies starts in identical env state,
    enabling like-for-like comparison.

    ``env_kwargs`` override gym-pusht defaults. Defaults match LeRobot's recipe
    (verified Phase 2): obs_type='pixels_agent_pos', observation_width/height=96.
    """
    env_kwargs = {
        "obs_type": "pixels_agent_pos",
        "observation_width": 96,
        "observation_height": 96,
        "render_mode": "rgb_array",
        **(env_kwargs or {}),
    }
    env = gym.make("gym_pusht/PushT-v0", **env_kwargs)
    logger.info("eval %s: %d episodes, success_threshold=%.2f", policy.name, n_episodes, success_threshold)

    t0 = time.time()
    episodes: list[EpisodeMetrics] = []
    for i in range(n_episodes):
        ep = _run_one_episode(
            policy=policy,
            env=env,
            seed=base_seed + i,
            episode_index=i,
            success_threshold=success_threshold,
            max_steps=max_steps_per_episode,
        )
        episodes.append(ep)
        if (i + 1) % progress_log_interval == 0 or i == n_episodes - 1:
            sr_so_far = sum(e.success for e in episodes) / len(episodes)
            logger.info(
                "  ep %3d/%d  success_so_far=%5.1f%%  ep_max_reward=%.3f  ep_steps=%d",
                i + 1, n_episodes, 100 * sr_so_far, ep.max_reward, ep.n_steps,
            )
    env.close()

    successes = sum(e.success for e in episodes)
    env_successes = sum(e.env_terminated_success for e in episodes)
    ci_low, ci_high = wilson_score_interval(successes, n_episodes)
    env_ci_low, env_ci_high = wilson_score_interval(env_successes, n_episodes)

    metrics = EvalMetrics(
        policy_name=policy.name,
        n_episodes=n_episodes,
        success_rate=successes / n_episodes if n_episodes else 0.0,
        success_ci_low=ci_low,
        success_ci_high=ci_high,
        env_success_rate=env_successes / n_episodes if n_episodes else 0.0,
        env_success_ci_low=env_ci_low,
        env_success_ci_high=env_ci_high,
        avg_max_reward=float(np.mean([e.max_reward for e in episodes])),
        avg_sum_reward=float(np.mean([e.sum_reward for e in episodes])),
        avg_episode_length=float(np.mean([e.n_steps for e in episodes])),
        avg_inference_time_s=float(np.mean([e.inference_time_s for e in episodes])),
        success_threshold=success_threshold,
        wall_time_s=time.time() - t0,
        episodes=episodes,
    )
    logger.info(
        "eval %s done:"
        " max_overlap_success=%.1f%% [%.1f%%, %.1f%%]"
        " pc_success=%.1f%% [%.1f%%, %.1f%%]"
        " avg_max_reward=%.3f wall=%.1fs",
        policy.name,
        100 * metrics.success_rate, 100 * ci_low, 100 * ci_high,
        100 * metrics.env_success_rate, 100 * env_ci_low, 100 * env_ci_high,
        metrics.avg_max_reward, metrics.wall_time_s,
    )
    return metrics


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Smoke: instantiate a randomly-initialised MLP-BC, run 3 episodes.

    Won't show useful success_rate (untrained) but verifies the pipeline:
    env creation, policy adapter, episode loop, metric aggregation.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = MLPBCConfig()
    model = MLPBC(cfg)
    # Fit normalisers on synthetic data spanning the env coord range so action
    # outputs are in the right ballpark (else the policy spams out-of-bounds actions).
    fake_state = torch.rand(100, 2) * 480 + 20
    fake_action = torch.rand(100, 2) * 480 + 20
    model.input_normalizer.fit(fake_state)
    model.output_normalizer.fit(fake_action)

    adapter = MLPBCAdapter(model, name="mlp_bc_smoke")
    metrics = run_evaluation(adapter, n_episodes=3, max_steps_per_episode=50, progress_log_interval=1)

    # Wilson sanity: 0/3 successes → CI should include 0 but not exceed ~70%
    print(f"\nWilson check (max_overlap): 0/3 → ({metrics.success_ci_low:.3f}, {metrics.success_ci_high:.3f})")
    print(f"Wilson check (env pc_success): 0/3 → ({metrics.env_success_ci_low:.3f}, {metrics.env_success_ci_high:.3f})")
    assert 0.0 <= metrics.success_ci_low <= metrics.success_ci_high <= 1.0
    assert 0.0 <= metrics.env_success_ci_low <= metrics.env_success_ci_high <= 1.0
    print("smoke test PASSED")
