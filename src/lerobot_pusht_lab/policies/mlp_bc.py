"""From-scratch MLP-BC baseline policy for PushT.

This is the "obvious" approach: a small MLP that maps the current agent state
to the next action, trained with MSE loss on expert demonstrations. It exists
as a *baseline* — not because we expect it to work well, but because **its
expected failure mode is the lesson** that motivates Diffusion Policy and ACT.

## Why we expect this to fail (the pedagogical point)

PushT expert demonstrations are *multimodal*. To push a block to a target,
two strategies are equally valid: approach from the left and push right, or
approach from the right and push left. Both appear in the dataset.

MSE loss minimises squared error to **all** demonstrations simultaneously.
For two valid strategies that diverge symmetrically, this means the network
learns to predict their **average** — which is to walk straight at the block
from above, missing it entirely.

This is the **mode-averaging** problem. Diffusion Policy and ACT solve it by
predicting an action *distribution* and sampling from it. MLP-BC cannot.

## What this module contains

- `StateActionNormalizer` — explicit min-max normalisation. PushT state and
  action are in raw env coordinates ~[44, 452] (verified Phase 1), not [0, 1].
  LeRobot's diffusion/ACT pipelines apply this internally; we have to do it
  ourselves because we're not using the LeRobot policy framework.
- `MLPBC` — the model. 2-layer MLP, 256 hidden, ReLU, dropout 0.1, per
  PLAN.md §7 (Phase 4) spec.
- `MLPBCConfig` — dataclass with all hyperparameters in one place.

## What this module does NOT contain

- The training loop → `scripts/train_mlp_bc.py`.
- The eval rollout → handled in Phase 5's eval harness alongside the other policies.

## Comparable libraries / "what would a manual implementation look like"

This *is* the manual implementation. There's no library shortcut for "tiny MLP
trained with MSE" — it's six lines of PyTorch. Equivalent libraries that
would do this for you (and which we'd reach for in production):
  - `stable_baselines3.bc` — has a BC class, but it's RL-flavoured and overkill here.
  - `imitation` library — same story.
  - LeRobot itself has no MLP-BC policy class; closest is the `pi0` family.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MLPBCConfig:
    """Hyperparameters for MLP-BC. All justifications in the field comments."""

    # Architecture (PLAN.md §7 Phase 4 spec).
    input_dim: int = 2  # observation.state for PushT (agent x, y position)
    output_dim: int = 2  # action for PushT (target x, y for agent)
    hidden_dim: int = 256  # PLAN.md spec; ample for a 2D→2D mapping
    num_hidden_layers: int = 2  # PLAN.md spec; "2-layer MLP"
    dropout: float = 0.1  # PLAN.md spec; mild regularisation
    activation: str = "relu"  # PLAN.md spec

    # Optimiser. Adam with relatively high lr — the model is tiny (~70K params)
    # so it tolerates aggressive updates. Compare: Diffusion Policy uses 1e-4
    # because the U-Net has ~76M params and would diverge at higher lrs.
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-5  # mild — prevents overfitting on 25K frames

    # Training schedule. Steps not epochs because that's the LeRobot convention
    # and keeps the comparison report uniform across all 3 policies.
    # 25650 frames / batch 256 ≈ 100 steps per epoch. 5K steps = 50 epochs,
    # which is more than enough for a model this small to plateau.
    batch_size: int = 256
    num_workers: int = 4
    steps: int = 5000
    log_freq: int = 100  # log every 100 steps → 50 log points across training
    eval_freq: int = 1000  # 5 evals across training (1000, 2000, ..., 5000)
    save_freq: int = 1000  # 5 checkpoints
    seed: int = 1000  # match Diffusion Policy config for fair comparison

    # Device
    device: str = "cuda"  # GPU is barely needed for this model size, but it's
                          # already in use by Diffusion/ACT — uniform device choice
                          # avoids data-movement gotchas in the comparison report.

    # Output
    output_dir: str = "checkpoints/mlp_bc_pusht"

    # Dataset
    dataset_repo_id: str = "lerobot/pusht"

    # Eval — used by Phase 5's harness; included here so the config is self-contained.
    eval_n_episodes: int = 20  # in-training; Phase 5 does the proper 50-ep eval


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


class StateActionNormalizer(nn.Module):
    """Min-max normalisation to [-1, 1], saved as buffers so it travels with the model.

    Why min-max not z-score: LeRobot's diffusion/ACT pipelines use min-max for
    STATE/ACTION (see DiffusionConfig.normalization_mapping, verified Phase 2).
    Matching that choice keeps the comparison clean — any difference between
    MLP-BC and the LeRobot policies is attributable to the *policy*, not to
    upstream data transforms.

    Why [-1, 1] not [0, 1]: makes the "no action" centre symmetric, which
    interacts better with `tanh`-bounded outputs if we ever want them. For our
    MLP we use linear output, but the convention is harmless and standard.

    Buffers (not parameters) → moved with .to(device) but not updated by the
    optimiser. Saved/loaded by safetensors transparently.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Initialised to identity (min=-1, max=1) so an untrained / unfit normaliser
        # is a no-op rather than a divide-by-zero.
        self.register_buffer("min", torch.full((dim,), -1.0))
        self.register_buffer("max", torch.full((dim,), 1.0))
        self._fitted: bool = False

    def fit(self, samples: torch.Tensor) -> None:
        """Compute min/max from a tensor of shape (N, dim).

        Equivalent manual: `min = samples.min(0).values; max = samples.max(0).values`.
        Wrapped in a method so the training script can call `.fit(...)` once at
        startup and never have to think about it again.
        """
        if samples.dim() != 2 or samples.shape[1] != self.min.shape[0]:
            raise ValueError(
                f"expected (N, {self.min.shape[0]}) tensor, got {tuple(samples.shape)}"
            )
        # Add a tiny epsilon to the range to guard against constant features
        # (would cause divide-by-zero in normalize). 1e-8 is safe for any
        # reasonable physical units.
        mn = samples.min(dim=0).values
        mx = samples.max(dim=0).values
        eps = 1e-8
        # If a feature is exactly constant, expand its range to ±0.5 so it
        # normalises to 0. Pure paranoia — PushT state/action are not constant.
        constant_mask = (mx - mn) < eps
        mx = torch.where(constant_mask, mn + 1.0, mx)
        mn = torch.where(constant_mask, mn - 0.0, mn)
        self.min.copy_(mn)
        self.max.copy_(mx)
        self._fitted = True
        logger.info("normaliser fit: min=%s max=%s", self.min.tolist(), self.max.tolist())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map x ∈ [min, max] → [-1, 1]. Linear, invertible."""
        return 2 * (x - self.min) / (self.max - self.min) - 1

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of normalize."""
        return (x + 1) / 2 * (self.max - self.min) + self.min


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation {name!r}")


class MLPBC(nn.Module):
    """State-only MLP behaviour-cloning policy.

    Architecture: input → [hidden, activation, dropout] × num_hidden_layers → output.
    No batch/layer norm — overkill at this scale.

    Forward signature deliberately matches the typical LeRobot policy convention
    (dict input/output) so Phase 5's eval harness can call all 3 policies
    uniformly. This adds ~3 lines of dict-shuffling but pays off downstream.
    """

    def __init__(self, config: MLPBCConfig) -> None:
        super().__init__()
        self.config = config
        self.input_normalizer = StateActionNormalizer(config.input_dim)
        self.output_normalizer = StateActionNormalizer(config.output_dim)

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(_activation_module(config.activation))
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict normalised action from raw (unnormalised) state.

        state: (B, input_dim) in env coords. Returns: (B, output_dim) in [-1, 1].

        Caller is responsible for denormalising the output if they want env coords —
        during training, we keep everything normalised so the loss is in a
        scale-free [-1,1]² space. At inference time, use `predict_action(...)`.
        """
        return self.net(self.input_normalizer.normalize(state))

    def predict_action(self, state: torch.Tensor) -> torch.Tensor:
        """Inference: raw state → raw action (env coords). Inverse-normalises."""
        normalised_action = self.forward(state)
        return self.output_normalizer.denormalize(normalised_action)

    def loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """MSE in normalised space. Both inputs are in raw env coords."""
        predicted = self.forward(state)
        target = self.output_normalizer.normalize(action)
        return torch.nn.functional.mse_loss(predicted, target)

    # ---- persistence ----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write weights + buffers to safetensors."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(path))
        logger.info("saved model: %s", path)

    @classmethod
    def load(cls, path: str | Path, config: MLPBCConfig) -> "MLPBC":
        """Restore from safetensors. Caller must supply the matching config."""
        model = cls(config)
        state = load_file(str(Path(path)))
        model.load_state_dict(state)
        return model


# ---------------------------------------------------------------------------
# Smoke test (CLAUDE.md mandates a runnable __main__ in every module)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = MLPBCConfig()
    model = MLPBC(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLPBC params: {n_params:,}")

    # Fake batch of 32 frames in PushT-like coords (~[40, 460])
    fake_state = torch.rand(32, cfg.input_dim) * 420 + 40
    fake_action = torch.rand(32, cfg.output_dim) * 420 + 40

    # Fit normalisers on the fake data
    model.input_normalizer.fit(fake_state)
    model.output_normalizer.fit(fake_action)

    # Forward + loss should run end-to-end
    loss = model.loss(fake_state, fake_action)
    print(f"forward + loss OK: loss={loss.item():.4f}")

    # Inference returns env-coord action
    action = model.predict_action(fake_state[:3])
    print(f"predict_action OK: shape={tuple(action.shape)}, range=[{action.min():.1f}, {action.max():.1f}]")

    # Persistence round-trip — compare in eval() mode so dropout is deterministic
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        model.save(f.name)
        restored = MLPBC.load(f.name, cfg)
    model.eval()
    restored.eval()
    with torch.no_grad():
        loss_eval = model.loss(fake_state, fake_action)
        restored_loss = restored.loss(fake_state, fake_action)
    assert torch.allclose(loss_eval, restored_loss), "persistence round-trip failed"
    print(f"persistence round-trip OK: loss={restored_loss.item():.4f}")

    print("smoke test PASSED")
