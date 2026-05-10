"""LeRobot policy adapter — wrap any LeRobot ``PreTrainedPolicy`` as a ``PolicyAdapter``.

Plugs Diffusion + ACT into the same eval loop the MLP-BC uses. The mapping
between gym-pusht observation dicts and LeRobot's batched tensor format is
the only per-policy nuance — once that's right, ``select_action`` handles
internal state (action-chunk buffers, observation history) automatically.

Why a separate file: keeps ``runner.py`` GPU-free and importable from CPU-only
environments (notebooks, tests). LeRobot's PreTrainedPolicy import chain pulls
in CUDA-aware modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def _get_policy_class(policy_type: str) -> type[PreTrainedPolicy]:
    """Map ``config.json`` policy ``type`` field → concrete class.

    Extend this dict when wiring up additional policy types. We do this lazily
    so importing ``lerobot_adapter`` doesn't pull in every LeRobot policy
    module on import (some have heavy CUDA-aware dependencies).
    """
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        return DiffusionPolicy
    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    raise ValueError(
        f"unknown policy type {policy_type!r}; "
        "extend _get_policy_class() in lerobot_adapter.py"
    )


class LeRobotPolicyAdapter:
    """Wraps a ``PreTrainedPolicy`` (Diffusion or ACT) as a ``PolicyAdapter``.

    The wrapper handles three things:

    1. **Observation translation.** gym-pusht returns
       ``{'pixels': uint8 HWC, 'agent_pos': float32 (2,)}``. LeRobot expects
       ``{'observation.image': float32 BCHW [0,1], 'observation.state': float32 B×2}``.
       Conversion is done in ``predict``.

    2. **Action denormalisation.** LeRobot's policies output actions through
       their internal post-processor (already in env coords). We just read
       the result. No extra work needed.

    3. **Episode reset.** ``select_action`` maintains an internal action-chunk
       buffer (Diffusion) or obs history (ACT). Calling ``policy.reset()``
       between episodes clears it. We do this in ``reset``.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        name: str,
        device: str = "cuda",
        preprocessor: Any = None,
        postprocessor: Any = None,
    ) -> None:
        """``preprocessor`` and ``postprocessor`` are LeRobot
        ``PolicyProcessorPipeline`` objects loaded from the same checkpoint dir.
        They handle observation normalisation (image MEAN_STD, state MIN_MAX
        for Diffusion / MEAN_STD for ACT) and action denormalisation back to
        env coords. Without them, ``select_action`` returns actions in the
        normalised [-1, 1] space — which is wrong by 250-pixel orders of
        magnitude in env coords. Both must be loaded from the checkpoint.
        """
        self.policy = policy.to(device)
        self.policy.eval()
        self.name = name
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str | Path, name: str | None = None,
                        device: str = "cuda") -> "LeRobotPolicyAdapter":
        """Load from a LeRobot-format checkpoint directory.

        Expects ``ckpt_dir/pretrained_model/`` (the layout LeRobot writes), or
        ``ckpt_dir`` may itself be the ``pretrained_model/`` dir. We try both.

        Dispatches to the right concrete subclass (DiffusionPolicy, ACTPolicy,
        ...) by reading the policy ``type`` field from ``config.json``.
        ``PreTrainedPolicy`` is abstract and can't be instantiated directly.
        """
        import json

        ckpt_dir = Path(ckpt_dir)
        if (ckpt_dir / "pretrained_model").exists():
            pretrained = ckpt_dir / "pretrained_model"
        elif (ckpt_dir / "model.safetensors").exists():
            pretrained = ckpt_dir
        else:
            raise FileNotFoundError(
                f"no model.safetensors or pretrained_model/ found at {ckpt_dir}"
            )

        # Determine policy class from the saved config's `type` field. There's
        # no public registry exposed on PreTrainedPolicy, so we map by name —
        # extend this dict when adding new policy classes.
        cfg_path = pretrained / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing config.json in {pretrained}")
        with open(cfg_path) as f:
            policy_cfg = json.load(f)
        policy_type = policy_cfg.get("type")
        if policy_type is None:
            raise ValueError(f"config.json has no `type` field: {cfg_path}")

        policy_class = _get_policy_class(policy_type)
        policy = policy_class.from_pretrained(pretrained)

        # Load the pre/post processors that LeRobot saves alongside the policy.
        # Without these, select_action returns actions in normalised space.
        # The processors live in the same dir as model.safetensors.
        from lerobot.policies.factory import make_pre_post_processors

        preprocessor_overrides = {
            "device_processor": {"device": device},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(pretrained),
            preprocessor_overrides=preprocessor_overrides,
        )

        if name is None:
            # Walk up from pretrained_model/ → checkpoints/<step> → output_dir
            name = pretrained.parent.parent.parent.name or "lerobot_policy"
        return cls(
            policy, name=name, device=device,
            preprocessor=preprocessor, postprocessor=postprocessor,
        )

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """gym-pusht obs → LeRobot batch dict → action numpy (env coords).

        Pipeline (mirrors lerobot_eval.rollout):
          1. Build batch dict from gym obs
          2. preprocessor: applies device move + image normalisation
          3. select_action: policy forward (handles internal action chunk buffer)
          4. postprocessor: denormalises action back to env coords
        """
        batch = self._observation_to_batch(observation)
        if self.preprocessor is not None:
            batch = self.preprocessor(batch)
        with torch.no_grad():
            action_tensor = self.policy.select_action(batch)
        if self.postprocessor is not None:
            action_tensor = self.postprocessor(action_tensor)
        return action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        self.policy.reset()

    # ------------------------------------------------------------------

    def _observation_to_batch(self, observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Translate gym-pusht obs to LeRobot's expected dict format.

        gym-pusht ``pixels_agent_pos`` obs format:
            pixels:    (H, W, 3) uint8     — env render
            agent_pos: (2,) float32        — agent xy position

        LeRobot policy input format (verified by inspecting the trained
        Diffusion config — ``input_features`` keys):
            observation.image: (B=1, 3, 96, 96) float32 in [0, 1]
            observation.state: (B=1, 2)         float32

        The image normalisation step (mean/std) is handled INSIDE the policy
        by its preprocessor, so we just hand it [0, 1]-scaled values.
        """
        out: dict[str, torch.Tensor] = {}

        if "pixels" in observation:
            img = observation["pixels"]
            # uint8 HWC → float32 [0,1] CHW with batch dim
            t = torch.from_numpy(img).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)  # (H,W,C) → (1,C,H,W)
            out["observation.image"] = t.to(self.device)

        if "agent_pos" in observation:
            t = torch.from_numpy(observation["agent_pos"]).float().unsqueeze(0)
            out["observation.state"] = t.to(self.device)

        if not out:
            raise ValueError(
                f"observation has no expected keys (got {list(observation)}); "
                "expected 'pixels' and/or 'agent_pos'"
            )
        return out


# ---------------------------------------------------------------------------
# Smoke / verification (uses an archived checkpoint if available)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Smoke: load an archived LeRobot checkpoint, run a 1-step prediction.

    Skipped silently if no checkpoint is available (e.g. fresh clone).
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    candidates = [
        Path("checkpoints/diffusion_pusht_50K_b8_5pc/checkpoints/050000"),
        Path("checkpoints/diffusion_pusht/checkpoints/last"),
        Path("checkpoints/act_pusht/checkpoints/last"),
    ]
    ckpt = next((c for c in candidates if c.exists()), None)
    if ckpt is None:
        print("no checkpoint available; skipping smoke")
        raise SystemExit(0)

    print(f"loading checkpoint: {ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use CPU during smoke so we don't compete with any in-progress training
    # — this should "just work" because LeRobot policies are device-agnostic.
    adapter = LeRobotPolicyAdapter.from_checkpoint(ckpt, device="cpu")
    print(f"adapter: {adapter.name}")

    # Fake gym-pusht observation
    fake_obs = {
        "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
        "agent_pos": np.array([256.0, 256.0], dtype=np.float32),
    }
    adapter.reset()
    action = adapter.predict(fake_obs)
    print(f"action: shape={action.shape} dtype={action.dtype} value={action}")
    assert action.shape == (2,)
    assert action.dtype == np.float32
    print("smoke test PASSED")
