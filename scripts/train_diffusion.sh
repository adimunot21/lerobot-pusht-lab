#!/usr/bin/env bash
# Train Diffusion Policy on lerobot/pusht.
#
# Reads configs/diffusion_pusht.yaml. Any extra args are forwarded to
# lerobot-train, which lets us reuse this script for the smoke test:
#
#   # Smoke test (100 steps, ~5 min):
#   ./scripts/train_diffusion.sh \
#       --steps=100 --eval_freq=1000 --save_freq=1000 \
#       --output_dir=checkpoints/diffusion_pusht_smoke \
#       --job_name=diffusion_pusht_smoke
#
#   # Full overnight run:
#   ./scripts/train_diffusion.sh
#
# Long-running variant: launch inside tmux per CLAUDE.md workflow rule
# ("Long training runs go in tmux outside Claude Code").
#
# Why a wrapper rather than calling lerobot-train directly: pins the exact
# config path + sources .env reproducibly, and gives us one place to add
# pre-flight checks (CUDA visibility, dataset cache freshness) if we ever need
# them.

set -euo pipefail

# Always run from the repo root so relative paths in the YAML resolve.
cd "$(dirname "$0")/.."

# Load HF_TOKEN etc. from .env if present. Set -a exports every var read while
# active, so they propagate to the lerobot-train subprocess.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Fail fast if the env doesn't have CUDA — the policy will silently fall back
# to CPU otherwise and the user will think it's hung.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "ERROR: CUDA not available in current Python env. Activate 'lerobot' conda env." >&2; exit 1; }

# Reduce CUDA allocator fragmentation. Recommended by the OOM error during
# Phase 2 smoke test: with the policy + Adam state filling ~75% of VRAM,
# fragmentation can cause spurious failures where allocations larger than the
# largest free block fail despite enough total free memory. Expandable segments
# let the allocator grow contiguously instead of pre-reserving chunks.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Wandb in offline mode — writes runs to ./wandb/offline-run-* without contacting
# the cloud. LeRobot 0.5.1 has no TensorBoard fallback, so we need wandb for
# plottable logs, but the user's cloud quota is exhausted. Offline mode bypasses
# this. Sync later with `wandb sync ./wandb/offline-run-*` if quota frees up.
export WANDB_MODE="${WANDB_MODE:-offline}"
# Silence the wandb login prompt — offline mode doesn't auth, but the library
# still nags about missing credentials without this.
export WANDB_SILENT="${WANDB_SILENT:-true}"

exec lerobot-train \
    --config_path=configs/diffusion_pusht.yaml \
    "$@"
