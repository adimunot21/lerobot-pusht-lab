#!/usr/bin/env bash
# Train ACT (Action Chunking Transformer) on lerobot/pusht.
#
# Reads configs/act_pusht.yaml. Extra args forwarded to lerobot-train, so
# the same script handles smoke and full runs:
#
#   # Smoke (50 steps, no eval, batch sweep one-by-one):
#   ./scripts/train_act.sh \
#       --steps=50 --eval_freq=1000 --save_freq=1000 \
#       --batch_size=32 \
#       --output_dir=checkpoints/act_pusht_smoke \
#       --job_name=act_pusht_smoke
#
#   # Extended smoke (1000 steps + 2 eval rounds — verifies eval doesn't OOM):
#   ./scripts/train_act.sh \
#       --steps=1000 --eval_freq=500 --save_freq=1000 --eval.n_episodes=10 \
#       --output_dir=checkpoints/act_pusht_extsmoke
#
#   # Full run:
#   ./scripts/train_act.sh
#
# Long-running runs go in tmux per CLAUDE.md workflow rule.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
    || { echo "ERROR: CUDA not available in current Python env. Activate 'lerobot' conda env." >&2; exit 1; }

# Same allocator + wandb-offline setup as the Diffusion wrapper.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

exec lerobot-train \
    --config_path=configs/act_pusht.yaml \
    "$@"
