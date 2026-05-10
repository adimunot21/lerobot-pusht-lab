# RunPod quickstart — full benchmark suite

End-to-end recipe for training all three policies (Diffusion / ACT / MLP-BC) on
RunPod and producing the Phase 5 comparison report. Targets paper-level numbers
(Diffusion ~85-91%, ACT ~30-50%, MLP-BC ~5%).

**Cost estimate:** ~$3-5 on an RTX 4090 24GB (Community Cloud @ $0.34/hr, ~12h
total wall time including Phase 5 eval).

**Prerequisites:**
- A RunPod account with credit
- HF account + write-scope token at https://huggingface.co/settings/tokens
- An SSH client locally

---

## 1. Spin up the pod

- Go to https://www.runpod.io → **Deploy**
- **Community Cloud** → **RTX 4090 24GB** (cheapest, ~$0.34/hr)
- **Template:** any `pytorch:2.x-cuda12.x` image (e.g. "RunPod PyTorch 2.4")
- **Disk:** 50 GB Network Volume (persistent — survives eviction) + 50 GB ephemeral
- **Expose SSH** → on (default)
- Deploy.

Once the pod is "Running", grab the SSH command from the dashboard. It looks
like `ssh root@<ip> -p <port> -i ~/.ssh/<keyfile>`.

## 2. Initial setup on the pod (~10 min)

```bash
# Headless rendering for gym-pusht (uses pygame which needs a display).
apt-get update && apt-get install -y xvfb git tmux

# Clone the repo
git clone https://github.com/adimunot21/lerobot-pusht-lab.git
cd lerobot-pusht-lab

# Verify CUDA is visible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install Python deps. Do NOT use --no-deps — we need transitive packages too.
pip install -r requirements.txt
pip install -e .
```

Verify the imports come up clean:

```bash
python -c "
import torch, lerobot, gym_pusht
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('lerobot:', lerobot.__version__)
print('all imports OK')
"
```

Configure HF token (used by Phase 6):

```bash
cp .env.example .env
echo "HF_TOKEN=hf_YOUR_TOKEN_HERE" >> .env
```

## 3. Smoke test (~3 min)

Verify the full 263M-param U-Net fits at batch 64. With 24 GB VRAM peak should
be ~6-8 GB.

```bash
xvfb-run -s "-screen 0 1024x768x24" \
  ./scripts/train_diffusion.sh \
    --steps=50 --eval_freq=1000 --save_freq=1000 --log_freq=10 \
    --output_dir=checkpoints/diffusion_pusht_smoke \
    --job_name=diffusion_pusht_smoke 2>&1 | tail -20
```

Expect to see `num_learnable_params=262709026 (263M)` and 50 steps in ~30s with
no OOM. Clean up: `rm -rf checkpoints/diffusion_pusht_smoke`.

## 4. Full training runs (in tmux)

```bash
tmux new -s training
cd ~/lerobot-pusht-lab

# Diffusion: ~6h (200K × 64, full U-Net)
xvfb-run ./scripts/train_diffusion.sh 2>&1 | tee logs/diffusion_$(date +%Y%m%d_%H%M).log

# Detach with Ctrl-B then D. Re-attach with `tmux attach -t training`.
```

When Diffusion finishes:

```bash
# ACT: ~5h (100K × 16-32 default)
# Smoke first to confirm batch fits (RTX 4090 should easily fit batch 32 or 64):
xvfb-run ./scripts/train_act.sh \
    --steps=50 --batch_size=32 --eval_freq=1000 --save_freq=1000 \
    --output_dir=checkpoints/act_pusht_smoke 2>&1 | tail -10
# If green, run full:
xvfb-run ./scripts/train_act.sh --batch_size=32 2>&1 | tee logs/act_$(date +%Y%m%d_%H%M).log

# MLP-BC: 30 seconds. Doesn't need xvfb (no env rollouts during training).
python scripts/train_mlp_bc.py
```

## 5. Phase 5: evaluate all 3 policies (~25-30 min)

```bash
# Runs 50-episode eval per policy with deterministic seeds + writes the
# cross-policy comparison report to outputs/comparison/.
xvfb-run python scripts/eval_all.py 2>&1 | tee logs/eval.log
```

Output:
- `outputs/eval/<policy>/eval_metrics.json`
- `outputs/comparison/comparison.md`
- `outputs/comparison/{success_rate_with_ci,max_reward_distribution}.png`

## 6. Phase 6: publish to HF Hub (~2 min)

```bash
python scripts/upload_to_hub.py --dry-run   # render cards, verify layout
python scripts/upload_to_hub.py              # publish
```

This creates 3 public repos at `adimunot21/<policy>-pusht-lab` with model
cards containing your eval results.

## 7. Pull results back locally + commit

From your laptop (NOT from the pod):

```bash
cd ~/projects/lerobot-pusht-lab

# Sync just the small artifacts (eval JSONs, comparison report, plots).
# Replace <ip> and <port> with your pod's SSH details.
rsync -avz -e "ssh -p <port> -i <keyfile>" \
    root@<ip>:lerobot-pusht-lab/outputs/eval/ ./outputs/eval/
rsync -avz -e "ssh -p <port> -i <keyfile>" \
    root@<ip>:lerobot-pusht-lab/outputs/comparison/ ./outputs/comparison/
rsync -avz -e "ssh -p <port> -i <keyfile>" \
    root@<ip>:lerobot-pusht-lab/logs/ ./logs/

git add outputs/eval outputs/comparison logs/
git commit -m "Add Phase 5 eval results from RunPod"
git push
```

## 8. Stop the pod

In the RunPod dashboard, **Stop** (not Terminate — Stop preserves the volume
in case you need to re-run anything; Terminate deletes it). Or if you're sure
you're done, Terminate to free the volume rent.

---

## Troubleshooting

**Pod evicted mid-training (Community Cloud):**
- Just spin up a new pod, mount the same network volume. Resume:
  `xvfb-run ./scripts/train_diffusion.sh --resume`

**`gym-pusht` errors with "no display":**
- All eval / training-eval invocations need `xvfb-run` prefix on cloud GPUs.
  pygame doesn't have a true headless mode.

**OOM at full U-Net on the 4090:**
- Shouldn't happen at batch 64 (smoke confirms ~8 GB peak). If it does, drop
  batch size — `--batch_size=48` or `--batch_size=32`. The U-Net dim is the
  binding lever for memory; smaller batch is the cheap fix.

**Wandb wants to log in:**
- Project uses offline mode by default (`wandb.mode: offline`). If you ever
  want cloud logging, comment out the `mode: offline` line in the configs and
  set `WANDB_API_KEY` in `.env`.
