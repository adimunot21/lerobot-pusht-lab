# LeRobot PushT Lab — Project Plan

**Status:** Phase 1 complete (Dataset Inspection) → next: Phase 2 (Diffusion Policy training)
**Owner:** adimunot21
**Started:** 2026-05-09
**Target completion:** ~1 week of evenings/days, before SO-101 arm arrival

---

## 1. Goal & Success Criteria

**Goal:** Build complete fluency with the LeRobot + HuggingFace Hub stack on a sim task,
so that day-one workflow with the SO-101 arm is "plug in, record, train, eval" rather
than "fight the tooling."

**Success criteria — the project is "done" when ALL of the following are true:**

1. Three policies trained on the `lerobot/pusht` dataset:
   - Diffusion Policy (using LeRobot's implementation, image observations)
   - ACT (using LeRobot's implementation, image observations)
   - Custom from-scratch MLP-BC baseline (state-only, written by me)
2. All three evaluated in `gym-pusht` over ≥50 episodes each, with a comparison report
   (success rates, max overlap distribution, plots) saved to `outputs/comparison/`
3. All three checkpoints pushed to my HuggingFace Hub account with proper model cards
4. Standalone SO-101 community-dataset inspection script producing a full schema report
   at `outputs/inspection/so101_<dataset>.md`
5. Test suite green (`pytest tests/`) — unit tests for data loading, the from-scratch
   policy, and eval harness
6. README with project description, setup, usage, architecture overview, results table
7. Clean repo on GitHub, gitignore correct, all commits with descriptive messages
8. Course skeleton (`course/00_introduction.md` minimum) — full course written after the arm arrives

**Non-goals:**
- Beating published PushT SOTA. Reproducing in the right ballpark is enough.
- Image-based MLP-BC. State-only is the cleaner baseline.
- Real hardware integration (the arm isn't here yet — that's a follow-up project).

---

## 2. Target User / Use Case

Primary user: me (adimunot21). Project doubles as portfolio piece showing the full
imitation-learning workflow on a public benchmark with reproducible results.

Secondary user: anyone evaluating my work — they should be able to clone, follow the
README, reproduce all three trained policies and the comparison numbers within a day.

---

## 3. Hardware Constraints & Strategy

| Constraint | Implication |
|---|---|
| GTX 1650, 4GB VRAM | Small batch sizes (16–32). AMP mandatory. Image policies need careful tuning. |
| 6-core CPU | Dataloader workers ≤6. Video decoding is CPU-heavy — preload to RAM where possible. |
| 32GB RAM | Comfortable. Can cache full PushT dataset in memory (~5GB). |
| 1TB SSD | Plenty for datasets + checkpoints. |

**VRAM fallback ladder** (revised after Phase 2 smoke test, 2026-05-09):

The original ladder assumed batch size was the binding lever. It isn't — for
LeRobot 0.5.1's default Diffusion Policy (263M params with `down_dims=(512,
1024, 2048)`), Adam state alone (3× model size) is ~3.1 GB, which overflows
the 1650's 3.63 GiB usable memory **before** any batch-dependent activations
are allocated. Model architecture is the binding lever for diffusion; batch
size is the lever for state-only / smaller policies.

Diffusion Policy (image obs):
1. `down_dims=(256, 512, 1024)`, batch=8, AMP on   ← original DP-paper recipe;
                                                     LeRobot's default is
                                                     larger than the paper.
2. `down_dims=(128, 256, 512)`, batch=8, AMP on
3. State-only (`pusht_keypoints` dataset), `n_obs_steps=2`, batch=64

ACT (Phase 3, ~52M params — tighter than DP at the U-Net but no Adam-state
blow-up):
1. Image obs, batch=8, AMP on
2. Image obs, batch=8, AMP on, gradient accumulation 2x (effective 16)
3. State-only

General techniques: enable AMP unconditionally, set
`PYTORCH_ALLOC_CONF=expandable_segments:True` in the wrapper script to reduce
fragmentation (recommended by the OOM error), prefer `--num_workers=2` over 4
if dataloader memory pressure is observed (we have 32 GB system RAM though,
so this is unlikely to bind).

---

## 4. System Architecture

### Component diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HuggingFace Hub                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ lerobot/pusht    │  │ adimunot21/      │  │ adimunot21/      │  │
│  │ (input dataset)  │  │ diffusion-pusht- │  │ act-pusht-       │  │
│  │                  │  │ lab (output)     │  │ lab (output)     │  │
│  └────────┬─────────┘  └────────▲─────────┘  └────────▲─────────┘  │
└───────────┼───────────────────────┼─────────────────────┼──────────┘
            │ download              │ upload              │ upload
            ▼                       │                     │
┌─────────────────────────────────────────────────────────────────────┐
│                          Local Training Env                         │
│                                                                     │
│   ┌──────────────────┐                                              │
│   │ LeRobotDataset   │──────► training loop ──────► checkpoint      │
│   │ (parquet + mp4)  │              │                  │            │
│   └──────────────────┘              ▼                  │            │
│                              ┌─────────────┐           │            │
│                              │   wandb     │           │            │
│                              │ (logging)   │           │            │
│                              └─────────────┘           │            │
│                                                        ▼            │
│                                              ┌─────────────────┐    │
│                                              │  gym-pusht env  │    │
│                                              │  (eval rollouts)│    │
│                                              └────────┬────────┘    │
│                                                       │             │
│                                                       ▼             │
│                                              ┌─────────────────┐    │
│                                              │ outputs/eval    │    │
│                                              │ (metrics, plots)│    │
│                                              └─────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Data flow per policy

```
  PushT episodes (HF Hub)
         │
         │ LeRobotDataset.__getitem__()
         ▼
  Sample = {
    observation.image: (T, 3, 96, 96),
    observation.state: (T, 2),
    action: (T, 2),
    timestamps, episode_index, ...
  }
         │
         │ policy.forward()
         ▼
  loss = MSE / Diffusion-loss / ACT-loss
         │
         │ optimizer.step()
         ▼
  checkpoint
         │
         │ lerobot-eval against gym-pusht
         ▼
  metrics = {success_rate, avg_max_reward, ...}
```

---

## 5. Technology Choices

| Library | Used For | Why this over alternatives |
|---|---|---|
| **PyTorch** (cu124 wheels) | All training | Industry standard. LeRobot is PyTorch-native. CUDA 12.4 wheels match driver 580 fine. JAX alternative exists but no LeRobot support. |
| **lerobot** | Dataset format, ACT/Diffusion policies, train/eval CLI | The whole point. Community standard for low-cost robotics. Uniform API across simulated and real arms (SO-101). |
| **gym-pusht** | Eval environment | LeRobot's standard eval target for PushT. Provides 95%-overlap success criterion used in published baselines. Alternative: roll a custom env. Don't. |
| **huggingface_hub** | Dataset/model versioning + hosting | Versioned, public, free. Same Hub the SO-101 community uses. Alternative: DVC + S3 = more infra, no community network effect. |
| **wandb (offline mode)** | Experiment tracking | LeRobot 0.5.1 has no TensorBoard fallback — disabling wandb produces zero plottable logs (verified Phase 2 smoke test 2026-05-10). Wandb's `WANDB_MODE=offline` writes runs to local `./wandb/` only, never contacts the cloud, so quota doesn't apply. Same rich logging as the published baselines (loss curves, GPU stats, system metrics). Can be synced to wandb.ai later if quota frees up via `wandb sync ./wandb/offline-run-*`. Alternative: tee stdout + custom plot script (crude, no GPU stats). |
| **pytest** | Testing | Standard. Alternative: unittest, but pytest's fixtures and parametrize are worth it. |
| **ruff** | Lint + format | Replaces black + isort + flake8 in one fast tool. Alternative: black-only is fine but ruff is strictly more capable now. |
| **python-dotenv** | Loading `.env` for HF token, wandb key | Standard. Alternative: bare `os.environ` reading, but dotenv handles missing files cleanly. |
| **PyYAML** | Config files | Standard. LeRobot configs are YAML/dataclass. |

---

## 6. Directory Structure

```
lerobot-pusht-lab/
├── CLAUDE.md                   # Persistent context for Claude Code
├── PLAN.md                     # This file
├── README.md                   # Written in Phase 8
├── .gitignore
├── .env.example                # Template — actual .env is gitignored
├── pyproject.toml              # Package metadata, ruff config
├── requirements.txt            # Pinned deps
├── environment.yml             # Conda env spec (python only)
│
├── src/lerobot_pusht_lab/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── inspection.py       # Reusable dataset inspection helpers
│   │   └── pusht_loader.py     # Thin wrapper around LeRobotDataset
│   ├── policies/
│   │   ├── __init__.py
│   │   └── mlp_bc.py           # Our from-scratch baseline
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py           # Wraps lerobot-eval + parses results
│   │   └── compare.py          # Cross-policy comparison + plotting
│   └── hub/
│       ├── __init__.py
│       └── upload.py           # Push checkpoints + write model cards
│
├── configs/
│   ├── diffusion_pusht.yaml    # Override file for lerobot-train
│   ├── act_pusht.yaml
│   └── mlp_bc_pusht.yaml       # Our own training script's config
│
├── scripts/                    # All scripts runnable as `python scripts/X.py`
│   ├── inspect_pusht.py        # Phase 1
│   ├── train_diffusion.sh      # Phase 2 — wraps lerobot-train
│   ├── train_act.sh            # Phase 3 — wraps lerobot-train
│   ├── train_mlp_bc.py         # Phase 4 — our training loop
│   ├── eval_all.py             # Phase 5 — runs eval across policies
│   ├── compare_policies.py     # Phase 5 — generates comparison report
│   ├── upload_to_hub.py        # Phase 6
│   └── inspect_so101.py        # Phase 7
│
├── tests/
│   ├── __init__.py
│   ├── test_data_inspection.py
│   ├── test_mlp_bc.py
│   └── test_eval_runner.py
│
├── notebooks/                  # Tracked, but exploratory only
│   └── 01_pusht_exploration.ipynb
│
├── data/                       # GITIGNORED — HF cache, downloaded videos
├── checkpoints/                # GITIGNORED — large weights
├── logs/                       # GITIGNORED — training logs
├── outputs/
│   ├── inspection/             # Tracked — small reference reports
│   ├── eval/                   # GITIGNORED — episode rollouts, big files
│   ├── comparison/             # Tracked — final comparison figures + tables
│   └── plots/                  # Tracked — publishable figures
└── course/                     # Written in final phase
    └── (markdown chapters)
```

---

## 7. Phase Breakdown

### Phase 0 — Environment Setup (1 evening, ~2h)

**Deliverable:** Working conda env, project structure, git initialized, GitHub repo
created and pushed, `python -c "import torch, lerobot; print(torch.cuda.is_available())"`
prints `True`.

**Checkpoints:**
- `nvidia-smi` shows GTX 1650 detected
- `torch.cuda.is_available()` returns `True`
- `from lerobot.datasets.lerobot_dataset import LeRobotDataset` succeeds
- `import gym_pusht` succeeds
- Repo visible on github.com under `adimunot21/lerobot-pusht-lab`

### Phase 1 — Dataset Inspection (half day, ~3h)

**Deliverable:** `scripts/inspect_pusht.py` runs, downloads `lerobot/pusht`, prints full
schema, value ranges, sample frames. Writes `outputs/inspection/lerobot_pusht.md` with
a full data contract. PLAN.md's "Data Contracts" section updated with verified facts.

**Checkpoints:**
- Schema printed: every field name, shape, dtype, value range
- 2–3 complete sample frames printed
- One image saved to `outputs/inspection/lerobot_pusht_sample.png`
- Episode boundaries verified — total episodes, frames per episode distribution

### Phase 2 — Diffusion Policy Training (~16–17 h, two-night job)

**Deliverable:** `lerobot-train` invocation via `scripts/train_diffusion.sh` with our
YAML config. Trained for **200K steps × batch 64 = 12.8M samples**, matching LeRobot's
published `lerobot/diffusion_pusht` recipe (which reports 65.4% success on 500 episodes).
Checkpoint saved to `checkpoints/diffusion_pusht/`. Tracked in wandb (offline mode).

**Initial attempt (2026-05-10) produced 5% success at 50K steps × batch 8 = 400K samples
(under-trained 32× vs published).** Revised to match LeRobot's recipe; only remaining
gap is U-Net width (256/512/1024 vs published 512/1024/2048 — needed to fit 4 GB VRAM).

**Target success rate:** ≥60% (within a few % of LeRobot's 65.4%).

**Checkpoints:**
- 50-step smoke test for each candidate batch size — verify VRAM fits ✅
- VRAM peak < 3.5 GB at chosen batch (1650 has 3.63 GB usable) ✅
- Full ~16h run completes without crashing
- Loss curve declines steadily and the cosine LR schedule plays out fully (lr → ~0 only at the very end)
- Final 20-episode in-training eval shows ≥40% success (Phase 5 does the proper 50-episode eval with CIs)

### Phase 3 — ACT Training (1 day + overnight)

**Deliverable:** Same shape as Phase 2 but for ACT. `scripts/train_act.sh`,
`checkpoints/act_pusht/`.

**Checkpoints:**
- Same as Phase 2 — smoke test, overnight run, loss curve, in-train eval

### Phase 4 — From-Scratch MLP-BC Baseline (1 day)

**Deliverable:** `src/lerobot_pusht_lab/policies/mlp_bc.py` (the model),
`scripts/train_mlp_bc.py` (the training loop), `configs/mlp_bc_pusht.yaml`. State-only
input (`observation.state`, 2-dim agent position), single-action output. MSE loss.
Trained on the SAME `lerobot/pusht` LeRobotDataset that the other policies use.

**Why state-only:** clean baseline, illustrates that PushT cannot be solved trivially
from state alone (needs visuomotor — that's the lesson). Fast to train (~30 min).

**Architecture:** 2-layer MLP, 256 hidden, ReLU, dropout 0.1, MSE loss, Adam.

**Checkpoints:**
- Forward pass smoke test: random tensor in, action-shaped tensor out
- Loss decreases on training data
- Trained checkpoint at `checkpoints/mlp_bc_pusht/`

### Phase 5 — Evaluation & Comparison (half day)

**Deliverable:** `scripts/eval_all.py` runs `lerobot-eval` against all three
checkpoints in `gym-pusht` (50 episodes each). `scripts/compare_policies.py` generates
`outputs/comparison/results.md` with success-rate table, max-overlap distribution
plots, sample rollout videos.

**Checkpoints:**
- All three eval runs complete
- Success rate table populated, no NaN/zero entries
- At least one sample rollout video per policy saved to `outputs/comparison/videos/`

### Phase 6 — HuggingFace Hub Publishing (half day)

**Deliverable:** Each checkpoint pushed to a HF Hub repo under `adimunot21/`. Each repo
has a model card explaining: training config, dataset used, eval results, how to load
and run.

**Checkpoints:**
- 3 model repos visible on huggingface.co/adimunot21
- Model cards render correctly
- One downstream "load and run inference" snippet tested (download from Hub → run one
  rollout in gym-pusht → confirms it works for someone else)

### Phase 7 — SO-101 Community Dataset Inspection (half day)

**Deliverable:** `scripts/inspect_so101.py` downloads a chosen SO-101 community
dataset, runs the same inspection routine as Phase 1, saves a report. Updates PLAN.md
with the SO-101 data contract.

**Why now:** dry-run of the inspection workflow on real SO-101 data so when the arm
arrives and I record my own dataset, I can re-run inspection and confirm it matches
expectations immediately.

**Dataset to inspect:** to be selected — search HF Hub for "so101" datasets in Phase 7.

### Phase 8 — Testing & Polish (half day)

**Deliverable:** Pytest suite green. README written. Dead code removed. Config values
checked for hardcoding. End-to-end demo walkthrough confirmed working from a fresh
clone.

**Tests to write:**
- `test_data_inspection.py`: schema reporter handles missing fields, edge cases
- `test_mlp_bc.py`: forward pass shape, gradient flow, deterministic with seed
- `test_eval_runner.py`: parses lerobot-eval output correctly

### Phase 9 — Course (after arm arrival, in parallel)

Course chapters per phase, written following the spec in CLAUDE.md / project guide.

---

## 8. Data Contracts

### 8.1 `lerobot/pusht` (VERIFIED Phase 1, 2026-05-09)

**Source:** https://huggingface.co/datasets/lerobot/pusht
**Format:** LeRobotDataset (parquet for tabular + mp4/av1 for video)
**Full report:** `outputs/inspection/lerobot_pusht.md`
**Inspector:** `scripts/inspect_pusht.py`

**Top-level magnitudes:**

| Property | Value |
|---|---|
| Total episodes | 206 |
| Total frames | 25 650 |
| FPS | 10 |
| Frames per episode | min=49, max=246, mean=124.5, median=122, std=35.7 |
| Tasks | 1 (`Push the T-shaped block onto the T-shaped target.`) |
| Robot type | unknown (sim) |

**Verified fields (output of `LeRobotDataset.__getitem__`):**

| Field | dtype | shape | range / value | meaning |
|---|---|---|---|---|
| `observation.image` | torch.float32 | (3, 96, 96) CHW | **[0, 1]** normalised | RGB frame. Storage layout HWC, dataloader returns CHW. |
| `observation.state` | torch.float32 | (2,) | **~[44, 450]** raw env coords | Agent (x, y) position, pixel-space on a ~512-unit board. Not normalised. |
| `action` | torch.float32 | (2,) | **~[44, 452]** raw env coords | Target (x, y) for agent. Same scale as state. |
| `episode_index` | torch.int64 | scalar `()` | [0, 205] | Episode this frame belongs to |
| `frame_index` | torch.int64 | scalar `()` | [0, 245] | Frame within episode |
| `timestamp` | torch.float32 | scalar `()` | [0, ~24.5] s | `frame_index / fps` |
| `next.reward` | torch.float32 | scalar `()` | [0, ~0.91] | Block-target overlap fraction (env-defined) |
| `next.done` | torch.bool | scalar `()` | True at episode terminal | |
| `next.success` | torch.bool | scalar `()` | Sparse — True only when overlap ≥95% threshold | NOT in original PLAN expectation |
| `task` | str | scalar | task instruction string | NOT in original PLAN expectation |
| `task_index` | torch.int64 | scalar `()` | always 0 (one task) | NOT in original PLAN expectation |
| `index` | torch.int64 | scalar `()` | global frame index across dataset | NOT in original PLAN expectation |

**Verified facts (resolved "to verify" questions):**
- ✅ Image normalisation: float32 in [0, 1] (NOT uint8). Confirmed by sampling 200 frames.
- ⚠️ State and action are **not** normalised — they're raw pixel coordinates ~[0, 512].
  Implications:
  - Diffusion Policy and ACT (LeRobot's implementations) apply their own input/output
    normalisation internally, so this is invisible in those phases.
  - **Phase 4 (from-scratch MLP-BC) must apply explicit normalisation** — fit
    mean/std on the training set, normalise inputs, denormalise outputs at inference.
- ⚠️ `next.success` is sparse: episodes can have `next.done=True` (terminated) but
  `next.success=False` (didn't reach 95% overlap). Use `next.success` for binary success
  rate, not `next.done`.
- ✅ Episode lengths vary widely (49–246 frames). Don't assume fixed episode length when
  building loaders / eval rollouts.

**Action chunking note:** Dataset is per-frame. Diffusion Policy and ACT consume action
chunks of length k via `LeRobotDataset(..., delta_timestamps={'action': [...]} )`. Inspect
in Phase 2 when configuring the policies.

### 8.2 SO-101 Community Dataset (VERIFIED Phase 7, 2026-05-10)

**Source:** https://huggingface.co/datasets/lerobot/svla_so101_pickplace
**Format:** LeRobotDataset (parquet + mp4)
**Full report:** `outputs/inspection/lerobot_svla_so101_pickplace.md`
**Inspector:** `scripts/inspect_so101.py` (reuses Phase 1's
`lerobot_pusht_lab.data.inspection` helpers — proves the abstraction is
dataset-agnostic)

**Top-level magnitudes:**

| Property | Value |
|---|---|
| Total episodes | 50 |
| Total frames | 11 939 |
| FPS | 30 (vs PushT's 10) |
| Frames per episode | min=183, max=306, mean=238.8, median=230 |
| Tasks | 1 ("pink lego brick into the transparent box") |
| Robot type | (community-recorded SO-100 follower per upstream README) |

**Key differences from PushT (verified):**

| Property | PushT | SO-101 |
|---|---|---|
| Cameras | 1 (`observation.image`) | **2** (`observation.images.up`, `observation.images.side`) |
| Image resolution | 96 × 96 | **480 × 640** (~33× pixels) |
| State dimensionality | 2 (xy position) | **6** (joint values, normalised ~[-100, 100]) |
| Action dimensionality | 2 | **6** |
| FPS | 10 | **30** |
| State/action units | env coords (pixels) | normalised joint values |

**Implications for SO-101 onboarding (when arm arrives):**

- Vision encoder must handle **2 camera streams** — LeRobot supports this via
  `use_separate_rgb_encoder_per_camera`.
- VRAM cost is much higher per frame: 480×640 vs 96×96 = ~33× pixels per camera × 2
  cameras = ~66× the visual data per frame. Expect to drop batch size dramatically
  vs PushT, possibly 2-4. Multi-camera image diffusion at full res is unlikely to
  fit on the 1650 — consider downsampling to 240×320 or using `pusht_keypoints`-
  style state-only baseline.
- Episode-length stats are different: SO-101 episodes are 5-10× longer in frames
  than PushT. Action chunking horizons may need tuning.

---

## 9. Risks & Open Questions

| Risk / Question | Mitigation / Resolution Plan |
|---|---|
| Image-based Diffusion Policy OOMs on 4GB VRAM | Fallback ladder defined in §3. Worst case use `pusht_keypoints` (state-only). |
| ACT's 52M params don't fit | Same fallback ladder. ACT does have a smaller config option. |
| LeRobot CLI flags / API change between install time and a later phase | Pin versions immediately after Phase 0 succeeds (`pip freeze > requirements.txt`). Don't `pip install -U` mid-project. |
| HF Hub rate limits when pushing checkpoints | Unlikely at this scale (3 models, ~MB-GB each). Use `huggingface_hub` upload with retry. |
| LeRobot 0.5.1 has no TensorBoard fallback — `wandb.enable=false` produces zero plottable logs | Use wandb in offline mode (`WANDB_MODE=offline` env + `wandb.enable=true`). Local-only, no quota cost. Verified during Phase 2 smoke test. |
| `gym-pusht 0.1.6` is incompatible with `pymunk ≥ 7` — uses `Space.add_collision_handler()` removed in pymunk 7.x | **Pinned `pymunk<7` (currently 6.11.1) in requirements.txt.** Discovered when extended Phase 2 smoke crashed at first eval rollout with `AttributeError: 'Space' object has no attribute 'add_collision_handler'`. `pip install gym-pusht` does not constrain the pymunk version, so a fresh install on a new machine will silently get pymunk 7.x and break eval. Re-pinned 2026-05-10. |
| Inspection script for SO-101 dataset fails because of an unexpected schema (e.g. dataset uses different LeRobotDataset version) | Inspection IS the discovery step. Failure here is informative — it's exactly what we're prepping for. |
| GTX 1650 training is slow vs 3080 baselines (LeRobot's published runs) | Reduce step count to "in the right ballpark" rather than reproducing full training budgets. Document the gap. |
| CUDA version mismatch (driver 580 says CUDA 13, PyTorch wheels are cu124) | This is fine. PyTorch wheels bundle their own CUDA runtime. Verify with `torch.cuda.is_available()` smoke test. |

---

## 10. Out of Scope (explicit non-goals)

- Real hardware integration. The arm isn't here. When it arrives, that's a separate project.
- Beating SOTA. We're matching ballpark numbers, not winning benchmarks.
- VLA / generalist models (SmolVLA, Pi0, GR00T). Out of scope for VRAM and time.
- Hyperparameter sweeps. One reasonable run per policy.
- Cross-task generalization. PushT only.
