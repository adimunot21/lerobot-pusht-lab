# HANDOVER — Get Paper-Level Diffusion Policy on PushT

**Written by:** Claude Sonnet 4.6, 2026-05-11, after a failed reproduction attempt
**For:** A new Claude Code session with a clean context
**User:** adimunot21 on GitHub, adimunot on HF Hub

The user is frustrated. The previous session (me) promised paper-level results from
training on LeRobot 0.5.1, but the resulting model only hit ~20% pc_success
(~35% by the standard max-overlap criterion), well below the 65% LeRobot benchmark
and far below the ~91% original-paper benchmark. This document is the briefing for
the new session to deliver on the user's stated goal: a Diffusion Policy on PushT
that hits **both** the LeRobot benchmark (~65%) **and** the original-paper recipe
(~80-91%).

---

## 1. The user's exact ask

> "I WANT YOU TO TRAIN A DIFFUSION POLICY ON PUSHt. EVEN IF IT MEANS REWRITING
> ALL THE CODE. I WANT BOTH, THE BENCHMARK AND THE WAY THE ORIGINAL PEOPLE DID IT."

So: **two Diffusion data points in the final Phase 5 comparison report**:

1. **LeRobot benchmark (~65%)**: use the published `lerobot/diffusion_pusht` checkpoint directly. Zero training, zero cost. Just evaluate it with our Phase 5 eval harness.
2. **Original-paper recipe (~80-91%)**: train using the ORIGINAL Chi et al. 2023 Diffusion Policy codebase. That codebase is what the paper used and is known to produce 91% on PushT. We integrate the resulting checkpoint into our Phase 5 comparison.

Plus the existing ACT and MLP-BC training that's already planned.

## 2. What's already done

Read these files in order before doing anything else:

1. **`PLAN.md`** — full 9-phase project plan, hardware, design decisions
2. **`CLAUDE.md`** — project-level instructions (code style, git workflow, data validation, etc.)
3. **`RUNPOD_SETUP.md`** — cloud workflow that the user is currently on (RunPod RTX 4090 24GB, Secure Cloud, Python 3.12 venv, ffmpeg + xvfb installed)
4. **`outputs/inspection/lerobot_pusht.md`** — verified data contract for the PushT dataset
5. **Git log** — `git log --oneline -25` shows phase-by-phase commit history

**Phases done (code complete, smoke-tested or already executed):**

| Phase | What | Status |
|---|---|---|
| 0 | Env setup, git, project scaffold | DONE (local + cloud pod) |
| 1 | `lerobot/pusht` dataset inspection (`scripts/inspect_pusht.py`) | DONE — data contract in `PLAN.md §8.1` |
| 2 | Diffusion training (LeRobot 0.5.1) | RAN, **only 20% pc_success** — the failure being remediated |
| 3 | ACT config + wrapper (`configs/act_pusht.yaml`, `scripts/train_act.sh`) | CODE READY, not yet trained |
| 4 | From-scratch MLP-BC (`src/lerobot_pusht_lab/policies/mlp_bc.py`, `scripts/train_mlp_bc.py`) | CODE READY + smoke green, not yet trained |
| 5 | Eval harness — `eval/runner.py`, `eval/compare.py`, `eval/lerobot_adapter.py`, 4 CLIs | CODE READY, not yet run on real checkpoints |
| 6 | HF Hub upload (`src/lerobot_pusht_lab/hub/upload.py`, `scripts/upload_to_hub.py`) | CODE READY |
| 7 | SO-101 community dataset inspection (`scripts/inspect_so101.py`) | DONE — data contract in `PLAN.md §8.2` |
| 8 | pytest suite (`tests/`) — 54 tests passing | DONE |

**Phase 2 failed checkpoint is preserved at:**
- `https://huggingface.co/adimunot/diffusion-pusht-200K-checkpoint` (private)
- Contains the trained `pretrained_model/` + run artifacts (eval videos, wandb log, training log) under `_run_artifacts/`
- Use this as the third Diffusion data point: "naïve LeRobot 0.5.1 reproduction attempt"

## 3. Why Phase 2 failed (so you don't repeat my mistakes)

The previous session's diagnosis:

1. **LeRobot 0.5.1 ≠ the version that produced the published `lerobot/diffusion_pusht` model card.** The training code, scheduler, and policy implementation have drifted. "Same hyperparameters" did not mean "same result."
2. **Gymnasium v1.0+ broke the `success` criterion.** See [LeRobot issue #470](https://github.com/huggingface/lerobot/issues/470). With the newer gymnasium, the env's `success` flag (sustained overlap) is stricter than the LeRobot model card's "max overlap ≥ 0.95" criterion. Our 20% pc_success would be ~35% under the max-overlap criterion. The actual eval methodology needs to match the model card's.
3. **20-episode eval has a 95% CI of ~30pp — too noisy to detect underperformance.** All Phase 5 evals should use **≥ 50 episodes**, ideally 100.

## 4. Concrete plan (this is the path to success)

### Step A — Resume the existing RunPod pod (~2 min)

The user has a Stopped pod with everything pre-installed (Python 3.12, ffmpeg, xvfb, our venv at `/workspace/venv`, our repo at `/workspace/lerobot-pusht-lab`). User will SSH back in and `source /workspace/venv/bin/activate`. If the pod was Terminated, see `RUNPOD_SETUP.md` to rebuild from scratch.

### Step B — Get the LeRobot benchmark data point WITHOUT training (~30 min, ~$0.35)

```bash
# Download the official checkpoint
hf download lerobot/diffusion_pusht --local-dir checkpoints/lerobot_official_diffusion_pusht

# Eval it with our Phase 5 harness — use 100 episodes for tight CIs.
# The success criterion question is REAL: report both the env's pc_success AND
# the max-overlap criterion. eval/runner.py already does max-overlap; pc_success
# would need to be added.
xvfb-run python scripts/eval_lerobot.py \
    checkpoints/lerobot_official_diffusion_pusht \
    --policy-name diffusion_lerobot_official \
    --n-episodes 100 \
    --output-dir outputs/eval/diffusion_lerobot_official
```

**Expected:** success_rate ~55-70% (LeRobot card says 65.4%; our recipe should land near that).

**If you get << 65%:** the bug is in our `LeRobotPolicyAdapter` or the env config we're using, not in the model. Compare to `lerobot-eval` CLI output. Run:
```bash
xvfb-run lerobot-eval --config_path=... # see lerobot/scripts/lerobot_eval.py
```
to get the OFFICIAL evaluation — that's the ground truth.

### Step C — Train the paper recipe using the ORIGINAL Diffusion Policy codebase (~12-24h, ~$8-16)

**THIS IS THE CRITICAL DECISION.** Do NOT try again with LeRobot 0.5.1. That recipe is broken for our purposes (Phase 2 proved it). Use the original Chi et al. 2023 repo, which is the gold standard for paper-level results on PushT.

```bash
# Original Diffusion Policy repo from the paper authors
git clone https://github.com/columbia-ai-robotics/diffusion_policy.git /workspace/dp_original
cd /workspace/dp_original

# They have detailed install instructions in their README; follow exactly.
# Key facts:
#   - Uses Hydra + YAML configs (not LeRobot's draccus)
#   - Uses zarr dataset format (NOT LeRobotDataset)
#   - Has a pre-converted PushT zarr dataset they distribute
#   - PushT image config: config/task/pusht_image.yaml
#   - Training command: python train.py --config-name=train_diffusion_unet_image_workspace task=pusht_image
#   - Default training: 3050 epochs (overkill for portfolio; 1000 epochs likely sufficient)
#   - Their PushT-image checkpoint hits ~91% per the paper
```

**Critical setup notes** (the previous session's pain points):

- The original repo may have older dependencies. Use a SEPARATE Python venv (or conda env) for it, so it doesn't fight with our LeRobot venv. The dataset format is different anyway.
- Download their pre-converted PushT zarr from the link in their README — DO NOT try to convert lerobot/pusht to zarr yourself; it's a different episode split.
- They report wall-clock numbers on RTX 3090. On RTX 4090 expect ~1.5-2× faster. Budget 12-24h for 1000 epochs.
- Save checkpoints every 50 epochs. Plot the eval curve from their own eval (they have one built-in). Don't blindly train to 3050 epochs.
- Their eval ALSO uses gym-pusht under the hood — but their version might be different. Verify before celebrating numbers.

**After training:** convert the resulting checkpoint to a format our Phase 5 harness can load. Two options:

1. Write a thin wrapper class implementing our `PolicyAdapter` protocol (`src/lerobot_pusht_lab/eval/runner.py:PolicyAdapter`) that loads the original-repo checkpoint and translates obs/action. This is the cleanest path.
2. Or: use the original repo's eval directly and just record the success_rate / avg_max_reward numbers in our comparison report.

Option 1 is preferred for an apples-to-apples eval. Option 2 is acceptable if there's no time.

### Step D — Train ACT and MLP-BC, run full Phase 5 eval (~4-5h, ~$3)

After the original Diffusion Policy is trained:

```bash
# ACT — config is ready at configs/act_pusht.yaml
xvfb-run ./scripts/train_act.sh 2>&1 | tee logs/act_$(date +%Y%m%d_%H%M).log

# MLP-BC — trivial, 30 seconds
python scripts/train_mlp_bc.py

# Phase 5 eval all (uses our runner — 50 episodes per policy with deterministic seeds)
# Modify scripts/eval_all.py to include 4 policies now:
#   - diffusion_lerobot_official (downloaded checkpoint)
#   - diffusion_paper_recipe (from original repo training)
#   - diffusion_lerobot_0_5_1 (our failed reproduction, from HF Hub)
#   - act_pusht
#   - mlp_bc_pusht
xvfb-run python scripts/eval_all.py --n-episodes 50
```

### Step E — Publish + write README (~30 min)

```bash
python scripts/upload_to_hub.py --dry-run     # verify cards
python scripts/upload_to_hub.py                # publish to adimunot/<policy>-pusht-lab

# Then write README.md at repo root. Phase 8 deliverable. Reference the
# outputs/comparison/comparison.md and embed the success-rate plot.
```

## 5. The eval-criterion bug — handle this explicitly

LeRobot issue #470 documents that gymnasium ≥ 1.0 changed the env's `success` flag, breaking comparison with the published 65% number. **Your Phase 5 report MUST report both:**

- `pc_success` (env's strict criterion — what `lerobot-eval` returns)
- Episodes with `max_reward ≥ 0.95` (what the LeRobot model card defines as success)

In `src/lerobot_pusht_lab/eval/runner.py`, `EpisodeMetrics.success` is computed as `max_reward >= success_threshold`. Good. But ADD a separate field `env_terminated_successful` that captures the env's own success signal too. The comparison report should show both columns.

## 6. Budget & timeline estimate

| Step | Time | Cost on RTX 4090 Secure ($0.69/hr) |
|---|---|---|
| A: Resume pod | 2 min | $0.02 |
| B: Eval LeRobot official checkpoint | 30 min | $0.35 |
| C: Train paper recipe (original repo, 1000 epochs) | 12-24h | $8-17 |
| D: ACT + MLP-BC + eval all | 4-5h | $3 |
| E: Publish + README | 30 min | $0.35 |
| **Total** | **~18-30h** | **~$12-21** |

The user has spent ~$3 so far. **Budget ~$25 total for the full success.** If user has put less in their RunPod account, ask them to top up before starting Step C.

## 7. Definition of done — testable success criteria

You are done when ALL of these are true:

1. **`outputs/comparison/comparison.md` exists** and includes at least 4 policy rows (3 Diffusion variants + ACT + MLP-BC ideally; minimum 3 if ACT/MLP-BC can't run)
2. **The Diffusion paper-recipe row reports success_rate ≥ 60%** by the max-overlap criterion. Below 60% means the original-repo training failed and needs debugging.
3. **The Diffusion LeRobot-official row reports 55-70%** by the max-overlap criterion. Below 50% means our eval harness has a bug.
4. **Each row reports BOTH `pc_success` AND `max_overlap_success` columns** so the eval-criterion ambiguity is transparent.
5. **All 3 (or 4) policies are pushed to HF Hub** with model cards under `adimunot/`
6. **README.md exists at repo root** with results table, project description, setup instructions
7. **`pytest tests/` all passes**

## 8. What NOT to do (lessons from the previous session)

1. **Don't promise specific success rates from training.** ML is stochastic. Promise the process; report the result.
2. **Don't trust `pc_success` alone.** Always also compute success by max_reward ≥ 0.95. Report both.
3. **Don't use small eval episode counts.** 20 is not enough. Use 50-100.
4. **Don't re-run failed approaches "just to try."** If LeRobot 0.5.1 produced 20%, don't try LeRobot 0.5.1 again with minor tweaks. Switch tools (use the original repo).
5. **Don't conflate "smoke test passed" with "training will hit benchmark."** Smoke only verifies the pipeline runs. Convergence is a separate question.
6. **Always check the user's HuggingFace username before pushing.** It's `adimunot`, not `adimunot21`. GitHub username is `adimunot21`.
7. **Pin gym-pusht to pymunk<7** (already in `requirements.txt`) — pymunk 7.x removed `Space.add_collision_handler()` which gym-pusht uses.
8. **xvfb is REQUIRED for any gym-pusht eval on the cloud pod.** Wrap with `xvfb-run -s "-screen 0 1024x768x24"`.

## 9. Communication style with the user

The user is intelligent, frustrated, and budget-conscious. Be:

- **Direct.** No hedging. State expected outcomes with realistic ranges, not single numbers.
- **Honest about risk.** When a step might not work, say so up front.
- **Transparent about cost.** Quote estimates BEFORE running anything that costs > $1.
- **Concise.** They've been at this for days. They want results, not essays.
- **No emoji.** Per CLAUDE.md.

When something goes wrong: own it, diagnose it, recommend the fix, ask permission to proceed if the fix costs money.

---

## PROMPT FOR NEW SESSION

(Paste everything below this line into a new Claude Code session, in the
`~/projects/lerobot-pusht-lab` directory.)

```
I'm continuing a project where a previous Claude session attempted to train
Diffusion Policy on PushT but only reached 20% pc_success (~35% by the
max-overlap criterion) — well below the 65% LeRobot benchmark and ~91% paper
benchmark.

Read these files IN ORDER before suggesting anything:

1. HANDOVER.md (the full briefing — most important; read this first)
2. PLAN.md (project plan, 9 phases, hardware, design decisions)
3. CLAUDE.md (project-level instructions — code style, git workflow)
4. RUNPOD_SETUP.md (cloud workflow I'm using)
5. The most recent git log: `git log --oneline -25`

After reading, summarize back to me in 5-7 bullets:
- What's already been built and what state it's in
- Why the previous training run failed (root causes, not just symptoms)
- The two Diffusion data points I want (LeRobot benchmark + original paper)
- The recommended approach for getting both
- Estimated cost and wall-clock time
- The eval-criterion bug we need to handle transparently
- The success criteria you'll use to know we're done

Then propose Step B (eval the LeRobot official checkpoint) with the exact
commands you'd run, and wait for my go-ahead before executing.

Constraints:
- I'm on RunPod RTX 4090 24GB Secure Cloud ($0.69/hr). I have a Stopped pod
  with everything pre-installed at /workspace/venv and /workspace/lerobot-pusht-lab.
- My HF username is `adimunot` (NOT `adimunot21` which is my GitHub).
- Budget: spent ~$3 so far, willing to spend up to $25 total.
- I want this to actually work this time. Be honest about risk.
- The failed training is preserved at:
  https://huggingface.co/adimunot/diffusion-pusht-200K-checkpoint
  (private, my HF token has access)

Do not start any GPU work until I confirm. Read HANDOVER.md first.
```

End of handover.
