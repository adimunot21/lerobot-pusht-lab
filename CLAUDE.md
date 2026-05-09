# LeRobot PushT Lab — Project Memory

Sim-based imitation learning lab to get fluent with LeRobot, HuggingFace Hub, and BC
workflows BEFORE an SO-101 arm arrives. Train + compare Diffusion Policy, ACT, and a
from-scratch MLP-BC baseline on the canonical `lerobot/pusht` task. Same CLI / dataset
format / Hub flow as the SO-101.

**Always read `PLAN.md` for current phase, architecture, data contracts, and status
before suggesting code or next steps.**

## Hardware

- Lenovo Legion Y540, Intel i7-9750H (6c/12t), 32GB DDR4
- NVIDIA GTX 1650, **4GB VRAM** (binding constraint), driver 580.126.09
- 1TB SSD, Ubuntu 24.04
- VRAM strategy: small batches (8–32), AMP everywhere, gradient accumulation if needed.
  If image-based training OOMs, fall back to `pusht_keypoints` (state-only) without me
  having to ask.

## Dev environment (already set up)

- zsh + Oh My Zsh, miniforge (conda), base auto-activation disabled
- Tools: git, wget, curl, cmake, tree, htop, tmux, neovim, jq, ffmpeg 6.1.1
- Git SSH (ed25519) linked to GitHub user `adimunot21`
- Docker, NVIDIA driver 580, VS Code (Python, Jupyter, Pylance, Docker, Remote-SSH, GitLens)

Don't re-install any of the above.

## Project Philosophy

Production software, not a learning exercise. Use the best library for each job — but
**never treat it as a black box**. When calling a high-level API:
- Comment / explain what's happening under the hood
- Note what an equivalent manual implementation would roughly look like
- Justify every config parameter (why this value? what changes if it's different?)
- One-or-two-sentence callout for the alternative library and when you'd reach for it

Production standards apply: error handling, input validation, logging, configuration
management, type hints, clean APIs, reproducibility.

## Code Style (non-negotiable)

- Type hints on function signatures and complex data structures (judgement on locals)
- `logging` module, not `print`, except for demo/CLI output
- Configuration in ONE place — `configs/*.yaml` or a `settings` module. Never scattered.
- Error handling for expected failure modes — catch, log, recover or fail with a clear message
- Every major module has a `if __name__ == "__main__":` block doing a smoke test, demo,
  or CLI. Modules must be runnable standalone.
- Separate concerns: data loading, processing, business logic, presentation in separate files
- Set random seeds for any training run; save the config alongside outputs
- Descriptive variable names; clarity over brevity

## Data Validation Rules (MANDATORY — non-negotiable)

Before using ANY external dataset, API, or pretrained model:

1. Write a standalone inspection script that:
   - Downloads / loads a small sample
   - Prints every field name + dtype + shape + value range (min/max/mean/std)
   - Prints 2–3 complete raw samples
   - Saves output to `outputs/inspection/<dataset_name>.md` as a permanent reference
2. **Document the data contract** in `PLAN.md` before writing any processing code:
   source, fields with types/meanings/ranges, format, rate limits, known quirks
3. End-to-end sanity check: pass one real sample through every pipeline stage,
   printing shapes/types/values at each stage, BEFORE any heavy processing
4. Never assume data semantics. If a field is called "score", inspect it — don't
   assume 0–1 or 0–100. State findings explicitly before proceeding.

## Git / GitHub Rules

- Commit + push after EVERY phase with a descriptive message
- **GitHub repo must be created on github.com FIRST** before pushing. I have forgotten
  this before — always remind me before the first push.
- `requirements.txt` and `environment.yml` always kept up to date
- Notebooks tracked under `notebooks/`
- Never commit secrets — `.env` is gitignored, `.env.example` is committed
- Large files (model weights, datasets, eval outputs) are gitignored
- GitHub username: `adimunot21`

## Known Issues to Watch

- Colab may have different PyTorch versions than local. Test API compatibility before
  porting code (e.g. `total_mem` vs `total_global_mem` attribute renames)
- Files in `.gitignore` won't be on GitHub. When cloning into Colab, remind me to
  re-download data files (PushT dataset, etc.)
- LeRobot is fast-moving — pin versions in `requirements.txt` and don't auto-upgrade
  mid-project. After Phase 0 install succeeds, freeze versions immediately.

## Communication Style

- Direct. No hedging. No unnecessary caveats.
- "Why" before "how". Reasoning first, then the code.
- When something breaks: identify the specific issue, explain the root cause, give the
  minimal fix. Do NOT rewrite whole files.
- Tradeoffs: state both sides briefly, recommend with justification.
- After each phase: brief summary of what was built, what it does, how to verify.
- For long training runs: suggest what to read or check while waiting.

## Workflow Rules

- ALWAYS plan in `PLAN.md` first, then follow it. If something needs to change, update
  `PLAN.md` first, then implement.
- If multiple tasks are queued and we get stuck on Task 1: after solving it, **confirm
  status of Tasks 2 and 3 explicitly** before moving to Task 4. Do not skip steps because
  they were mentioned earlier.
- Long training runs go in tmux outside Claude Code. Inside Claude Code, parse logs and
  diagnose — don't tail loops in the chat.
- For each phase: read PLAN.md → propose what we're building → wait for approval → write
  code → run smoke test → commit + push.

## Project Structure (see PLAN.md for full tree)

```
src/lerobot_pusht_lab/  # importable package
configs/                # YAML training configs
scripts/                # entry-point scripts (inspect, train, eval, compare)
tests/                  # pytest tests
notebooks/              # exploration only, not production logic
data/                   # gitignored — datasets land here
checkpoints/            # gitignored — model weights
outputs/                # eval results, plots, inspection reports
logs/                   # gitignored — training logs
course/                 # written in final phase
```
