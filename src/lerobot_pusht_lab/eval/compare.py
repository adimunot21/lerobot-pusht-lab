"""Phase 5 cross-policy comparison: aggregate EvalMetrics → markdown report + plots.

Inputs: a list of ``EvalMetrics`` (from ``runner.py``), one per policy.
Outputs:
  - ``comparison.md`` — headline table + per-metric breakdown + interpretation
  - ``success_rate_with_ci.png`` — bar chart with Wilson 95% CIs
  - ``max_reward_distribution.png`` — boxplot/violin per policy

Design note: this module is presentation-only. It never re-runs evaluation;
it only visualises results that ``runner.py`` produced. This keeps the slow
part (env rollouts) decoupled from the fast part (plotting), so we can iterate
on the report without re-running 50-episode evals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — we save PNGs, never show()
import matplotlib.pyplot as plt
import numpy as np

from lerobot_pusht_lab.eval.runner import EvalMetrics, EpisodeMetrics

logger = logging.getLogger(__name__)


def load_eval_metrics(path: str | Path) -> EvalMetrics:
    """Restore an EvalMetrics from the JSON written by ``EvalMetrics.save``."""
    data = json.loads(Path(path).read_text())
    eps = [EpisodeMetrics(**e) for e in data.pop("episodes")]
    return EvalMetrics(**data, episodes=eps)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _format_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_markdown_report(
    metrics: list[EvalMetrics],
    title: str = "PushT — Policy Comparison",
    notes: str = "",
) -> str:
    """Render the headline comparison table + per-policy details in markdown.

    Order: by descending success_rate (best policy on top).
    """
    metrics = sorted(metrics, key=lambda m: m.success_rate, reverse=True)
    lines: list[str] = []
    lines.append(f"# {title}\n")
    if notes:
        lines.append(notes + "\n")

    # Headline table
    lines.append("## Headline results\n")
    lines.append(
        "| Rank | Policy | Success rate (95% CI) | avg max_reward | "
        "avg episode length | inference s/step |"
    )
    lines.append("|---|---|---|---|---|---|")
    for i, m in enumerate(metrics, 1):
        ci = f"{_format_pct(m.success_rate)} [{_format_pct(m.success_ci_low)}, {_format_pct(m.success_ci_high)}]"
        steps_per_ep = m.avg_episode_length
        s_per_step = m.avg_inference_time_s / steps_per_ep if steps_per_ep else float("nan")
        lines.append(
            f"| {i} | `{m.policy_name}` | {ci} | {m.avg_max_reward:.3f} | "
            f"{steps_per_ep:.0f} | {s_per_step:.4f} |"
        )
    lines.append("")

    # Per-policy breakdown
    lines.append("## Per-policy details\n")
    for m in metrics:
        lines.append(f"### `{m.policy_name}`\n")
        lines.append(f"- **Episodes evaluated:** {m.n_episodes}")
        lines.append(f"- **Success threshold:** max_reward ≥ {m.success_threshold:.2f}")
        lines.append(
            f"- **Success rate:** {_format_pct(m.success_rate)} "
            f"(Wilson 95% CI: [{_format_pct(m.success_ci_low)}, {_format_pct(m.success_ci_high)}])"
        )
        lines.append(f"- **avg_max_reward:** {m.avg_max_reward:.4f}")
        lines.append(f"- **avg_sum_reward:** {m.avg_sum_reward:.4f}")
        lines.append(f"- **avg_episode_length:** {m.avg_episode_length:.1f} steps")
        lines.append(f"- **avg_inference_time/episode:** {m.avg_inference_time_s:.3f} s")
        lines.append(f"- **eval wall time:** {m.wall_time_s:.1f} s")
        # Distribution callouts
        successes = [e.episode_index for e in m.episodes if e.success]
        max_rewards = sorted((e.max_reward for e in m.episodes), reverse=True)
        lines.append(f"- **Successful episode indices:** {successes if successes else '(none)'}")
        if max_rewards:
            lines.append(
                f"- **max_reward percentiles:** "
                f"min={max_rewards[-1]:.3f}, "
                f"25th={np.percentile(max_rewards, 25):.3f}, "
                f"50th={np.percentile(max_rewards, 50):.3f}, "
                f"75th={np.percentile(max_rewards, 75):.3f}, "
                f"max={max_rewards[0]:.3f}"
            )
        lines.append("")

    # Interpretation hooks — auto-generated observations
    lines.append("## Auto-generated observations\n")
    if len(metrics) >= 2:
        best, *rest = metrics
        gap = best.success_rate - rest[0].success_rate
        lines.append(
            f"- Best policy: **`{best.policy_name}`** ({_format_pct(best.success_rate)}). "
            f"Gap to next-best: {gap*100:+.1f} pp."
        )
    near_misses = [
        m for m in metrics
        if m.success_rate < 0.5 and m.avg_max_reward > 0.7
    ]
    if near_misses:
        lines.append(
            "- Near-miss policies (avg_max_reward > 0.7 but success_rate < 50%): "
            + ", ".join(f"`{m.policy_name}`" for m in near_misses)
            + ". Pattern indicates the policy approaches the target consistently "
            "but doesn't quite cross the 95%-overlap success threshold — typical "
            "of under-trained or capacity-limited models."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_success_rate_with_ci(metrics: list[EvalMetrics], path: str | Path) -> Path:
    """Bar chart: success rate per policy, error bars = Wilson 95% CI."""
    metrics = sorted(metrics, key=lambda m: m.success_rate, reverse=True)
    names = [m.policy_name for m in metrics]
    rates = [100 * m.success_rate for m in metrics]
    err_low = [100 * (m.success_rate - m.success_ci_low) for m in metrics]
    err_high = [100 * (m.success_ci_high - m.success_rate) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, rates, yerr=[err_low, err_high], capsize=6, color="#3a7ca5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"PushT success rate by policy (N={metrics[0].n_episodes} per policy, Wilson 95% CI)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("saved plot: %s", path)
    return path


def plot_max_reward_distribution(metrics: list[EvalMetrics], path: str | Path) -> Path:
    """Boxplot of per-episode max_reward, one box per policy.

    Why max_reward distribution (not just the mean): on PushT a policy can have
    avg_max_reward ≈ 0.7 (close to the goal *on average*) yet 0% success
    (because the threshold is 0.95 and *no* episode crosses it). The
    distribution shows whether the policy is consistently near-but-not-at the
    goal (under-converged) vs sometimes-great-sometimes-bad (high variance).
    """
    metrics = sorted(metrics, key=lambda m: m.avg_max_reward, reverse=True)
    data = [[e.max_reward for e in m.episodes] for m in metrics]
    names = [m.policy_name for m in metrics]
    threshold = metrics[0].success_threshold if metrics else 0.95

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, tick_labels=names, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a8d8ea")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"success threshold ({threshold:.2f})")
    ax.set_ylabel("Per-episode max reward")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-episode max_reward distribution by policy")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("saved plot: %s", path)
    return path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class ComparisonOutputs:
    report_md: Path
    success_plot: Path
    distribution_plot: Path


def write_full_comparison(
    metrics: list[EvalMetrics],
    output_dir: str | Path,
    title: str = "PushT — Policy Comparison",
    notes: str = "",
) -> ComparisonOutputs:
    """Write report + both plots to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_markdown_report(metrics, title=title, notes=notes)
    report_path = output_dir / "comparison.md"
    report_path.write_text(report)
    logger.info("saved report: %s", report_path)

    success_plot = plot_success_rate_with_ci(metrics, output_dir / "success_rate_with_ci.png")
    distribution_plot = plot_max_reward_distribution(metrics, output_dir / "max_reward_distribution.png")
    return ComparisonOutputs(
        report_md=report_path,
        success_plot=success_plot,
        distribution_plot=distribution_plot,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """Smoke: synthesise three fake EvalMetrics, render full report, sanity-check."""
    import logging
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    def _fake_metrics(name: str, n: int, success_rate: float, mean_max: float) -> EvalMetrics:
        rng = np.random.default_rng(hash(name) & 0xFFFF)
        # Synthesise per-episode max_reward distribution centred on mean_max,
        # with successes set to >= 0.95.
        n_success = int(round(n * success_rate))
        max_rewards = list(rng.uniform(0.95, 1.0, size=n_success)) + \
                      list(rng.normal(mean_max, 0.1, size=n - n_success).clip(0.0, 0.94))
        rng.shuffle(max_rewards)
        episodes = [
            EpisodeMetrics(
                episode_index=i,
                seed=100000 + i,
                success=(mr >= 0.95),
                max_reward=float(mr),
                sum_reward=float(mr * 100),
                n_steps=200,
                inference_time_s=0.1,
            )
            for i, mr in enumerate(max_rewards)
        ]
        from lerobot_pusht_lab.eval.runner import wilson_score_interval
        successes = sum(e.success for e in episodes)
        ci_low, ci_high = wilson_score_interval(successes, n)
        return EvalMetrics(
            policy_name=name,
            n_episodes=n,
            success_rate=successes / n,
            success_ci_low=ci_low,
            success_ci_high=ci_high,
            avg_max_reward=float(np.mean([e.max_reward for e in episodes])),
            avg_sum_reward=float(np.mean([e.sum_reward for e in episodes])),
            avg_episode_length=200.0,
            avg_inference_time_s=0.1,
            success_threshold=0.95,
            wall_time_s=20.0,
            episodes=episodes,
        )

    metrics = [
        _fake_metrics("diffusion_pusht", 50, success_rate=0.62, mean_max=0.85),
        _fake_metrics("act_pusht",       50, success_rate=0.36, mean_max=0.70),
        _fake_metrics("mlp_bc_pusht",    50, success_rate=0.04, mean_max=0.25),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        outs = write_full_comparison(
            metrics, tmp,
            title="SMOKE — synthetic data, not real eval",
            notes="This is a smoke test of the comparison report machinery.",
        )
        report = Path(outs.report_md).read_text()
        assert "diffusion_pusht" in report
        assert "Wilson" in report
        assert outs.success_plot.exists() and outs.success_plot.stat().st_size > 1000
        assert outs.distribution_plot.exists() and outs.distribution_plot.stat().st_size > 1000

    print("smoke test PASSED")
    print("---")
    print(report[:1500])
    print("...")
