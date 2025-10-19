#!/usr/bin/env python3
"""Plot how key streaming metrics vary with number of videos for each config (goalstep only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------------------------

DEFAULT_CONFIG_IDS: Tuple[str, ...] = (
    "base",
    "random_m",
    "random_2",
    "round_robin_m",
    "round_robin_2",
)
DEFAULT_NUM_VIDEOS: Tuple[int, ...] = (1, 3, 5, 8, 10, 15, 20)
DEFAULT_ITERATIONS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_BASE_DIR = Path("figures/goalstep")

METRIC_ORDER: Tuple[Tuple[str, str, str, str], ...] = (
    ("rebuffer_time", "Average Rebuffering Time (s)", "Rebuffering Time", "rebuffer"),
    ("ttft", "Average TTFT (s)", "TTFT", "ttft"),
    ("scheduling_score", "Final Scheduling Score", "Scheduling Score", "scheduling"),
    ("perplexity", "Average VLM Perplexity", "Perplexity", "perplexity"),
)


# ------------------------------------------------------------------------------
# Data loading and aggregation helpers
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate goalstep evaluation summaries across iterations and plot how "
            "metrics vary with the number of videos for each scheduling config."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory containing goalstep summaries (default: figures/goalstep).",
    )
    parser.add_argument(
        "--config-ids",
        nargs="+",
        default=list(DEFAULT_CONFIG_IDS),
        help="Config IDs to include in the comparison.",
    )
    parser.add_argument(
        "--num-videos",
        nargs="+",
        type=int,
        default=list(DEFAULT_NUM_VIDEOS),
        help="Number of videos (N) buckets to plot.",
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        default=list(DEFAULT_ITERATIONS),
        help="Iteration indices to average over.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the comparison figure (PNG).",
    )
    return parser.parse_args()


def safe_mean_std(values: Iterable[float]) -> Tuple[float, float]:
    """Return (mean, std) ignoring NaNs. Std is 0.0 for a single sample."""
    valid = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return float("nan"), float("nan")
    if len(valid) == 1:
        return valid[0], 0.0
    arr = np.asarray(valid, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def load_iteration_metrics(summary_path: Path) -> Dict[str, float]:
    """Extract metrics from a single overall_summary.json file."""
    with summary_path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)

    # Rebuffering time: average across conversations.
    buffer_states = summary.get("onthefly_buffer_data", {}) or {}
    rebuffer_values: List[float] = []
    for state in buffer_states.values():
        total = state.get("total_rebuffer")
        if total is None:
            rebuffer_series = state.get("rebuffer_values") or []
            total = rebuffer_series[-1] if rebuffer_series else 0.0
        rebuffer_values.append(float(total))
    rebuffer_time = float(np.mean(rebuffer_values)) if rebuffer_values else float("nan")

    # TTFT: average per conversation response latency when prompts exist.
    results = summary.get("results", []) or []
    ttft_samples: List[float] = []
    ppl_samples: List[float] = []
    for result in results:
        if result.get("generated_turns"):
            resp_time = result.get("response_time")
            if resp_time is not None:
                ttft_samples.append(float(resp_time))
        ppl_data = result.get("ppl_data") or {}
        ppl_values = ppl_data.get("gt_ppls_vlm_prefix_visual") or []
        ppl_samples.extend(float(p) for p in ppl_values if p is not None)
    ttft = float(np.mean(ttft_samples)) if ttft_samples else float("nan")
    perplexity = float(np.mean(ppl_samples)) if ppl_samples else float("nan")

    # Scheduling score: use the final cumulative score as overall value.
    scheduling = summary.get("scheduling_data") or {}
    score_series = scheduling.get("selected_score") or []
    scheduling_score = float(score_series[-1]) if score_series else float("nan")

    return {
        "rebuffer_time": rebuffer_time,
        "ttft": ttft,
        "scheduling_score": scheduling_score,
        "perplexity": perplexity,
    }


# ------------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------------

def configure_plot_style() -> None:
    """Set matplotlib defaults to a publication-friendly style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
        }
    )


def _scientific_colors() -> List[str]:
    """Return a color palette suited for scientific plots."""
    return [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]


def plot_grouped_bar(
    metric_key: str,
    title: str,
    y_label: str,
    slug: str,
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    num_videos: List[int],
    config_ids: List[str],
    output_path: Path,
) -> bool:
    """Grouped bar chart of metric vs number of videos."""
    configure_plot_style()
    colors = _scientific_colors()
    num_configs = len(config_ids)
    num_groups = len(num_videos)
    bar_width = min(0.8 / max(num_configs, 1), 0.18)
    x = np.arange(num_groups)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    has_data = False
    for idx, config_id in enumerate(config_ids):
        offset = (idx - (num_configs - 1) / 2) * bar_width
        stats_for_config = summary_stats.get(metric_key, {}).get(config_id, {})
        means = []
        stds = []
        for n in num_videos:
            metric_entry = stats_for_config.get(n, {})
            means.append(metric_entry.get("mean", float("nan")))
            stds.append(metric_entry.get("std", float("nan")))

        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue

        has_data = True
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            label=config_id,
        )

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in num_videos])
    ax.set_xlabel("Number of Videos")
    if has_data:
        ax.legend(frameon=False)
        fig.tight_layout(pad=0.6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return has_data


def plot_overall_bar(
    metric_key: str,
    title: str,
    y_label: str,
    slug: str,
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
) -> bool:
    """Bar chart summarizing mean/std aggregated across all video counts."""
    configure_plot_style()
    colors = _scientific_colors()
    means = []
    stds = []
    config_labels = []

    for idx, config_id in enumerate(config_ids):
        num_map = summary_stats.get(metric_key, {}).get(config_id, {})
        collected: List[float] = []
        for stats in num_map.values():
            collected.extend(stats.get("values", []))
        if not collected:
            continue
        mean, std = safe_mean_std(collected)
        means.append(mean)
        stds.append(std)
        config_labels.append(config_id)

    if not means:
        return False

    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=2.5,
        color=[colors[i % len(colors)] for i in range(len(means))],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title(f"{title} (All Videos)")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=20, ha="right")
    ax.set_xlabel("Config ID")
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_dir: Path = args.base_dir
    config_ids: List[str] = list(args.config_ids)
    num_videos: List[int] = list(args.num_videos)
    iterations: List[int] = list(args.iterations)

    metrics_storage: Dict[str, Dict[str, Dict[int, List[float]]]] = {
        metric: {cfg: {n: [] for n in num_videos} for cfg in config_ids}
        for metric, *_ in METRIC_ORDER
    }

    missing_paths: List[Path] = []
    for num in num_videos:
        for config_id in config_ids:
            for iteration in iterations:
                summary_path = (
                    base_dir
                    / f"N{num}"
                    / config_id
                    / f"I{iteration}"
                    / "overall_summary.json"
                )
                if not summary_path.exists():
                    missing_paths.append(summary_path)
                    continue
                metrics = load_iteration_metrics(summary_path)
                for metric_key in metrics_storage:
                    metrics_storage[metric_key][config_id][num].append(metrics[metric_key])

    if missing_paths:
        print("Warning: Missing summaries for the following settings:")
        for path in missing_paths:
            print(f"  - {path}")

    summary_stats: Dict[
        str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]
    ] = {metric: {} for metric, *_ in METRIC_ORDER}

    for metric_key, configs in metrics_storage.items():
        for config_id, num_map in configs.items():
            summary_stats[metric_key].setdefault(config_id, {})
            for num, values in num_map.items():
                mean, std = safe_mean_std(values)
                summary_stats[metric_key][config_id][num] = {
                    "mean": mean,
                    "std": std,
                    "values": list(values),
                }

    output_path = args.output
    plot_root = args.output if args.output is not None else base_dir / "goalstep_config_metrics"
    plot_root = Path(plot_root)
    if plot_root.suffix:
        plot_root_base = plot_root.with_suffix("")
    else:
        plot_root_base = plot_root

    metric_meta = {key: (y_label, title, slug) for key, y_label, title, slug in METRIC_ORDER}

    for metric_key, (y_label, title, slug) in metric_meta.items():
        per_video_path = plot_root_base.parent / f"{plot_root_base.name}_{slug}_per_video.pdf"
        overall_path = plot_root_base.parent / f"{plot_root_base.name}_{slug}_overall.pdf"
        per_video_saved = plot_grouped_bar(
            metric_key,
            title,
            y_label,
            slug,
            summary_stats,
            num_videos,
            config_ids,
            per_video_path,
        )
        if per_video_saved:
            print(f"Saved per-video bar plot to {per_video_path}")
        else:
            print(f"Skipped per-video bar plot for {metric_key}; no valid data.")

        overall_saved = plot_overall_bar(
            metric_key,
            title,
            y_label,
            slug,
            summary_stats,
            config_ids,
            overall_path,
        )
        if overall_saved:
            print(f"Saved aggregated bar plot to {overall_path}")
        else:
            print(f"Skipped aggregated bar plot for {metric_key}; no valid data.")

    csv_path = (plot_root_base.parent / f"{plot_root_base.name}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["metric", "config_id", "num_videos", "mean", "std", "iteration_values"]
        )
        for metric_key, configs in summary_stats.items():
            for config_id, num_map in configs.items():
                for num, stats in num_map.items():
                    writer.writerow(
                        [
                            metric_key,
                            config_id,
                            num,
                            stats["mean"],
                            stats["std"],
                            ";".join(f"{v:.6f}" for v in stats["values"]),
                        ]
                    )
    print(f"Wrote summary table to {csv_path}")


if __name__ == "__main__":
    main()
