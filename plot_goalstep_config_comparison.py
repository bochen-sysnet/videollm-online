#!/usr/bin/env python3
"""Plot how key streaming metrics vary with number of videos across data sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
DEFAULT_BASE_DIR = Path("figures")
BASE_CONFIG_ID = "base"

METRIC_ORDER: Tuple[Tuple[str, str, str, str], ...] = (
    ("rebuffer_time", "Average Rebuffering Time (s)", "Rebuffering Time", "rebuffer"),
    ("ttft", "Average TTFT (s)", "TTFT", "ttft"),
    ("scheduling_score", "Final Scheduling Score", "Scheduling Score", "scheduling"),
    ("perplexity", "Average VLM Perplexity", "Perplexity", "perplexity"),
)

DELAY_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("processing_delay", "Processing Delay (s)", "Processing Delay"),
    ("queuing_delay", "Queuing Delay (s)", "Queuing Delay"),
    ("network_delay", "Networking Delay (s)", "Networking Delay"),
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
        "--data-sources",
        nargs="+",
        default=["goalstep"],
        choices=["goalstep", "narration"],
        help="Data sources to aggregate (default: goalstep).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root directory containing data source folders (default: figures).",
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
        default=None,
        help="Number of videos (N) buckets to plot; auto-detected if omitted.",
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
    processing_delays: List[float] = []
    queuing_delays: List[float] = []
    networking_delays: List[float] = []
    for state in buffer_states.values():
        # Some states store metrics directly, others under nested 'listening'
        candidate = state.get("listening") if isinstance(state, dict) and "listening" in state else state
        if not isinstance(candidate, dict):
            continue

        total = candidate.get("total_rebuffer")
        if total is None:
            rebuffer_series = candidate.get("rebuffer_values") or []
            total = rebuffer_series[-1] if rebuffer_series else 0.0
        rebuffer_values.append(float(total))

        if candidate.get("processing_delays"):
            processing_delays.append(float(candidate["processing_delays"][-1]))
        elif candidate.get("total_processing_delay") is not None:
            processing_delays.append(float(candidate["total_processing_delay"]))

        if candidate.get("queuing_delays"):
            queuing_delays.append(float(candidate["queuing_delays"][-1]))
        elif candidate.get("total_queuing_delay") is not None:
            queuing_delays.append(float(candidate["total_queuing_delay"]))

        if candidate.get("networking_delays"):
            networking_delays.append(float(candidate["networking_delays"][-1]))
        elif candidate.get("total_networking_delay") is not None:
            networking_delays.append(float(candidate["total_networking_delay"]))
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

    processing_delay = float(np.mean(processing_delays)) if processing_delays else float("nan")
    queuing_delay = float(np.mean(queuing_delays)) if queuing_delays else float("nan")
    networking_delay = float(np.mean(networking_delays)) if networking_delays else float("nan")

    return {
        "rebuffer_time": rebuffer_time,
        "ttft": ttft,
        "scheduling_score": scheduling_score,
        "perplexity": perplexity,
        "processing_delay": processing_delay,
        "queuing_delay": queuing_delay,
        "network_delay": networking_delay,
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


def plot_base_delay_trends(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    num_videos: List[int],
    output_path: Path,
) -> bool:
    """Plot base config delay metrics vs number of videos."""
    base_stats = {
        metric: summary_stats.get(metric, {}).get(BASE_CONFIG_ID, {}) for metric, *_ in DELAY_METRICS
    }
    if not any(base_stats.values()):
        return False

    configure_plot_style()
    colors = _scientific_colors()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    has_data = False

    markers = ["o", "s", "^"]
    for idx, (metric_key, _, title) in enumerate(DELAY_METRICS):
        metric_data = base_stats.get(metric_key, {})
        means = []
        stds = []
        for n in num_videos:
            entry = metric_data.get(n, {})
            means.append(entry.get("mean", float("nan")))
            stds.append(entry.get("std", float("nan")))
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        has_data = True
        ax.errorbar(
            num_videos,
            means_arr,
            yerr=stds_arr,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=1.6,
            markersize=4,
            capsize=3,
            label=title,
        )

    if not has_data:
        plt.close(fig)
        return False

    ax.set_title(f"{BASE_CONFIG_ID.title()} Delay vs Number of Videos")
    ax.set_xlabel("Number of Videos")
    ax.set_ylabel("Delay (s)")
    ax.set_xticks(num_videos)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_delay_comparison_by_config(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
) -> bool:
    """Grouped bar comparing delay metrics averaged over all videos per config."""
    configure_plot_style()
    colors = _scientific_colors()
    delay_keys = [metric for metric, *_ in DELAY_METRICS]
    collected = []
    config_labels = []

    for config_id in config_ids:
        per_metric_means = []
        has_values = False
        for metric_key in delay_keys:
            num_map = summary_stats.get(metric_key, {}).get(config_id, {})
            values = []
            for stats in num_map.values():
                values.extend(stats.get("values", []))
            if values:
                mean, _ = safe_mean_std(values)
                has_values = True
            else:
                mean = float("nan")
            per_metric_means.append(mean)
        if has_values:
            collected.append(per_metric_means)
            config_labels.append(config_id)

    if not collected:
        return False

    arr = np.asarray(collected, dtype=float)
    x = np.arange(len(config_labels))
    bar_width = min(0.8 / len(delay_keys), 0.2)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for idx, (_, _, title) in enumerate(DELAY_METRICS):
        ax.bar(
            x + (idx - (len(delay_keys) - 1) / 2) * bar_width,
            arr[:, idx],
            width=bar_width * 0.95,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            label=title,
        )

    ax.set_title("Delay Comparison Across Configs")
    ax.set_ylabel("Delay (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=20, ha="right")
    ax.set_xlabel("Config ID")
    ax.legend(frameon=False)
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
    requested_config_ids: Optional[List[str]] = list(args.config_ids) if args.config_ids else None
    requested_num_videos: Optional[List[int]] = list(args.num_videos) if args.num_videos else None
    iterations: List[int] = list(args.iterations)
    data_sources: List[str] = list(dict.fromkeys(args.data_sources))

    all_metric_keys = [key for key, *_ in METRIC_ORDER] + [key for key, *_ in DELAY_METRICS]

    all_metrics_storage: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}
    available_numbers: Dict[str, Set[int]] = defaultdict(set)
    missing_paths: Dict[str, List[Path]] = defaultdict(list)

    for data_source in data_sources:
        # Determine directory for this data source
        candidate_dir = base_dir / data_source
        if candidate_dir.exists():
            ds_root = candidate_dir
        elif base_dir.name == data_source and base_dir.exists():
            ds_root = base_dir
        else:
            ds_root = candidate_dir
        ds_root = ds_root.resolve()

        if not ds_root.exists():
            print(f"Warning: Data source directory not found for '{data_source}': {ds_root}")
            continue

        # Determine number of videos to inspect
        if requested_num_videos:
            num_videos = sorted(set(requested_num_videos))
        else:
            detected = sorted(
                int(p.name[1:])
                for p in ds_root.glob("N*")
                if p.is_dir() and len(p.name) > 1 and p.name[1:].isdigit()
            )
            num_videos = detected if detected else list(DEFAULT_NUM_VIDEOS)

        # Discover configs if not explicitly provided
        configs_for_ds: Set[str] = set(requested_config_ids or [])
        for n in num_videos:
            n_dir = ds_root / f"N{n}"
            if not n_dir.exists():
                continue
            for cfg_dir in n_dir.iterdir():
                if cfg_dir.is_dir():
                    if requested_config_ids is None or cfg_dir.name in requested_config_ids:
                        configs_for_ds.add(cfg_dir.name)
        if not configs_for_ds:
            configs_for_ds.update(DEFAULT_CONFIG_IDS)

        # Initialize storage for this data source
        metrics_storage = {
            metric: defaultdict(lambda: defaultdict(list)) for metric in all_metric_keys
        }

        for num in num_videos:
            for config_id in configs_for_ds:
                for iteration in iterations:
                    summary_path = (
                        ds_root
                        / f"N{num}"
                        / config_id
                        / f"I{iteration}"
                        / "overall_summary.json"
                    )
                    if not summary_path.exists():
                        missing_paths[data_source].append(summary_path)
                        continue
                    metrics = load_iteration_metrics(summary_path)
                    available_numbers[data_source].add(num)
                    for metric_key in all_metric_keys:
                        metrics_storage[metric_key][config_id][num].append(
                            metrics.get(metric_key, float("nan"))
                        )

        all_metrics_storage[data_source] = metrics_storage

    if missing_paths:
        for data_source, paths in missing_paths.items():
            if paths:
                print(f"Warning: Missing summaries for data_source '{data_source}':")
                for path in paths:
                    print(f"  - {path}")

    if not all_metrics_storage:
        print("No data available to plot. Exiting.")
        return

    # Compute summary stats per data source
    all_summary_stats: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]] = {}
    for data_source, ds_metrics in all_metrics_storage.items():
        summary_stats = {metric: {} for metric in all_metric_keys}
        for metric_key, configs in ds_metrics.items():
            for config_id, num_map in configs.items():
                summary_stats[metric_key].setdefault(config_id, {})
                for num, values in num_map.items():
                    mean, std = safe_mean_std(values)
                    summary_stats[metric_key][config_id][num] = {
                        "mean": mean,
                        "std": std,
                        "values": list(values),
                    }
        all_summary_stats[data_source] = summary_stats

    metric_meta = {key: (y_label, title, slug) for key, y_label, title, slug in METRIC_ORDER}

    for data_source, summary_stats in all_summary_stats.items():
        # Determine numbers and configs for plotting
        ds_numbers = sorted(available_numbers.get(data_source, []))
        if requested_num_videos:
            ds_numbers = sorted(set(ds_numbers) | set(requested_num_videos))
        if not ds_numbers:
            ds_numbers = list(DEFAULT_NUM_VIDEOS)

        configs_for_ds_set: Set[str] = set()
        for configs in summary_stats.values():
            configs_for_ds_set.update(configs.keys())
        configs_for_ds = sorted(configs_for_ds_set)
        if requested_config_ids:
            configs_for_ds = sorted(set(configs_for_ds) | set(requested_config_ids))
        if not configs_for_ds:
            configs_for_ds = list(DEFAULT_CONFIG_IDS)

        if args.output is None:
            plot_root_base = (base_dir / data_source / f"{data_source}_config_metrics")
        else:
            output = Path(args.output)
            if len(data_sources) > 1:
                plot_root_base = output.with_name(f"{output.stem}_{data_source}").with_suffix("")
            else:
                plot_root_base = output.with_suffix("")

        plot_root_base.parent.mkdir(parents=True, exist_ok=True)

        for metric_key, (y_label, title, slug) in metric_meta.items():
            per_video_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_{slug}_per_video.pdf"
            overall_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_{slug}_overall.pdf"
            per_video_saved = plot_grouped_bar(
                metric_key,
                title,
                y_label,
                slug,
                summary_stats,
                ds_numbers,
                configs_for_ds,
                per_video_path,
            )
            if per_video_saved:
                print(f"[{data_source}] Saved per-video bar plot to {per_video_path}")
            else:
                print(f"[{data_source}] Skipped per-video bar plot for {metric_key}; no valid data.")

            overall_saved = plot_overall_bar(
                metric_key,
                title,
                y_label,
                slug,
                summary_stats,
                configs_for_ds,
                overall_path,
            )
            if overall_saved:
                print(f"[{data_source}] Saved aggregated bar plot to {overall_path}")
            else:
                print(f"[{data_source}] Skipped aggregated bar plot for {metric_key}; no valid data.")

        # Base delay trends
        base_delay_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_base_delay_vs_videos.pdf"
        if plot_base_delay_trends(summary_stats, ds_numbers, base_delay_path):
            print(f"[{data_source}] Saved base delay trend plot to {base_delay_path}")
        else:
            print(f"[{data_source}] Skipped base delay trend plot; no valid data.")

        # Delay comparison across configs
        delay_compare_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_delay_by_config.pdf"
        if plot_delay_comparison_by_config(summary_stats, configs_for_ds, delay_compare_path):
            print(f"[{data_source}] Saved delay comparison plot to {delay_compare_path}")
        else:
            print(f"[{data_source}] Skipped delay comparison plot; no valid data.")

        # CSV summary
        csv_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}.csv"
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
        print(f"[{data_source}] Wrote summary table to {csv_path}")


if __name__ == "__main__":
    main()
