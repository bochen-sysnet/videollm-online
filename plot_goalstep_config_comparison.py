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
DEFAULT_ABLATION_NUM_VIDEOS: Tuple[int, ...] = (5, 10, 15)
DEFAULT_ITERATIONS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_BASE_DIR = Path("figures")
BASE_CONFIG_ID = "base"

METRIC_ORDER: Tuple[Tuple[str, str, str, str], ...] = (
    ("rebuffer_time", "Average Rebuffering Time (s)", "Rebuffering Time", "rebuffer"),
    ("ttft", "Average TTFT (s)", "TTFT", "ttft"),
    ("scheduling_score", "Final Scheduling Score", "Scheduling Score", "scheduling"),
    ("perplexity", "Average VLM Perplexity", "Perplexity", "perplexity"),
)

LATENCY_COMPONENTS: Tuple[Tuple[str, str, str], ...] = (
    ("visual_embedding_time", "Prefilling", "prefilling"),
    ("model_forward_time", "Scoring", "scoring"),
    ("generation_time", "Generation", "generation"),
    ("kv_offload_time", "KV Offload", "kv_offload"),
    ("kv_reload_time", "KV Reload", "kv_reload"),
    ("total_sending_time", "Network", "network_time"),
    ("total_processing_time", "Total Processing", "total_processing"),
)

DELAY_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("processing_delay", "Processing Delay", "processing_delay"),
    ("queuing_delay", "Queuing Delay", "queuing_delay"),
    ("network_delay", "Networking Delay", "network_delay"),
)

SCHEDULING_COMPONENTS: Tuple[Tuple[str, str, str], ...] = (
    ("lowest_buffer_score", "Lowest Buffer", "lowest_buffer"),
    ("nonzero_score", "Nonzero", "nonzero"),
    ("ending_score", "Ending", "ending"),
)

MEMORY_COMPONENTS: Tuple[Tuple[str, str, str], ...] = (
    ("model_params_memory", "Model Params", "model_params"),
    ("combined_dynamic_memory", "Combined Dynamic", "combined_dynamic"),
    ("cpu_memory_growth_peak", "CPU Growth", "cpu_growth"),
)
GENERATION_SPEED_METRIC = "generation_speed"

ABLATION_GROUPS = {
    "rl": {
        "title": "Running Length Weight Ablation",
        "configs": [
            ("base", "Default (0.1)"),
            ("rl_ablation1", "Weight 0"),
            ("rl_ablation2", "Weight 1"),
            ("rl_ablation3", "Weight 10"),
        ],
        "num_videos": [5, 10, 15, 20],
        "slug": "rl_ablation",
    },
    "comp": {
        "title": "Computation Modules Ablation",
        "configs": [
            ("base", "Default"),
            ("comp_ablation1", "No Comp 1"),
            ("comp_ablation2", "No Comp 2"),
            ("comp_ablation3", "No Comp 3"),
            ("comp_ablation4", "No Comp 4"),
        ],
        "num_videos": [5, 10, 15, 20],
        "slug": "comp_ablation",
    },
    "chunk": {
        "title": "Chunk Size Ablation",
        "configs": [
            ("base", "Chunk 2 (Default)"),
            ("chunk_ablation1", "Chunk 1"),
            ("chunk_ablation2", "Chunk 4"),
            ("chunk_ablation3", "Chunk 8"),
            ("chunk_ablation4", "Chunk 16"),
            ("chunk_ablation5", "Chunk 32"),
        ],
        "num_videos": [5, 10, 15, 20],
        "slug": "chunk_ablation",
    },
    "factor": {
        "title": "Factor Ablation",
        "configs": [
            ("base", "Default"),
            ("factor_ablation1", "Factor 0"),
            ("factor_ablation2", "Factor 0.1"),
            ("factor_ablation3", "Factor 0.3"),
            ("factor_ablation4", "Factor 0.4"),
            ("factor_ablation5", "Factor 0.5"),
            ("factor_ablation6", "Factor 0.6"),
            ("factor_ablation7", "Factor 0.7"),
            ("factor_ablation8", "Factor 0.8"),
            ("factor_ablation9", "Factor 0.9"),
            ("factor_ablation10", "Factor 1.0"),
        ],
        "num_videos": [5, 10, 15, 20],
        "slug": "factor_ablation",
    },
    "consumption": {
        "title": "Consumption Ablation",
        "grouped_configs": [
            ("Speed 1", [
                ("consumption_ablation1_base", "Base"),
                ("consumption_ablation1_rr_2", "RR-2"),
                ("consumption_ablation1_rr_m", "RR-m"),
                ("consumption_ablation1_rand_2", "Rand-2"),
                ("consumption_ablation1_rand_m", "Rand-m"),
            ]),
            ("Speed 2", [
                ("consumption_ablation2_base", "Base"),
                ("consumption_ablation2_rr_2", "RR-2"),
                ("consumption_ablation2_rr_m", "RR-m"),
                ("consumption_ablation2_rand_2", "Rand-2"),
                ("consumption_ablation2_rand_m", "Rand-m"),
            ]),
            ("Speed 3", [
                ("consumption_ablation3_base", "Base"),
                ("consumption_ablation3_rr_2", "RR-2"),
                ("consumption_ablation3_rr_m", "RR-m"),
                ("consumption_ablation3_rand_2", "Rand-2"),
                ("consumption_ablation3_rand_m", "Rand-m"),
            ]),
            ("Speed 4", [
                ("consumption_ablation4_base", "Base"),
                ("consumption_ablation4_rr_2", "RR-2"),
                ("consumption_ablation4_rr_m", "RR-m"),
                ("consumption_ablation4_rand_2", "Rand-2"),
                ("consumption_ablation4_rand_m", "Rand-m"),
            ]),
            ("Speed 5", [
                ("consumption_ablation5_base", "Base"),
                ("consumption_ablation5_rr_2", "RR-2"),
                ("consumption_ablation5_rr_m", "RR-m"),
                ("consumption_ablation5_rand_2", "Rand-2"),
                ("consumption_ablation5_rand_m", "Rand-m"),
            ]),
        ],
        "num_videos": [5],
        "slug": "consumption_ablation",
    },
}

ABLATION_CONFIG_SET: Set[str] = set()
ABLATION_CONFIG_ALLOWED_NUMS: Dict[str, Set[int]] = {}
for group_meta in ABLATION_GROUPS.values():
    allowed_nums = set(group_meta.get("num_videos", []))
    if "configs" in group_meta:
        for cfg, _ in group_meta["configs"]:
            if cfg != "base":
                ABLATION_CONFIG_SET.add(cfg)
                ABLATION_CONFIG_ALLOWED_NUMS[cfg] = allowed_nums.copy()
    if "grouped_configs" in group_meta:
        for _, cfgs in group_meta["grouped_configs"]:
            for cfg, _ in cfgs:
                if cfg != "base":
                    ABLATION_CONFIG_SET.add(cfg)
                    ABLATION_CONFIG_ALLOWED_NUMS[cfg] = allowed_nums.copy()


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
        default=["goalstep", "narration"],
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
        default=["base", "random_m", "random_2", "round_robin_m", "round_robin_2"],
        help="Config IDs to include in the general (non-ablation) comparison.",
    )
    parser.add_argument(
        "--general-num-videos",
        nargs="+",
        type=int,
        default=None,
        help="Number of videos for general (non-ablation) plots. Auto-detect when omitted.",
    )
    parser.add_argument(
        "--ablation-num-videos",
        nargs="+",
        type=int,
        default=None,
        help="Number of videos for ablation plots (default: 5 10 15 20).",
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
    component_samples: Dict[str, List[float]] = {key: [] for key, _, _ in LATENCY_COMPONENTS}
    results = summary.get("results", []) or []
    ttft_samples: List[float] = []
    ppl_samples: List[float] = []
    total_generated_words = 0.0
    total_generation_time = 0.0
    for result in results:
        if result.get("generated_turns"):
            resp_time = result.get("response_time")
            if resp_time is not None:
                ttft_samples.append(float(resp_time))
        ppl_data = result.get("ppl_data") or {}
        ppl_values = ppl_data.get("gt_ppls_vlm_prefix_visual") or []
        ppl_samples.extend(float(p) for p in ppl_values if p is not None)
        for key, _, _ in LATENCY_COMPONENTS:
            value = result.get(key)
            if value is not None:
                component_samples[key].append(float(value))

        for turn in result.get("generated_turns", []):
            text = turn.get("text", "")
            if "Assistant:" in text:
                text = text.split("Assistant:", 1)[-1]
            word_count = len([w for w in text.strip().split() if w])
            total_generated_words += word_count
            total_generation_time += float(turn.get("generation_time", 0.0))
    ttft = float(np.mean(ttft_samples)) if ttft_samples else float("nan")
    perplexity = float(np.mean(ppl_samples)) if ppl_samples else float("nan")

    # Scheduling score: use the final cumulative score as overall value.
    scheduling = summary.get("scheduling_data") or {}
    score_series = scheduling.get("selected_score") or []
    scheduling_score = float(score_series[-1]) if score_series else float("nan")
    lowest_series = scheduling.get("selected_lowest") or []
    increment_series = scheduling.get("selected_increment") or []
    ending_series = scheduling.get("selected_ending") or []
    lowest_buffer_score = float(lowest_series[-1]) if lowest_series else float("nan")
    nonzero_score = float(increment_series[-1]) if increment_series else float("nan")
    ending_score = float(ending_series[-1]) if ending_series else float("nan")

    # Memory metrics aggregated across conversations
    all_memory_data = summary.get("all_memory_data") or {}
    model_params_values: List[float] = []
    combined_dynamic_values: List[float] = []
    cpu_growth_values: List[float] = []
    for memory_data in all_memory_data.values():
        model_mem = memory_data.get("model_memory") or []
        kv_mem = memory_data.get("kv_cache_memory") or []
        activation_mem = memory_data.get("activation_memory") or []
        other_mem = memory_data.get("other_memory") or []
        cpu_mem_growth = memory_data.get("cpu_memory_growth") or []

        if model_mem:
            model_params_values.append(float(max(model_mem)))
        if kv_mem and activation_mem and other_mem:
            combined_dynamic_values.append(
                float(max(kv_mem)) + float(max(activation_mem)) + float(max(other_mem))
            )
        if cpu_mem_growth:
            cpu_growth_values.append(float(max(cpu_mem_growth)))

    processing_delay = float(np.mean(processing_delays)) if processing_delays else float("nan")
    queuing_delay = float(np.mean(queuing_delays)) if queuing_delays else float("nan")
    networking_delay = float(np.mean(networking_delays)) if networking_delays else float("nan")

    component_means = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in component_samples.items()
    }

    metrics = {
        "rebuffer_time": rebuffer_time,
        "ttft": ttft,
        "scheduling_score": scheduling_score,
        "perplexity": perplexity,
        "processing_delay": processing_delay,
        "queuing_delay": queuing_delay,
        "network_delay": networking_delay,
    }
    metrics.update(
        {
            "lowest_buffer_score": lowest_buffer_score,
            "nonzero_score": nonzero_score,
            "ending_score": ending_score,
            "model_params_memory": float(np.mean(model_params_values)) if model_params_values else float("nan"),
            "combined_dynamic_memory": float(np.mean(combined_dynamic_values)) if combined_dynamic_values else float("nan"),
            "cpu_memory_growth_peak": float(np.mean(cpu_growth_values)) if cpu_growth_values else float("nan"),
        }
    )
    if total_generation_time > 0:
        metrics[GENERATION_SPEED_METRIC] = total_generated_words / total_generation_time
    else:
        metrics[GENERATION_SPEED_METRIC] = float("nan")
    metrics.update(component_means)
    return metrics


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


def plot_latency_components_by_config(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
) -> bool:
    """Compare latency components averaged across all videos for each config."""
    configure_plot_style()
    colors = _scientific_colors()
    component_keys = [key for key, _, _ in LATENCY_COMPONENTS]
    component_labels = [label for _, label, _ in LATENCY_COMPONENTS]
    num_components = len(component_keys)
    num_configs = len(config_ids)
    bar_width = min(0.8 / max(num_configs, 1), 0.18)
    x = np.arange(num_components)

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    has_data = False

    for idx, config_id in enumerate(config_ids):
        means = []
        stds = []
        for key in component_keys:
            num_map = summary_stats.get(key, {}).get(config_id, {})
            collected: List[float] = []
            for stats in num_map.values():
                collected.extend(stats.get("values", []))
            if collected:
                mean, std = safe_mean_std(collected)
            else:
                mean, std = (float("nan"), float("nan"))
            means.append(mean)
            stds.append(std)

        means_arr = np.asarray(means, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue

        has_data = True
        offset = (idx - (num_configs - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds,
            capsize=2.5,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            label=config_id,
        )

    ax.set_title("Latency Components by Config")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(component_labels, rotation=25, ha="right")
    ax.set_xlabel("Component")
    if has_data:
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout(pad=0.6)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return has_data


def plot_latency_components_vs_videos(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    num_videos: List[int],
    output_path: Path,
) -> bool:
    """Show how latency components evolve as the number of videos increases."""
    configure_plot_style()
    colors = _scientific_colors()
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    component_entries = [(key, label) for key, label, _ in LATENCY_COMPONENTS]
    if not component_entries:
        return False

    cols = min(4, len(component_entries))
    rows = math.ceil(len(component_entries) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 2.3 * rows),
        sharex=True,
        sharey=False,
    )
    axes = np.atleast_1d(axes).flatten()
    plotted = False
    combined_nums: Set[int] = set()

    for ax, (metric_key, comp_label) in zip(axes, component_entries):
        for idx, config_id in enumerate(config_ids):
            stats_for_config = summary_stats.get(metric_key, {}).get(config_id, {})
            means = []
            for n in num_videos:
                entry = stats_for_config.get(n, {})
                means.append(entry.get("mean", float("nan")))
            means_arr = np.asarray(means, dtype=float)
            if np.all(~np.isfinite(means_arr)):
                continue
            plotted = True
            ax.plot(
                num_videos,
                means_arr,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=1.6,
                markersize=4,
                label=config_id,
            )
        ax.set_title(comp_label)
        ax.set_ylabel("Time (s)")
        ax.set_xticks(num_videos)

    for ax in axes[len(component_entries) :]:
        ax.axis("off")

    if plotted:
        for ax in axes[-cols:]:
            ax.set_xlabel("Number of Videos")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(3, len(config_ids)),
                frameon=False,
            )
            fig.subplots_adjust(bottom=0.15)
        fig.tight_layout(pad=0.7)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return plotted


def plot_scheduling_components_by_config(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
) -> bool:
    """Grouped bar chart comparing scheduling components across configs."""
    configure_plot_style()
    colors = _scientific_colors()
    component_keys = [key for key, _, _ in SCHEDULING_COMPONENTS]
    component_labels = [label for _, label, _ in SCHEDULING_COMPONENTS]

    means_matrix: List[List[float]] = []
    std_matrix: List[List[float]] = []
    valid_configs: List[str] = []

    for config_id in config_ids:
        config_means = []
        config_stds = []
        has_data = False
        for key in component_keys:
            num_map = summary_stats.get(key, {}).get(config_id, {})
            collected: List[float] = []
            for stats in num_map.values():
                collected.extend(stats.get("values", []))
            if collected:
                mean, std = safe_mean_std(collected)
                has_data = True
            else:
                mean, std = float("nan"), float("nan")
            config_means.append(mean)
            config_stds.append(std)
        if has_data:
            valid_configs.append(config_id)
            means_matrix.append(config_means)
            std_matrix.append(config_stds)

    if not means_matrix:
        return False

    x = np.arange(len(valid_configs))
    bar_width = min(0.8 / len(component_keys), 0.2)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    for idx, (label, color) in enumerate(zip(component_labels, colors)):
        means_arr = np.asarray([row[idx] for row in means_matrix], dtype=float)
        stds_arr = np.asarray([row[idx] for row in std_matrix], dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        offset = (idx - (len(component_keys) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

    ax.set_title("Scheduling Components by Config")
    ax.set_ylabel("Cumulative Score")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_configs, rotation=20, ha="right")
    ax.set_xlabel("Config ID")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scheduling_components_base_vs_videos(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    num_videos: List[int],
    output_path: Path,
) -> bool:
    """Line plot showing scheduling components over number of videos for base config."""
    configure_plot_style()
    base_components = {
        key: summary_stats.get(key, {}).get(BASE_CONFIG_ID, {})
        for key, _, _ in SCHEDULING_COMPONENTS
    }

    if not any(base_components.values()):
        return False

    colors = _scientific_colors()
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    has_data = False

    for idx, (key, label, _) in enumerate(SCHEDULING_COMPONENTS):
        stats_for_base = base_components.get(key, {})
        means = []
        stds = []
        for n in num_videos:
            entry = stats_for_base.get(n, {})
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
            label=label,
        )

    if not has_data:
        plt.close(fig)
        return False

    ax.set_title(f"{BASE_CONFIG_ID.title()} Scheduling Components vs Number of Videos")
    ax.set_xlabel("Number of Videos")
    ax.set_ylabel("Cumulative Score")
    ax.set_xticks(num_videos)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_memory_components_by_config(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
) -> bool:
    """Grouped bar chart comparing memory components across configs (log scale)."""
    configure_plot_style()
    colors = _scientific_colors()
    component_keys = [key for key, _, _ in MEMORY_COMPONENTS]
    component_labels = [label for _, label, _ in MEMORY_COMPONENTS]

    means_matrix: List[List[float]] = []
    std_matrix: List[List[float]] = []
    valid_configs: List[str] = []

    for config_id in config_ids:
        config_means = []
        config_stds = []
        has_data = False
        for key in component_keys:
            num_map = summary_stats.get(key, {}).get(config_id, {})
            collected: List[float] = []
            for stats in num_map.values():
                collected.extend(stats.get("values", []))
            if collected:
                mean, std = safe_mean_std(collected)
                has_data = True
            else:
                mean, std = float("nan"), float("nan")
            config_means.append(mean)
            config_stds.append(std)
        if has_data:
            valid_configs.append(config_id)
            means_matrix.append(config_means)
            std_matrix.append(config_stds)

    if not means_matrix:
        return False

    x = np.arange(len(valid_configs))
    bar_width = min(0.8 / len(component_keys), 0.2)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    for idx, (label, color) in enumerate(zip(component_labels, colors)):
        means_arr = np.asarray([row[idx] for row in means_matrix], dtype=float)
        stds_arr = np.asarray([row[idx] for row in std_matrix], dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        offset = (idx - (len(component_keys) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

    ax.set_title("Memory Components by Config")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_configs, rotation=20, ha="right")
    ax.set_xlabel("Config ID")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_memory_components_base_vs_videos(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    num_videos: List[int],
    output_path: Path,
) -> bool:
    """Line plot of memory components vs number of videos for the base config (log scale)."""
    configure_plot_style()
    base_stats = {
        key: summary_stats.get(key, {}).get(BASE_CONFIG_ID, {})
        for key, _, _ in MEMORY_COMPONENTS
    }
    if not any(base_stats.values()):
        return False

    colors = _scientific_colors()
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    has_data = False

    for idx, (key, label, _) in enumerate(MEMORY_COMPONENTS):
        stats_for_base = base_stats.get(key, {})
        means = []
        for n in num_videos:
            entry = stats_for_base.get(n, {})
            means.append(entry.get("mean", float("nan")))
        means_arr = np.asarray(means, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        has_data = True
        ax.plot(
            num_videos,
            means_arr,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.6,
            markersize=4,
            label=label,
        )

    if not has_data:
        plt.close(fig)
        return False

    ax.set_title(f"{BASE_CONFIG_ID.title()} Memory Components vs Number of Videos")
    ax.set_xlabel("Number of Videos")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_yscale("log")
    ax.set_xticks(num_videos)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_roundrobin_comparison(
    general_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    data_sources: List[str],
    available_numbers: Dict[str, Set[int]],
    general_num_override: Optional[List[int]],
    output_path: Path,
) -> bool:
    """Plot round_robin_m rebuffering vs number of videos for each data source."""
    configure_plot_style()
    colors = _scientific_colors()
    markers = ["o", "s", "^", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    plotted = False
    combined_nums: Set[int] = set()

    for idx, data_source in enumerate(data_sources):
        summary_stats = general_stats.get(data_source)
        if not summary_stats:
            continue
        rebuffer_stats = summary_stats.get("rebuffer_time", {}).get("round_robin_m", {})
        if not rebuffer_stats:
            continue
        nums = sorted(available_numbers.get(data_source, []))
        if general_num_override:
            nums = [n for n in nums if n in general_num_override]
        means = []
        stds = []
        valid_nums = []
        for n in nums:
            entry = rebuffer_stats.get(n, {})
            mean = entry.get("mean")
            if mean is None or not math.isfinite(mean):
                continue
            std = entry.get("std", 0.0)
            if std is None or not math.isfinite(std):
                std = 0.0
            means.append(mean)
            stds.append(std)
            valid_nums.append(n)
        if not means:
            continue
        plotted = True
        combined_nums.update(valid_nums)
        ax.errorbar(
            valid_nums,
            means,
            yerr=stds,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.6,
            markersize=4,
            capsize=3,
            label=data_source.title(),
        )

    if not plotted or not combined_nums:
        plt.close(fig)
        return False

    ax.set_title("RoundRobin_m Rebuffering vs Number of Videos")
    ax.set_xlabel("Number of Videos")
    ax.set_ylabel("Total Rebuffering Time (s)")
    ax.set_xticks(sorted(combined_nums))
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_roundrobin_ratio(
    general_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    data_sources: List[str],
    available_numbers: Dict[str, Set[int]],
    general_num_override: Optional[List[int]],
    output_path: Path,
) -> bool:
    """Plot ratio of round_robin_2 to round_robin_m rebuffering vs number of videos for each data source."""
    configure_plot_style()
    colors = _scientific_colors()
    markers = ["o", "s", "^", "D", "P", "X"]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    plotted = False
    combined_nums: Set[int] = set()
    combined_nums: Set[int] = set()

    for idx, data_source in enumerate(data_sources):
        summary_stats = general_stats.get(data_source)
        if not summary_stats:
            continue
        m_stats = summary_stats.get("rebuffer_time", {}).get("round_robin_m", {})
        rr2_stats = summary_stats.get("rebuffer_time", {}).get("round_robin_2", {})
        if not m_stats or not rr2_stats:
            continue
        nums = sorted(available_numbers.get(data_source, []))
        if general_num_override:
            nums = [n for n in nums if n in general_num_override]
        ratios = []
        stds = []
        valid_nums = []
        for n in nums:
            entry_m = m_stats.get(n, {})
            entry_rr2 = rr2_stats.get(n, {})
            mean_m = entry_m.get("mean")
            mean_rr2 = entry_rr2.get("mean")
            if (
                mean_m is None
                or mean_rr2 is None
                or not math.isfinite(mean_m)
                or not math.isfinite(mean_rr2)
                or mean_m == 0
            ):
                continue

            # compute ratio
            ratio = mean_rr2 / mean_m
            ratios.append(ratio)

            # approximate std of ratio assuming independence
            std_m = entry_m.get("std", 0.0) or 0.0
            std_rr2 = entry_rr2.get("std", 0.0) or 0.0
            # avoid division by zero
            ratio_std = 0.0
            if mean_m != 0:
                ratio_std = ratio * math.sqrt(
                    (std_rr2 / mean_rr2) ** 2 if mean_rr2 else 0.0
                    + (std_m / mean_m) ** 2
                )
            stds.append(ratio_std)
            valid_nums.append(n)
        if not ratios:
            continue
        plotted = True
        combined_nums.update(valid_nums)
        ax.errorbar(
            valid_nums,
            ratios,
            yerr=stds,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.6,
            markersize=4,
            capsize=3,
            label=data_source.title(),
        )

    if not plotted or not combined_nums:
        plt.close(fig)
        return False

    ax.set_title("RoundRobin_2 / RoundRobin_m Rebuffering Ratio")
    ax.set_xlabel("Number of Videos")
    ax.set_ylabel("Rebuffering Ratio")
    ax.set_xticks(sorted(combined_nums))
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_generation_speed_single_video(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    configs: List[str],
    output_path: Path,
    single_video_num: int = 1,
) -> bool:
    """Bar plot comparing generation speed (words/s) for single-video experiments."""
    speed_stats = summary_stats.get(GENERATION_SPEED_METRIC, {})
    if not speed_stats:
        return False

    means = []
    stds = []
    labels = []
    for cfg in configs:
        entry = speed_stats.get(cfg, {}).get(single_video_num)
        if not entry:
            continue
        mean = entry.get("mean")
        if mean is None or not math.isfinite(mean):
            continue
        std = entry.get("std", 0.0) or 0.0
        means.append(mean)
        stds.append(std)
        labels.append(cfg)

    if not means:
        return False

    configure_plot_style()
    colors = _scientific_colors()
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
    ax.set_title(f"Generation Speed (N={single_video_num})")
    ax.set_ylabel("Words per Second")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_rebuffering_ablation_group(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    group_key: str,
    available_configs: Set[str],
    available_nums: List[int],
    target_nums: List[int],
    output_path: Path,
) -> bool:
    """Bar plot comparing rebuffering time across ablation configs with per-N differentiation."""
    group_meta = ABLATION_GROUPS.get(group_key)
    if not group_meta:
        return False

    rebuffer_stats = summary_stats.get("rebuffer_time", {})
    if not rebuffer_stats:
        return False

    # Handle grouped configs (e.g., consumption ablation)
    if "grouped_configs" in group_meta:
        grouped_cfgs = group_meta["grouped_configs"]
        # Determine which number of videos to use (first available intersection)
        single_num = None
        for n in target_nums:
            if n in available_nums:
                single_num = n
                break
        if single_num is None:
            return False

        group_labels = []
        baseline_labels: List[str] = []
        baselines = []  # list of config names per baseline index (common across groups)

        # Initialize from first group that has data
        for group_label, configs_list in grouped_cfgs:
            # On first valid group, capture baseline ordering
            if not baseline_labels:
                baseline_labels = [label for _, label in configs_list]
                baselines = [cfg for cfg, _ in configs_list]
            # Ensure group ordering is consistent even if some configs missing
            group_labels.append(group_label)

        if not baseline_labels:
            return False

        values = {label: [] for label in baseline_labels}
        errors = {label: [] for label in baseline_labels}

        for group_label, configs_list in grouped_cfgs:
            label_to_cfg = {label: cfg for cfg, label in configs_list}
            for baseline_label in baseline_labels:
                cfg_name = label_to_cfg.get(baseline_label)
                if cfg_name is None:
                    values[baseline_label].append(float("nan"))
                    errors[baseline_label].append(float("nan"))
                    continue
                entry = rebuffer_stats.get(cfg_name, {}).get(single_num, {})
                mean = entry.get("mean")
                std = entry.get("std", 0.0)
                if mean is None or not math.isfinite(mean):
                    values[baseline_label].append(float("nan"))
                    errors[baseline_label].append(float("nan"))
                else:
                    if std is None or not math.isfinite(std):
                        std = 0.0
                    values[baseline_label].append(mean)
                    errors[baseline_label].append(std)

        # Remove baselines with no data at all
        valid_baselines = [bl for bl in baseline_labels if any(math.isfinite(v) for v in values[bl])]
        if not valid_baselines:
            return False

        configure_plot_style()
        colors = _scientific_colors()
        x = np.arange(len(group_labels))
        bar_width = min(0.75 / max(len(valid_baselines), 1), 0.18)
        fig, ax = plt.subplots(figsize=(3.8, 2.6))

        for idx, baseline_label in enumerate(valid_baselines):
            means_arr = np.asarray(values[baseline_label], dtype=float)
            stds_arr = np.asarray(errors[baseline_label], dtype=float)
            offset = (idx - (len(valid_baselines) - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means_arr,
                width=bar_width * 0.95,
                yerr=stds_arr,
                capsize=2.5,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.4,
                label=baseline_label,
            )

        ax.set_title(group_meta["title"])
        ax.set_ylabel("Total Rebuffering Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=20, ha="right")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(pad=0.8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return True

    ablation_configs = [
        (cfg, label)
        for cfg, label in group_meta["configs"]
        if cfg in available_configs
    ]
    if not ablation_configs:
        return False

    selected_nums = [
        n for n in target_nums if n in available_nums
    ]
    if not selected_nums:
        return False

    configure_plot_style()
    colors = _scientific_colors()
    num_videos = selected_nums
    x_positions = np.arange(len(num_videos))
    bar_width = min(0.75 / max(len(ablation_configs), 1), 0.18)

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    plotted = False

    for idx, (cfg, label) in enumerate(ablation_configs):
        means = []
        stds = []
        for n in num_videos:
            entry = rebuffer_stats.get(cfg, {}).get(n, {})
            mean = entry.get("mean", float("nan"))
            std = entry.get("std", 0.0)
            if std is None or math.isnan(std):
                std = 0.0
            means.append(mean)
            stds.append(std)

        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        plotted = True

        offset = (idx - (len(ablation_configs) - 1) / 2) * bar_width
        ax.bar(
            x_positions + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(group_meta["title"])
    ax.set_ylabel("Total Rebuffering Time (s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in num_videos])
    ax.set_xlabel("Number of Videos")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(pad=0.8)
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
    general_config_ids: List[str] = list(args.config_ids) if args.config_ids else list(DEFAULT_CONFIG_IDS)
    general_num_override: Optional[List[int]] = (
        sorted(set(args.general_num_videos)) if args.general_num_videos else None
    )
    if args.ablation_num_videos:
        ablation_target_numbers: List[int] = sorted(set(args.ablation_num_videos))
    else:
        nums_union: Set[int] = set()
        for group_meta in ABLATION_GROUPS.values():
            nums_union.update(group_meta.get("num_videos", []))
        ablation_target_numbers = sorted(nums_union) if nums_union else list(DEFAULT_ABLATION_NUM_VIDEOS)
    ablation_num_set = set(ablation_target_numbers)
    iterations: List[int] = list(args.iterations)
    data_sources: List[str] = list(dict.fromkeys(args.data_sources))

    all_metric_keys = (
        [key for key, *_ in METRIC_ORDER]
        + [key for key, *_ in LATENCY_COMPONENTS]
        + [key for key, *_ in DELAY_METRICS]
        + [key for key, *_ in SCHEDULING_COMPONENTS]
        + [key for key, *_ in MEMORY_COMPONENTS]
        + [GENERATION_SPEED_METRIC]
    )

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
        if general_num_override:
            num_videos = general_num_override
        else:
            detected = sorted(
                int(p.name[1:])
                for p in ds_root.glob("N*")
                if p.is_dir() and len(p.name) > 1 and p.name[1:].isdigit()
            )
            num_videos = detected if detected else list(DEFAULT_NUM_VIDEOS)
        num_videos = sorted(set(num_videos) | set(ablation_target_numbers))

        # Discover configs present in this data source
        configs_for_ds: Set[str] = set()
        for n in num_videos:
            n_dir = ds_root / f"N{n}"
            if not n_dir.exists():
                continue
            for cfg_dir in n_dir.iterdir():
                if cfg_dir.is_dir():
                    configs_for_ds.add(cfg_dir.name)
        if not configs_for_ds:
            configs_for_ds.update(general_config_ids)
        configs_for_ds = sorted(configs_for_ds)

        # Initialize storage for this data source
        metrics_storage = {
            metric: defaultdict(lambda: defaultdict(list)) for metric in all_metric_keys
        }

        for num in num_videos:
            for config_id in configs_for_ds:
                allowed = ABLATION_CONFIG_ALLOWED_NUMS.get(config_id)
                if allowed:
                    if num not in allowed:
                        continue
                elif config_id in ABLATION_CONFIG_SET and num not in ablation_num_set:
                    continue
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
        if general_num_override:
            ds_numbers = sorted(set(ds_numbers) | set(general_num_override))
        if not ds_numbers:
            ds_numbers = list(DEFAULT_NUM_VIDEOS)

        configs_for_ds_set: Set[str] = set()
        for configs in summary_stats.values():
            configs_for_ds_set.update(configs.keys())
        configs_for_ds = sorted(configs_for_ds_set | set(general_config_ids))
        general_config_set = set(general_config_ids)
        general_configs = [cfg for cfg in configs_for_ds if cfg in general_config_set]
        if not general_configs:
            general_configs = [cfg for cfg in configs_for_ds if cfg not in ABLATION_CONFIG_SET]
        if not general_configs:
            general_configs = list(general_config_ids)

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
                general_configs,
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
                general_configs,
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
        if plot_delay_comparison_by_config(summary_stats, general_configs, delay_compare_path):
            print(f"[{data_source}] Saved delay comparison plot to {delay_compare_path}")
        else:
            print(f"[{data_source}] Skipped delay comparison plot; no valid data.")

        # Scheduling components comparison across configs
        scheduling_config_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_scheduling_components_by_config.pdf"
        if plot_scheduling_components_by_config(summary_stats, general_configs, scheduling_config_path):
            print(f"[{data_source}] Saved scheduling components by config plot to {scheduling_config_path}")
        else:
            print(f"[{data_source}] Skipped scheduling components by config plot; no valid data.")

        # Scheduling components vs number of videos (base config)
        scheduling_base_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_scheduling_components_base_vs_videos.pdf"
        if plot_scheduling_components_base_vs_videos(summary_stats, ds_numbers, scheduling_base_path):
            print(f"[{data_source}] Saved scheduling components (base) vs videos plot to {scheduling_base_path}")
        else:
            print(f"[{data_source}] Skipped scheduling components (base) vs videos plot; no valid data.")

        # Memory components across configs
        memory_config_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_memory_components_by_config.pdf"
        if plot_memory_components_by_config(summary_stats, general_configs, memory_config_path):
            print(f"[{data_source}] Saved memory components by config plot to {memory_config_path}")
        else:
            print(f"[{data_source}] Skipped memory components by config plot; no valid data.")

        # Memory components vs number of videos (base config)
        memory_base_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_memory_components_base_vs_videos.pdf"
        if plot_memory_components_base_vs_videos(summary_stats, ds_numbers, memory_base_path):
            print(f"[{data_source}] Saved memory components (base) vs videos plot to {memory_base_path}")
        else:
            print(f"[{data_source}] Skipped memory components (base) vs videos plot; no valid data.")

        # Rebuffering ablation comparisons
        available_config_set = set(configs_for_ds)
        for group_key, group_meta in ABLATION_GROUPS.items():
            ablation_path = (
                plot_root_base.parent
                / f"{plot_root_base.name}_{data_source}_{group_meta['slug']}_rebuffer.pdf"
            )
            if plot_rebuffering_ablation_group(
                summary_stats,
                group_key,
                available_config_set,
                ds_numbers,
                group_meta.get("num_videos", ablation_target_numbers),
                ablation_path,
            ):
                print(f"[{data_source}] Saved rebuffering ablation plot ({group_key}) to {ablation_path}")
            else:
                print(f"[{data_source}] Skipped rebuffering ablation plot ({group_key}); no valid data.")

        # Latency components across configs
        latency_config_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_latency_components_by_config.pdf"
        if plot_latency_components_by_config(summary_stats, general_configs, latency_config_path):
            print(f"[{data_source}] Saved latency components by config plot to {latency_config_path}")
        else:
            print(f"[{data_source}] Skipped latency components by config plot; no valid data.")

        # Latency components over number of videos
        latency_videos_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_latency_components_vs_videos.pdf"
        if plot_latency_components_vs_videos(summary_stats, general_configs, ds_numbers, latency_videos_path):
            print(f"[{data_source}] Saved latency components vs videos plot to {latency_videos_path}")
        else:
            print(f"[{data_source}] Skipped latency components vs videos plot; no valid data.")

        # Generation speed for single-video experiments
        generation_speed_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_generation_speed_single_video.pdf"
        if plot_generation_speed_single_video(summary_stats, general_configs, generation_speed_path):
            print(f"[{data_source}] Saved generation speed plot to {generation_speed_path}")
        else:
            print(f"[{data_source}] Skipped generation speed plot; no valid data.")

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

    # Combined round_robin_m comparison across data sources
    if data_sources:
        if args.output:
            base_output = Path(args.output)
            combined_path = base_output.parent / f"{base_output.stem}_roundrobin_rebuffer.pdf"
        else:
            combined_path = base_dir / "roundrobin_m_rebuffer_across_sources.pdf"
        if plot_roundrobin_comparison(
            all_summary_stats,
            data_sources,
            available_numbers,
            general_num_override,
            combined_path,
        ):
            print(f"Saved round_robin_m comparison to {combined_path}")
        else:
            print("Skipped round_robin_m comparison plot; no valid data.")

    # Combined round_robin_2 / round_robin_m ratio comparison across data sources
    if data_sources:
        if args.output:
            base_output = Path(args.output)
            combined_path = base_output.parent / f"{base_output.stem}_roundrobin_ratio.pdf"
        else:
            combined_path = base_dir / "roundrobin_2_over_m_ratio_across_sources.pdf"
        if plot_roundrobin_ratio(
            all_summary_stats,
            data_sources,
            available_numbers,
            general_num_override,
            combined_path,
        ):
            print(f"Saved round_robin_2 / round_robin_m ratio comparison to {combined_path}")
        else:
            print("Skipped round_robin_2 / round_robin_m ratio comparison plot; no valid data.")


if __name__ == "__main__":
    main()
