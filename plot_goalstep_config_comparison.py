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
CONFIG_DISPLAY_NAMES: Dict[str, str] = {
    "base": "Ours",
    "random_2": "RD-S",
    "random_m": "RD-M",
    "round_robin_2": "RR-S",
    "round_robin_m": "RR-M",
}
DEFAULT_NUM_VIDEOS: Tuple[int, ...] = (1,2,3,4,5,6,7,8,9,10)
DEFAULT_ABLATION_NUM_VIDEOS: Tuple[int, ...] = (5, 10, 15)
DEFAULT_ITERATIONS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_BASE_DIR = Path("figures")
BASE_CONFIG_ID = "base"

METRIC_ORDER: Tuple[Tuple[str, str, str, str], ...] = (
    ("rebuffer_time", "Rebuffering (s)", "", "rebuffer"),
    ("ttft", "TTFT (s)", "", "ttft"),
    ("scheduling_score", "Scheduling Score", "", "scheduling"),
    ("perplexity", "Perplexity", "", "perplexity"),
)

LATENCY_COMPONENTS: Tuple[Tuple[str, str, str], ...] = (
    ("visual_embedding_time", "Prefilling", "prefilling"),
    ("model_forward_time", "EOS", "scoring"),
    ("generation_time", "Decoding", "generation"),
    ("kv_offload_time", "Offload", "kv_offload"),
    ("kv_reload_time", "Reload", "kv_reload"),
    ("total_sending_time", "Network", "network_time"),
    ("total_processing_time", "Total", "total_processing"),
)

DELAY_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("processing_delay", "Processing", "processing_delay"),
    ("queuing_delay", "Queuing", "queuing_delay"),
    ("network_delay", "Networking", "network_delay"),
)

SCHEDULING_COMPONENTS: Tuple[Tuple[str, str, str], ...] = (
    ("lowest_buffer_score", "Lowest", "lowest_buffer"),
    ("nonzero_score", "Nonzero", "nonzero"),
    ("ending_score", "Finishing", "ending"),
)

MEMORY_COMPONENTS: Tuple[Tuple[str, str, str], ...] = (
    ("model_params_memory", "Model Params", "model_params"),
    ("combined_dynamic_memory", "KV GPU", "combined_dynamic"),
    ("cpu_memory_growth_peak", "KV CPU", "cpu_growth"),
)
KV_CACHE_MEMORY = "kv_cache_memory"
GENERATION_SPEED_METRIC = "generation_speed"
FRACTION_RESPONSE_FRAMES_METRIC = "fraction_response_frames"
GENERATION_LENGTHS = "generation_lengths"
KV_OFFLOAD_SCATTER = "kv_offload_pairs"
KV_RELOAD_SCATTER = "kv_reload_pairs"
KV_SECONDARY_METRIC = "kv_transfer_per_second"
KV_OFFLOAD_SLOPE = "kv_offload_slope"
KV_RELOAD_SLOPE = "kv_reload_slope"
TIMING_BREAKDOWN_METRIC = "timing_breakdown"
BAR_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 10
DISTRIBUTION_CONFIG_ID = "max_frames_memory_test"

EXTEND_METRICS = {
    GENERATION_LENGTHS,
    KV_OFFLOAD_SCATTER,
    KV_RELOAD_SCATTER,
    KV_SECONDARY_METRIC,
    KV_OFFLOAD_SLOPE,
    KV_RELOAD_SLOPE,
    TIMING_BREAKDOWN_METRIC,
}
GENERATION_LENGTHS = "generation_lengths"

ABLATION_GROUPS = {
    "rl": {
        "title": "Running Length Weight Ablation",
        "configs": [
            ("base", "Default (0.1)"),
            ("rl_ablation1", "Weight 0"),
            ("rl_ablation2", "Weight 1"),
            ("rl_ablation3", "Weight 10"),
        ],
        "num_videos": [5, 10, 15],
        "slug": "rl_ablation",
    },
    "comp": {
        "title": "Computation Modules Ablation",
        "configs": [
            ("base", "Ours"),
            ("comp_ablation1", "RL"),
            ("comp_ablation2", "Age"),
            ("comp_ablation3", "Thr."),
            ("comp_ablation4", "LBF"),
            ("comp_ablation5", "Slice"),
        ],
        "num_videos": [5, 10, 15],
        "slug": "comp_ablation",
    },
    "chunk": {
        "title": "Chunk Size Ablation",
        "configs": [
            ("base", "Default (Chunk 2)"),
            ("chunk_ablation1", "Chunk 1"),
            ("chunk_ablation2", "Chunk 4"),
            ("chunk_ablation3", "Chunk 8"),
            ("chunk_ablation4", "Chunk 16"),
            ("chunk_ablation5", "Chunk 32"),
        ],
        "num_videos": [5, 10, 15],
        "slug": "chunk_ablation",
    },
    "factor": {
        "title": "Factor Ablation",
        "configs": [
            ("base", "Default (0.2)"),
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
        "num_videos": [5, 10, 15],
        "slug": "factor_ablation",
    },
    "consumption": {
        "title": "Consumption Ablation",
        "grouped_configs": [
            ("1", [
                ("consumption_ablation1_base", "Base"),
                ("consumption_ablation1_rr_2", "RR-2"),
                ("consumption_ablation1_rr_m", "RR-m"),
                ("consumption_ablation1_rand_2", "Rand-2"),
                ("consumption_ablation1_rand_m", "Rand-m"),
            ]),
            ("2", [
                ("consumption_ablation2_base", "Base"),
                ("consumption_ablation2_rr_2", "RR-2"),
                ("consumption_ablation2_rr_m", "RR-m"),
                ("consumption_ablation2_rand_2", "Rand-2"),
                ("consumption_ablation2_rand_m", "Rand-m"),
            ]),
            ("3", [
                ("base", "Base"),
                ("round_robin_2", "RR-2"),
                ("round_robin_m", "RR-m"),
                ("random_2", "Rand-2"),
                ("random_m", "Rand-m"),
            ]),
            ("4", [
                ("consumption_ablation3_base", "Base"),
                ("consumption_ablation3_rr_2", "RR-2"),
                ("consumption_ablation3_rr_m", "RR-m"),
                ("consumption_ablation3_rand_2", "Rand-2"),
                ("consumption_ablation3_rand_m", "Rand-m"),
            ]),
            ("5", [
                ("consumption_ablation4_base", "Base"),
                ("consumption_ablation4_rr_2", "RR-2"),
                ("consumption_ablation4_rr_m", "RR-m"),
                ("consumption_ablation4_rand_2", "Rand-2"),
                ("consumption_ablation4_rand_m", "Rand-m"),
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

    def _is_true_ablation(config_id: str) -> bool:
        """Treat entries containing 'ablation' as the ablation-specific configs."""
        return "ablation" in config_id

    if "configs" in group_meta:
        for cfg, _ in group_meta["configs"]:
            if cfg != "base" and _is_true_ablation(cfg):
                ABLATION_CONFIG_SET.add(cfg)
                ABLATION_CONFIG_ALLOWED_NUMS[cfg] = allowed_nums.copy()

    if "grouped_configs" in group_meta:
        for _, cfgs in group_meta["grouped_configs"]:
            for cfg, _ in cfgs:
                if cfg != "base" and _is_true_ablation(cfg):
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
        "--comparison-base-dir",
        type=Path,
        default=None,
        help="Optional secondary root (e.g., ../results-3090) for cross-server comparisons.",
    )
    parser.add_argument(
        "--primary-label",
        type=str,
        default=None,
        help="Label for the primary results (default: base directory name).",
    )
    parser.add_argument(
        "--comparison-label",
        type=str,
        default="3090",
        help="Label for the comparison results used in cross-server plots.",
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
        default=list(DEFAULT_NUM_VIDEOS),
        help="Number of videos for general (non-ablation) plots. Auto-detect when omitted.",
    )
    parser.add_argument(
        "--baseline-comparison-nums",
        nargs="+",
        type=int,
        default=list(DEFAULT_NUM_VIDEOS),
        help="Number of videos used when averaging for the cross-server baseline comparison.",
    )
    parser.add_argument(
        "--ablation-num-videos",
        nargs="+",
        type=int,
        default=list(DEFAULT_ABLATION_NUM_VIDEOS),
        help="Number of videos for ablation plots (default: 5 10 15).",
    )
    parser.add_argument(
        "--per-video-numbers",
        nargs="+",
        type=int,
        default=[3, 5, 8, 10],
        help="Video counts to display in per-video plots.",
    )
    parser.add_argument(
        "--overall-video-numbers",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Video counts to include when aggregating across videos.",
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


def load_iteration_metrics(summary_path: Path, num_videos: Optional[int] = None) -> Dict[str, float]:
    """Extract metrics from a single overall_summary.json file, optionally normalizing per video."""
    with summary_path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)

    scaling: float = 1.0
    if num_videos:
        try:
            num_videos_int = int(num_videos)
        except (TypeError, ValueError):
            num_videos_int = 0
        if num_videos_int > 0:
            scaling = 1.0 / float(num_videos_int)

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
        rebuffer_values.append(float(total) * scaling)

        if candidate.get("processing_delays"):
            processing_delays.append(float(candidate["processing_delays"][-1]) * scaling)
        elif candidate.get("total_processing_delay") is not None:
            processing_delays.append(float(candidate["total_processing_delay"]) * scaling)

        if candidate.get("queuing_delays"):
            queuing_delays.append(float(candidate["queuing_delays"][-1]) * scaling)
        elif candidate.get("total_queuing_delay") is not None:
            queuing_delays.append(float(candidate["total_queuing_delay"]) * scaling)

        if candidate.get("networking_delays"):
            networking_delays.append(float(candidate["networking_delays"][-1]) * scaling)
        elif candidate.get("total_networking_delay") is not None:
            networking_delays.append(float(candidate["total_networking_delay"]) * scaling)
    rebuffer_time = float(np.mean(rebuffer_values)) if rebuffer_values else float("nan")

    # TTFT: average per conversation response latency when prompts exist.
    component_samples: Dict[str, List[float]] = {key: [] for key, _, _ in LATENCY_COMPONENTS}
    component_per_frame_samples: Dict[str, List[float]] = {
        key: [] for key, _, _ in LATENCY_COMPONENTS
    }
    results = summary.get("results", []) or []
    ttft_samples: List[float] = []
    ppl_samples: List[float] = []
    total_generated_words = 0.0
    total_generation_time = 0.0
    total_nonzero_turns = 0
    total_frames = 0
    generation_lengths: List[int] = []
    kv_pairs_offload: List[Tuple[float, float]] = []
    kv_pairs_reload: List[Tuple[float, float]] = []
    kv_per_second: List[List[Tuple[float, float]]] = []
    kv_cache_max_values: List[float] = []
    offload_slopes: List[float] = []
    reload_slopes: List[float] = []
    timing_breakdowns: List[Tuple[float, float, float, float, float, float, float]] = []
    for result in results:
        num_frames = int(result.get("num_frames", 0) or 0)
        total_frames += max(num_frames, 0)
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
                denom = max(num_frames, 1)
                component_per_frame_samples[key].append(float(value) / denom)

        for turn in result.get("generated_turns", []):
            text = turn.get("text", "")
            if "Assistant:" in text:
                text = text.split("Assistant:", 1)[-1]
            word_count = len([w for w in text.strip().split() if w])
            total_generated_words += word_count
            total_generation_time += float(turn.get("generation_time", 0.0))
            if word_count > 0:
                total_nonzero_turns += 1
            generation_lengths.append(word_count)
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
        transfer_sizes = memory_data.get("kv_transfer_size") or []
        offload_times = memory_data.get("kv_offload_time") or []
        reload_times = memory_data.get("kv_reload_time") or []

        if model_mem:
            model_params_values.append(float(max(model_mem)))
        if kv_mem:
            kv_cache_max_values.append(float(max(kv_mem)))
        if kv_mem and activation_mem and other_mem:
            combined_dynamic_values.append(
                float(max(kv_mem)) + float(max(activation_mem)) + float(max(other_mem))
            )
        if cpu_mem_growth:
            cpu_growth_values.append(float(max(cpu_mem_growth)))

        for size, off_t in zip(transfer_sizes, offload_times):
            if off_t and math.isfinite(off_t):
                kv_pairs_offload.append((float(size), float(off_t)))
        for size, reload_t in zip(transfer_sizes, reload_times):
            if reload_t and math.isfinite(reload_t):
                kv_pairs_reload.append((float(size), float(reload_t)))
        combined = []
        frames = min(len(transfer_sizes), len(offload_times), len(reload_times))
        for idx in range(frames):
            size = float(transfer_sizes[idx])
            total_time = float(offload_times[idx]) + float(reload_times[idx])
            if math.isfinite(size) and math.isfinite(total_time):
                combined.append((size, total_time))
        if combined:
            per_second = []
            for i in range(0, len(combined) - 1, 2):
                size_total = combined[i][0] + combined[i + 1][0]
                time_total = (combined[i][1] + combined[i + 1][1]) * 1000.0
                per_second.append((size_total, time_total))
            if per_second:
                kv_per_second.append(per_second)

        if len(offload_times) > 1:
            indices = np.arange(len(offload_times))
            slope, _ = np.polyfit(indices, [float(t) * 1000.0 for t in offload_times], 1)
            if math.isfinite(slope):
                offload_slopes.append(float(slope))
        if len(reload_times) > 1:
            indices = np.arange(len(reload_times))
            slope, _ = np.polyfit(indices, [float(t) * 1000.0 for t in reload_times], 1)
            if math.isfinite(slope):
                reload_slopes.append(float(slope))

    processing_delay = float(np.mean(processing_delays)) if processing_delays else float("nan")
    queuing_delay = float(np.mean(queuing_delays)) if queuing_delays else float("nan")
    networking_delay = float(np.mean(networking_delays)) if networking_delays else float("nan")

    component_means = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in component_samples.items()
    }
    component_per_frame_means = {
        f"{key}_per_frame": float(np.mean(values)) if values else float("nan")
        for key, values in component_per_frame_samples.items()
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
            KV_CACHE_MEMORY: float(np.mean(kv_cache_max_values)) if kv_cache_max_values else float("nan"),
        }
    )
    if total_generation_time > 0:
        metrics[GENERATION_SPEED_METRIC] = total_generated_words / total_generation_time
    else:
        metrics[GENERATION_SPEED_METRIC] = float("nan")
    if total_frames > 0:
        metrics[FRACTION_RESPONSE_FRAMES_METRIC] = total_nonzero_turns / total_frames
    else:
        metrics[FRACTION_RESPONSE_FRAMES_METRIC] = float("nan")
    metrics[GENERATION_LENGTHS] = generation_lengths
    metrics[KV_OFFLOAD_SCATTER] = kv_pairs_offload
    metrics[KV_RELOAD_SCATTER] = kv_pairs_reload
    flattened_pairs = [pair for per_second in kv_per_second for pair in per_second]
    metrics[KV_SECONDARY_METRIC] = flattened_pairs
    # Debug for slopes
    # print(f"[DEBUG] slopes collected off={len(offload_slopes)} reload={len(reload_slopes)}")
    metrics[KV_OFFLOAD_SLOPE] = offload_slopes
    metrics[KV_RELOAD_SLOPE] = reload_slopes
    metrics[TIMING_BREAKDOWN_METRIC] = timing_breakdowns
    metrics.update(component_means)
    metrics.update(component_per_frame_means)
    return metrics


# ------------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------------

def configure_plot_style(font_size: int = 10) -> None:
    """Set matplotlib defaults to a publication-friendly style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
        }
    )


def _display_config_name(config_id: str) -> str:
    """Return human-friendly label for a config id."""
    return CONFIG_DISPLAY_NAMES.get(config_id, config_id.replace("_", " ").title())


def _scientific_colors() -> List[str]:
    """Return a color palette suited for scientific plots."""
    return [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A10F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]


def _color_hatch_cycle(count: int) -> List[Tuple[str, str]]:
    """Generate consistent (color, hatch) pairs for bar/stacked plots."""
    colors = _scientific_colors()
    hatches = ["/", "\\", "x", "-", "+", "o", ".", "*", "//", "||"]
    palette: List[Tuple[str, str]] = []
    for idx in range(max(1, count)):
        palette.append((colors[idx % len(colors)], hatches[idx % len(hatches)]))
    return palette


def _collect_per_video_means(
    num_map: Dict[int, Dict[str, Iterable[float]]],
    allowed_nums: Optional[Iterable[int]] = None,
) -> List[float]:
    values: List[float] = []
    numbers = allowed_nums if allowed_nums is not None else sorted(num_map)
    for num in numbers:
        entry = num_map.get(num)
        if not entry:
            continue
        mean = entry.get("mean")
        if mean is not None and math.isfinite(mean):
            values.append(float(mean))
    return values


def _aggregate_across_videos(
    num_map: Dict[int, Dict[str, Iterable[float]]],
    allowed_nums: Optional[Iterable[int]] = None,
) -> Tuple[float, float]:
    values = _collect_per_video_means(num_map, allowed_nums)
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


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
    configure_plot_style(font_size=10)
    palette = _color_hatch_cycle(len(config_ids))
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
        stds_arr[~np.isfinite(stds_arr)] = 0.0
        if np.all(~np.isfinite(means_arr)):
            continue

        has_data = True
        color, hatch = palette[idx]
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=_display_config_name(config_id),
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
    allowed_nums: Optional[Iterable[int]] = None,
) -> bool:
    """Bar chart summarizing mean/std aggregated across all video counts."""
    configure_plot_style()
    palette = _color_hatch_cycle(len(config_ids))
    means = []
    stds = []
    config_labels = []

    for idx, config_id in enumerate(config_ids):
        num_map = summary_stats.get(metric_key, {}).get(config_id, {})
        mean, std = _aggregate_across_videos(num_map, allowed_nums)
        if not math.isfinite(mean):
            continue
        means.append(mean)
        stds.append(std if math.isfinite(std) else 0.0)
        config_labels.append(_display_config_name(config_id))

    if not means:
        return False

    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for idx, (mean, std) in enumerate(zip(means, stds)):
        color, hatch = palette[idx]
        ax.bar(
            [x[idx]],
            [mean],
            yerr=[std],
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
        )
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
    """Bar plot of base config delay metrics vs number of videos."""
    if not num_videos:
        return False

    base_stats = {
        metric: summary_stats.get(metric, {}).get(BASE_CONFIG_ID, {}) for metric, *_ in DELAY_METRICS
    }
    if not any(base_stats.values()):
        return False

    configure_plot_style(font_size=10)
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    palette = _color_hatch_cycle(len(DELAY_METRICS))
    x = np.arange(len(num_videos))
    bar_width = min(0.8 / max(len(DELAY_METRICS), 1), 0.22)
    has_data = False

    for idx, (metric_key, _, title) in enumerate(DELAY_METRICS):
        metric_data = base_stats.get(metric_key, {})
        means = []
        stds = []
        for n in num_videos:
            entry = metric_data.get(n, {})
            mean = entry.get("mean")
            std = entry.get("std")
            if mean is not None and math.isfinite(mean):
                means.append(float(mean))
                stds.append(float(std) if std is not None and math.isfinite(std) else 0.0)
            else:
                means.append(float("nan"))
                stds.append(0.0)
        means_arr = np.asarray(means, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        has_data = True
        color, hatch = palette[idx]
        offset = (idx - (len(DELAY_METRICS) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=title,
        )

    if not has_data:
        plt.close(fig)
        return False

    ax.set_title(f"{BASE_CONFIG_ID.title()} Delay vs Number of Videos", fontsize=10)
    ax.set_xlabel("Number of Videos", fontsize=10)
    ax.set_ylabel("Delay (s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in num_videos], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_delay_comparison_by_config(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
    allowed_nums: Optional[Iterable[int]] = None,
) -> bool:
    """Grouped bar comparing delay metrics averaged over all videos per config."""
    configure_plot_style()
    delay_keys = [metric for metric, *_ in DELAY_METRICS]
    palette = _color_hatch_cycle(len(delay_keys))
    collected = []
    config_labels = []

    for config_id in config_ids:
        per_metric_means = []
        has_values = False
        for metric_key in delay_keys:
            num_map = summary_stats.get(metric_key, {}).get(config_id, {})
            values = _collect_per_video_means(num_map, allowed_nums)
            if values:
                mean = float(np.mean(values))
                has_values = True
            else:
                mean = float("nan")
            per_metric_means.append(mean)
        if has_values:
            collected.append(per_metric_means)
            config_labels.append(_display_config_name(config_id))

    if not collected:
        return False

    arr = np.asarray(collected, dtype=float)
    x = np.arange(len(config_labels))
    bar_width = min(0.8 / len(delay_keys), 0.2)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for idx, (_, title, _) in enumerate(DELAY_METRICS):
        color, hatch = palette[idx]
        ax.bar(
            x + (idx - (len(delay_keys) - 1) / 2) * bar_width,
            arr[:, idx],
            width=bar_width * 0.95,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=title,
        )

    # show numbers on bars
    for idx, (_, _, title) in enumerate(DELAY_METRICS):
        color, hatch = palette[idx]
        for i in range(len(config_labels)):
            ax.text(x[i] + (idx - (len(delay_keys) - 1) / 2) * bar_width, arr[i, idx], f"{arr[i, idx]:.2f}", ha="center", va="bottom", color=color, rotation=90)

    ax.set_ylabel("Rebuffering (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels)
    ax.set_xlabel("Scheduler")
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
    allowed_nums: Optional[Iterable[int]] = None,
) -> bool:
    """Compare latency components averaged across all videos for each config."""
    configure_plot_style()
    palette = _color_hatch_cycle(len(config_ids))
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
            collected = _collect_per_video_means(num_map, allowed_nums)
            if collected:
                arr = np.asarray(collected, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=0))
            else:
                mean, std = (float("nan"), float("nan"))
            means.append(mean)
            stds.append(std)

        means_arr = np.asarray(means, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue

        has_data = True
        offset = (idx - (num_configs - 1) / 2) * bar_width
        color, hatch = palette[idx]
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=_display_config_name(config_id),
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


def plot_scheduling_components_by_config(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    output_path: Path,
    allowed_nums: Optional[Iterable[int]] = None,
) -> bool:
    """Grouped bar chart comparing scheduling components across configs."""
    configure_plot_style()
    component_keys = [key for key, _, _ in SCHEDULING_COMPONENTS]
    component_labels = [label for _, label, _ in SCHEDULING_COMPONENTS]
    palette = _color_hatch_cycle(len(component_labels))

    means_matrix: List[List[float]] = []
    std_matrix: List[List[float]] = []
    valid_configs: List[str] = []

    for config_id in config_ids:
        config_means = []
        config_stds = []
        has_data = False
        for key in component_keys:
            num_map = summary_stats.get(key, {}).get(config_id, {})
            collected = _collect_per_video_means(num_map, allowed_nums)
            if collected:
                arr = np.asarray(collected, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=0))
                has_data = True
            else:
                mean, std = float("nan"), float("nan")
            config_means.append(mean)
            config_stds.append(std)
        if has_data:
            valid_configs.append(_display_config_name(config_id))
            means_matrix.append(config_means)
            std_matrix.append(config_stds)

    if not means_matrix:
        return False

    x = np.arange(len(valid_configs))
    bar_width = min(0.8 / len(component_keys), 0.2)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    palette = _color_hatch_cycle(len(component_labels))

    for idx, label in enumerate(component_labels):
        color, hatch = palette[idx]
        means_arr = np.asarray([row[idx] for row in means_matrix], dtype=float)
        stds_arr = np.asarray([row[idx] for row in std_matrix], dtype=float)
        stds_arr[~np.isfinite(stds_arr)] = 0.0
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
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

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
    """Bar plot showing scheduling components over number of videos for base config."""
    if not num_videos:
        return False

    base_components = {
        key: summary_stats.get(key, {}).get(BASE_CONFIG_ID, {})
        for key, _, _ in SCHEDULING_COMPONENTS
    }

    if not any(base_components.values()):
        return False

    configure_plot_style(font_size=10)
    palette = _color_hatch_cycle(len(num_videos))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(SCHEDULING_COMPONENTS))
    bar_width = min(0.8 / max(len(num_videos), 1), 0.22)
    has_data = False

    for idx, n in enumerate(num_videos):
        means = []
        stds = []
        for key, _, label in SCHEDULING_COMPONENTS:
            entry = base_components.get(key, {}).get(n, {})
            mean = entry.get("mean")
            std = entry.get("std")
            if mean is not None and math.isfinite(mean):
                means.append(float(mean))
                stds.append(float(std) if std is not None and math.isfinite(std) else 0.0)
            else:
                means.append(float("nan"))
                stds.append(0.0)
        means_arr = np.asarray(means, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        has_data = True
        color, hatch = palette[idx]
        offset = (idx - (len(num_videos) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=f"N={n}",
        )

    if not has_data:
        plt.close(fig)
        return False

    ax.set_ylabel("Cumulative Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in SCHEDULING_COMPONENTS], rotation=20, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=1, fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_memory_breakdown_multi_videos(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_id: str,
    target_num_videos: Iterable[int],
    output_path: Path,
) -> bool:
    """Visualize memory components across multiple video counts for a single config."""
    target_nums = [int(n) for n in target_num_videos]
    component_defs = [
        ("model_params_memory", "Params"),
        ("combined_dynamic_memory", "KV (GPU)"),
        ("cpu_memory_growth_peak", "KV (CPU)"),
    ]

    def _extract(metric_key: str, num: int) -> Tuple[float, float, List[float]]:
        entry = (
            summary_stats.get(metric_key, {})
            .get(config_id, {})
            .get(num, {})
        )
        mean = entry.get("mean")
        std = entry.get("std")
        values = entry.get("values", [])
        clean_values: List[float] = []
        for v in values:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(vf):
                clean_values.append(vf)
        return (
            float(mean) if mean is not None and math.isfinite(mean) else float("nan"),
            float(std) if std is not None and math.isfinite(std) else float("nan"),
            clean_values,
        )

    component_stats: Dict[str, Dict[int, Tuple[float, float]]] = {}
    model_values_cache: Dict[int, List[float]] = {}
    dynamic_values_cache: Dict[int, List[float]] = {}

    for metric_key, label in component_defs:
        per_num: Dict[int, Tuple[float, float, List[float]]] = {}
        for num in target_nums:
            mean, std, values = _extract(metric_key, num)
            per_num[num] = (mean, std, values)
            if metric_key == "model_params_memory":
                model_values_cache[num] = values
            if metric_key == "combined_dynamic_memory":
                dynamic_values_cache[num] = values
        component_stats[label] = {
            num: (data[0], data[1]) for num, data in per_num.items()
        }

    # GPU total computed from raw values when available, otherwise mean sum fallback.
    gpu_totals: Dict[int, Tuple[float, float]] = {}
    for num in target_nums:
        combined = []
        model_vals = model_values_cache.get(num, [])
        dyn_vals = dynamic_values_cache.get(num, [])
        if model_vals and dyn_vals and len(model_vals) == len(dyn_vals):
            for m, d in zip(model_vals, dyn_vals):
                if math.isfinite(m) and math.isfinite(d):
                    combined.append(m + d)
        if combined:
            arr = np.asarray(combined, dtype=float)
            gpu_totals[num] = (float(arr.mean()), float(arr.std(ddof=0)))
        else:
            model_mean, model_std = component_stats["Params"].get(num, (float("nan"), float("nan")))
            dyn_mean, dyn_std = component_stats["KV (GPU)"].get(num, (float("nan"), float("nan")))
            if math.isfinite(model_mean) and math.isfinite(dyn_mean):
                combined_mean = model_mean + dyn_mean
                # simple std combination if available
                if math.isfinite(model_std) and math.isfinite(dyn_std):
                    combined_std = math.sqrt(model_std ** 2 + dyn_std ** 2)
                else:
                    combined_std = float("nan")
                gpu_totals[num] = (combined_mean, combined_std)
            else:
                gpu_totals[num] = (float("nan"), float("nan"))

    component_stats["GPU Total"] = gpu_totals

    # Filter out components lacking any finite data
    components_order = ["Params", "KV (GPU)", "GPU Total", "KV (CPU)"]
    components = []
    for label in components_order:
        stats_per_num = component_stats.get(label, {})
        if any(math.isfinite(stats_per_num.get(num, (float("nan"),))[0]) for num in target_nums):
            components.append(label)

    if not components:
        return False

    means_matrix = []
    std_matrix = []
    for label in components:
        means = []
        stds = []
        stats_per_num = component_stats.get(label, {})
        for num in target_nums:
            mean, std = stats_per_num.get(num, (float("nan"), float("nan")))
            means.append(mean)
            stds.append(std)
        means_matrix.append(means)
        std_matrix.append(stds)

    configure_plot_style()
    palette = _color_hatch_cycle(len(target_nums))

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(components))
    num_groups = len(target_nums)
    bar_width = min(0.8 / max(num_groups, 1), 0.22)

    for idx, num in enumerate(target_nums):
        color, hatch = palette[idx]
        offset = (idx - (num_groups - 1) / 2) * bar_width
        means = [means_matrix[comp_idx][idx] for comp_idx in range(len(components))]
        stds = [std_matrix[comp_idx][idx] for comp_idx in range(len(components))]
        means_arr = np.asarray(means, dtype=float)
        if np.all(~np.isfinite(means_arr)):
            continue
        stds_arr = np.asarray(stds, dtype=float)
        stds_arr[~np.isfinite(stds_arr)] = 0.0
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.9,
            yerr=stds_arr,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=f"N={num}",
        )

    ax.set_ylabel("Peak Memory (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.grid(axis="y", alpha=0.3)
    positive_values = [
        val for row in means_matrix for val in row if math.isfinite(val) and val > 0
    ]
    if positive_values:
        min_pos = min(positive_values)
        max_pos = max(positive_values)
        ax.set_yscale("log")
        ax.set_ylim(min_pos / 1.8, max_pos * 1.6)
    ax.legend(frameon=False, ncol=1)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_memory_speed_ratios(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_id: str,
    target_num_videos: Iterable[int],
    output_path: Path,
) -> bool:
    target_nums = [int(n) for n in target_num_videos]
    if not target_nums:
        return False

    cpu_ratios: List[float] = []
    kv_ratios: List[float] = []
    consumption_ratios: List[float] = []

    for num in target_nums:
        cpu_mean = (
            summary_stats.get("cpu_memory_growth_peak", {})
            .get(config_id, {})
            .get(num, {})
            .get("mean")
        )
        cpu_ratio = float("nan")
        if cpu_mean is not None and math.isfinite(cpu_mean):
            cpu_ratio = cpu_mean / (64 * 1024)  # convert MB ratio
        cpu_ratios.append(cpu_ratio)

        model_mean = (
            summary_stats.get("model_params_memory", {})
            .get(config_id, {})
            .get(num, {})
            .get("mean")
        )
        kv_mean = (
            summary_stats.get(KV_CACHE_MEMORY, {})
            .get(config_id, {})
            .get(num, {})
            .get("mean")
        )
        kv_ratio = float("nan")
        if (
            kv_mean is not None
            and math.isfinite(kv_mean)
            and model_mean is not None
            and math.isfinite(model_mean)
        ):
            available_gpu_mb = 24 * 1024 - model_mean
            if available_gpu_mb > 0:
                kv_ratio = kv_mean / available_gpu_mb
        kv_ratios.append(kv_ratio)

        gen_speed = (
            summary_stats.get(GENERATION_SPEED_METRIC, {})
            .get(config_id, {})
            .get(num, {})
            .get("mean")
        )
        consumption_ratio = float("nan")
        if gen_speed is not None and math.isfinite(gen_speed) and gen_speed > 0:
            consumption_speed = num * 2.7  # words/sec
            consumption_ratio = consumption_speed / gen_speed
        consumption_ratios.append(consumption_ratio)

    if all(
        not any(math.isfinite(val) for val in ratio_list)
        for ratio_list in (cpu_ratios, kv_ratios, consumption_ratios)
    ):
        return False

    categories = ["CPU", "GPU", "Throughput"]
    ratio_matrix = [cpu_ratios, kv_ratios, consumption_ratios]
    # convert to percentages
    ratio_matrix = [[val * 100 for val in row] for row in ratio_matrix]

    configure_plot_style()
    palette = _color_hatch_cycle(len(target_nums))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(categories))
    num_groups = len(target_nums)
    bar_width = min(0.8 / max(num_groups, 1), 0.22)

    plotted = False
    for idx, num in enumerate(target_nums):
        color, hatch = palette[idx]
        offset = (idx - (num_groups - 1) / 2) * bar_width
        values = [ratio_matrix[row][idx] for row in range(len(categories))]
        values_arr = np.asarray(values, dtype=float)
        if np.all(~np.isfinite(values_arr)):
            continue
        plotted = True
        ax.bar(
            x + offset,
            values_arr,
            width=bar_width * 0.9,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=f"N={num}",
        )
        for xpos, value, label in zip(x + offset, values_arr, categories):
            if label not in {"CPU", "GPU"} or not math.isfinite(value):
                continue
            ax.text(
                xpos,
                value + 0.03,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONT_SIZE,
                rotation=90,
            )

    if not plotted:
        plt.close(fig)
        return False
        
    ax.set_ylabel("Ratio (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ylim_upper = max(
        [value for row in ratio_matrix for value in row if math.isfinite(value)]
        + [1.0]
    )
    ax.set_ylim(0, ylim_upper * 1.2)
    ax.legend(frameon=False, ncol=1, fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_rebuffer_baseline_comparison_across_bases(
    primary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    comparison_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    target_nums: List[int],
    primary_label: str,
    comparison_label: str,
    output_path: Path,
) -> bool:
    """Compare average rebuffering across baseline configs for two result sets."""
    target_nums = sorted(set(target_nums))
    if not target_nums:
        return False

    def aggregate(
        stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
        config_id: str,
    ) -> Tuple[float, float]:
        config_stats = stats.get("rebuffer_time", {}).get(config_id, {})
        values: List[float] = []
        for num in target_nums:
            entry = config_stats.get(num, {})
            mean = entry.get("mean")
            if mean is not None and math.isfinite(mean):
                values.append(float(mean))
        if not values:
            return float("nan"), float("nan")
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    dataset_entries = []
    for label, stats in (
        (primary_label, primary_stats),
        (comparison_label, comparison_stats),
    ):
        means = []
        stds = []
        for cfg in config_ids:
            mean, std = aggregate(stats, cfg)
            means.append(mean)
            stds.append(std if math.isfinite(std) else 0.0)
        dataset_entries.append((label, means, stds))

    valid_indices = [
        idx
        for idx in range(len(config_ids))
        if any(math.isfinite(entry[1][idx]) for entry in dataset_entries)
    ]
    if not valid_indices:
        return False

    configure_plot_style()
    palette = _color_hatch_cycle(len(dataset_entries))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(valid_indices))
    bar_width = 0.35

    plotted = False
    for dataset_idx, (label, means, stds) in enumerate(dataset_entries):
        color, hatch = palette[dataset_idx]
        dataset_means = [means[i] for i in valid_indices]
        dataset_stds = [stds[i] for i in valid_indices]
        if all(not math.isfinite(m) for m in dataset_means):
            continue
        plotted = True
        offset = (dataset_idx - (len(dataset_entries) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            dataset_means,
            width=bar_width * 0.95,
            yerr=dataset_stds,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

        for xpos, value in zip(x + offset, dataset_means):
            if math.isfinite(value):
                ax.text(
                    xpos,
                    value + 0.02 * max(value, 1.0),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_FONT_SIZE,
                    rotation=90,
                )

    if not plotted:
        plt.close(fig)
        return False

    cfg_labels = [_display_config_name(config_ids[i]) for i in valid_indices]
    nums_str = ", ".join(str(n) for n in target_nums)
    ax.set_ylabel("Rebuffering (s)", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, rotation=20, ha="right", fontsize=10)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)

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
    """Plot goalstep round_robin_m rebuffering and delay components across users."""
    summary_stats = general_stats.get("goalstep")
    if not summary_stats:
        return False

    target_numbers = list(range(1, 11))
    series_def = [
        ("rebuffer_time", "Total"),
        ("processing_delay", "Processing"),
        ("queuing_delay", "Queuing"),
        ("network_delay", "Networking"),
    ]

    data_series: List[Tuple[str, List[int], List[float], List[float]]] = []
    for metric_key, label in series_def:
        metric_stats = summary_stats.get(metric_key, {}).get("round_robin_m", {})
        if not metric_stats:
            continue
        nums: List[int] = []
        means: List[float] = []
        stds: List[float] = []
        for n in target_numbers:
            entry = metric_stats.get(n, {})
            mean = entry.get("mean")
            if mean is None or not math.isfinite(mean):
                continue
            std = entry.get("std")
            nums.append(n)
            means.append(float(mean))
            stds.append(float(std) if std is not None and math.isfinite(std) else 0.0)
        if nums:
            data_series.append((label, nums, means, stds))

    if not data_series:
        return False

    configure_plot_style()
    colors = _scientific_colors()
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    for idx, (label, nums, means, stds) in enumerate(data_series):
        ax.errorbar(
            nums,
            means,
            yerr=stds,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linewidth=1.6,
            markersize=4,
            capsize=3,
            label=label,
        )

    ax.set_xlabel("Number of Videos", fontsize=10)
    ax.set_ylabel("RebufferingTime (s)", fontsize=10)
    ax.set_xticks(target_numbers)
    # set xtick fontsize to 10
    ax.tick_params(axis='x', labelsize=10)
    # set ytick fontsize to 10
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.3)
    # set legend fontsize to 10
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

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
    """Plot ratio of round_robin_2 to round_robin_m rebuffering vs number of videos for goalstep."""
    if "goalstep" not in data_sources:
        return False

    summary_stats = general_stats.get("goalstep")
    if not summary_stats:
        return False

    m_stats = summary_stats.get("rebuffer_time", {}).get("round_robin_m", {})
    rr2_stats = summary_stats.get("rebuffer_time", {}).get("round_robin_2", {})
    if not m_stats or not rr2_stats:
        return False

    target_numbers = list(range(1, 11))
    ratios = []
    stds = []
    nums = []
    for n in target_numbers:
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
        ratio = mean_rr2 / mean_m
        std_m = entry_m.get("std")
        std_rr2 = entry_rr2.get("std")
        ratio_std = 0.0
        if std_m is not None and std_rr2 is not None and math.isfinite(std_m) and math.isfinite(std_rr2):
            ratio_std = ratio * math.sqrt(
                (std_rr2 / mean_rr2) ** 2 if mean_rr2 else 0.0
                + (std_m / mean_m) ** 2
            )
        nums.append(n)
        ratios.append(ratio)
        stds.append(ratio_std)

    if not ratios:
        return False

    configure_plot_style()
    colors = _scientific_colors()
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    ax.errorbar(
        nums,
        ratios,
        yerr=stds,
        marker=markers[0],
        color=colors[0],
        linewidth=1.6,
        markersize=4,
        capsize=3,
        label="RR-2 / RR-m",
    )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(
        target_numbers[0],
        1.3,
        "Slicing benefits\ndiminish",
        ha="left",
        va="bottom",
        fontsize=10,
        color="red",
    )

    ax.set_xlabel("Number of Videos", fontsize=10)
    ax.set_ylabel("Rebuffering Ratio", fontsize=10)
    # set x tick fontsize to 10
    ax.tick_params(axis='x', labelsize=10)
    # set y tick fontsize to 10
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xticks(target_numbers)
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_generation_speed_vs_listening(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_id: str,
    output_path: Path,
) -> bool:
    """Compare base generation speed with listening benchmarks."""
    speed_stats = summary_stats.get(GENERATION_SPEED_METRIC, {}).get(config_id, {})
    speeds: List[float] = []
    for num_data in speed_stats.values():
        mean = num_data.get("mean")
        if mean is not None and math.isfinite(mean):
            speeds.append(float(mean))

    if not speeds:
        return False

    base_mean = float(np.mean(speeds))
    base_std = float(np.std(speeds))

    studies = [
        ("Kuperman et al. 2021", 270 / 60.0),
        ("Sommers et al. 2009", 240 / 60.0),
        ("OAText 2017", 124 / 60.0),
        ("Adams & Weber 2006", ((160 + 207) / 2) / 60.0),
    ]

    labels = [s[0] for s in studies] + ["VLM Generation"]
    values = [s[1] for s in studies] + [base_mean]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#F28E2B"]

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.6)

    for bar, value, label in zip(bars, values, labels):
        if label == "VLM Generation":
            text = f"{value:.2f}±{base_std:.2f}"
        else:
            text = f"{value:.2f}"
        ax.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            fontsize=BAR_LABEL_FONT_SIZE,
        )

    ax.set_xlabel("Words Per Second", fontsize=10)
    ax.set_ylabel("Listening Speed Study", fontsize=10)
    max_val = max(values)
    ax.set_xlim(0, max(max_val + 0.8, 5.5))
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    # set ytick size to 10
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _collect_distribution_values(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    metric_key: str,
    config_id: str,
) -> List[float]:
    config_stats = summary_stats.get(metric_key, {}).get(config_id, {})
    values: List[float] = []
    for num_map in config_stats.values():
        raw = num_map.get("values", [])
        values.extend(float(v) for v in raw if v is not None and math.isfinite(v))
    return values


def _plot_cdf_for_sources(
    summary_stats_by_source: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]],
    metric_key: str,
    xlabel: str,
    output_path: Path,
    config_id: str,
    x_limits: Optional[Tuple[float, float]] = None,
) -> bool:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    colors = _scientific_colors()

    plotted = False
    annotations: List[Tuple[float, str, str]] = []
    for idx, data_source in enumerate(sorted(summary_stats_by_source.keys())):
        summary_stats = summary_stats_by_source[data_source]
        values = _collect_distribution_values(summary_stats, metric_key, config_id)
        if not values:
            continue
        values_arr = np.sort(np.asarray(values, dtype=float))
        if values_arr.size == 0:
            continue
        cdf = np.arange(1, values_arr.size + 1, dtype=float) / values_arr.size
        color = colors[idx % len(colors)]
        mean_val = float(values_arr.mean())
        ax.step(
            values_arr,
            cdf,
            where="post",
            color=color,
            linewidth=1.6,
            label=data_source.title(),
        )
        annotations.append((mean_val, data_source.title(), color))
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)

    for idx, (mean_val, label, color) in enumerate(annotations):
        if x_limits is not None and (mean_val < x_limits[0] or mean_val > x_limits[1]):
            continue
        ax.axvline(mean_val, linestyle="--", linewidth=1.0, color=color, alpha=0.6)
        ax.text(
            mean_val,
            0.75 - idx * 0.5,
            f"{label} mean\n{mean_val:.1f}",
            ha="left",
            va="center",
            fontsize=BAR_LABEL_FONT_SIZE,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7),
        )

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_fraction_response_distribution(
    summary_stats_by_source: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]],
    config_id: str,
    output_path: Path,
) -> bool:
    return _plot_cdf_for_sources(
        summary_stats_by_source,
        FRACTION_RESPONSE_FRAMES_METRIC,
        "Fraction of Frames with Nonzero Response",
        output_path,
        config_id,
        x_limits=(0.0, 1.0),
    )


def plot_generation_length_distribution(
    summary_stats_by_source: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]],
    config_id: str,
    output_path: Path,
) -> bool:
    return _plot_cdf_for_sources(
        summary_stats_by_source,
        GENERATION_LENGTHS,
        "Generated Words per Response",
        output_path,
        config_id,
        x_limits=None,
    )


def plot_kv_transfer_scatter(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_id: str,
    output_path: Path,
) -> bool:
    offload_stats = summary_stats.get(KV_OFFLOAD_SCATTER, {}).get(config_id, {})
    reload_stats = summary_stats.get(KV_RELOAD_SCATTER, {}).get(config_id, {})

    def flatten_numeric_entries(num_map: Dict[int, Dict[str, Iterable[float]]]) -> List[float]:
        values: List[float] = []
        for stats in num_map.values():
            raw = stats.get("values")
            if raw:
                values.extend(float(v) for v in raw if v is not None and math.isfinite(v))
            else:
                mean_val = stats.get("mean")
                if mean_val is not None and math.isfinite(mean_val):
                    values.append(float(mean_val))
        return values

    off_entries: List[Tuple[float, float]] = []
    for num_map in offload_stats.values():
        for pair in num_map.get("values", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                size, time = pair
                if math.isfinite(size) and math.isfinite(time):
                    off_entries.append((float(size), float(time) * 1000.0))

    reload_entries: List[Tuple[float, float]] = []
    for num_map in reload_stats.values():
        for pair in num_map.get("values", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                size, time = pair
                if math.isfinite(size) and math.isfinite(time):
                    reload_entries.append((float(size), float(time) * 1000.0))

    def filter_outliers(entries: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(entries) < 6:
            return entries
        times = np.array([t for _, t in entries])
        q1 = np.percentile(times, 25)
        q3 = np.percentile(times, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [(s, t) for s, t in entries if lower <= t <= upper]

    off_filtered = filter_outliers(off_entries)
    reload_filtered = filter_outliers(reload_entries)

    off_times = [t for _, t in off_filtered]
    reload_times = [t for _, t in reload_filtered]

    def aggregate_metric_ms(metric_key: str) -> float:
        num_map = summary_stats.get(metric_key, {}).get(config_id, {})
        values = flatten_numeric_entries(num_map)
        if not values:
            return 0.0
        return float(np.mean(values) * 1000.0)

    prefilling_ms = aggregate_metric_ms("visual_embedding_time_per_frame")
    eos_ms = aggregate_metric_ms("model_forward_time_per_frame")
    decoding_ms = aggregate_metric_ms("generation_time_per_frame")
    network_ms = aggregate_metric_ms("total_sending_time_per_frame")

    if not any([off_times, reload_times, prefilling_ms, eos_ms, decoding_ms, network_ms]):
        return False

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    components = [
        ("Prefilling", prefilling_ms),
        ("EOS", eos_ms),
        ("Decoding", decoding_ms),
        ("Network", network_ms),
        ("Offload", float(np.mean(off_times)) if off_times else 0.0),
        ("Reload", float(np.mean(reload_times)) if reload_times else 0.0),
    ]

    components = [(name, value) for name, value in components if math.isfinite(value) and value > 0]
    if not components:
        plt.close(fig)
        return False

    colors = _scientific_colors()
    x = np.arange(len(components))
    values = [value for _, value in components]
    ax.bar(
        x,
        values,
        color=[colors[idx % len(colors)] for idx in range(len(components))],
        edgecolor="black",
        linewidth=0.4,
    )

    total_time = sum(values)
    for xpos, value in zip(x, values):
        frac = value / total_time if total_time > 0 else 0.0
        ax.text(
            xpos,
            value * 1.05 if value > 0 else 0.02,
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_FONT_SIZE,
        )

    ax.set_ylabel("Time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in components], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_kv_slope_comparison(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_id: str,
    output_path: Path,
) -> bool:
    off_stats = summary_stats.get(KV_OFFLOAD_SLOPE, {}).get(config_id, {})
    reload_stats = summary_stats.get(KV_RELOAD_SLOPE, {}).get(config_id, {})

    off_values: List[float] = []
    for num_map in off_stats.values():
        raw = num_map.get("values", [])
        off_values.extend([float(v) for v in raw if v is not None and math.isfinite(v)])

    reload_values: List[float] = []
    for num_map in reload_stats.values():
        raw = num_map.get("values", [])
        reload_values.extend([float(v) for v in raw if v is not None and math.isfinite(v)])

    datasets = [off_values, reload_values]
    if not any(values for values in datasets):
        return False

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    positions = [1, 2]
    colors = ["#4E79A7", "#F28E2B"]
    labels = ["Offload", "Reload"]

    violin = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    for body, color, values in zip(violin["bodies"], colors, datasets):
        if values:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.6)
        else:
            body.set_visible(False)

    for pos, values, color, label in zip(positions, datasets, colors, labels):
        if not values:
            continue
        mean_val = float(np.mean(values))
        ax.scatter(pos, mean_val, color=color, edgecolor="black", zorder=3, s=25)
        ax.text(
            pos + 0.05,
            mean_val,
            f"{mean_val:.2f}",
            ha="left",
            va="center",
            fontsize=BAR_LABEL_FONT_SIZE,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7),
        )

    ax.set_ylabel("Inflation (x1e-3)", fontsize=18)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_buffering_components_comparison(
    summary_stats: Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]],
    config_ids: List[str],
    target_num_videos: int,
    output_path: Path,
) -> bool:
    metric_keys = [
        ("rebuffer_time", "Total Rebuffering"),
        ("processing_delay", "Processing"),
        ("queuing_delay", "Queuing"),
        ("network_delay", "Networking"),
    ]

    config_stats = []
    for config in config_ids:
        entry = []
        for metric_key, _ in metric_keys:
            metric_data = summary_stats.get(metric_key, {}).get(config, {}).get(target_num_videos, {})
            mean = metric_data.get("mean")
            std = metric_data.get("std")
            if mean is None or not math.isfinite(mean):
                entry.append((float("nan"), 0.0))
            else:
                entry.append((float(mean), float(std) if std is not None and math.isfinite(std) else 0.0))
        config_stats.append(entry)

    if not any(math.isfinite(val[0]) for stats in config_stats for val in stats):
        return False

    labels = [label for _, label in metric_keys]
    x = np.arange(len(labels))
    bar_width = 0.35

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    colors = ["#4E79A7", "#F28E2B"]
    methods = ["Monolithic", "Sliced"]
    for idx, (config, stats) in enumerate(zip(config_ids, config_stats)):
        means = [val[0] for val in stats]
        stds = [val[1] for val in stats]
        offset = (idx - (len(config_ids) - 1) / 2) * (bar_width + 0.02)
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=stds,
            capsize=2.5,
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            label=methods[idx],
        )

    ax.set_ylabel("Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_timing_breakdown(
    all_summary_stats: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]],
    data_sources: List[str],
    output_path: Path,
    target_num_videos: int = 5,
    config_id: str = "round_robin_m",
) -> bool:
    components = [
        ("visual_embedding_time", "Prefilling"),
        ("model_forward_time", "EOS"),
        ("generation_time", "Decoding"),
        ("kv_offload_time", "Offload"),
        ("kv_reload_time", "Reload"),
        ("total_sending_time", "Network"),
        ("total_processing_time", "Total"),
    ]

    ds_order: List[str] = []
    means_matrix: List[List[float]] = []
    std_matrix: List[List[float]] = []
    for data_source in data_sources:
        summary_stats = all_summary_stats.get(data_source)
        if not summary_stats:
            continue
        ds_means: List[float] = []
        ds_stds: List[float] = []
        for metric_key, _ in components:
            metric_data = (
                summary_stats.get(metric_key, {})
                .get(config_id, {})
                .get(target_num_videos, {})
            )
            mean = metric_data.get("mean")
            std = metric_data.get("std")
            if mean is None or not math.isfinite(mean):
                ds_means.append(float("nan"))
                ds_stds.append(0.0)
            else:
                ds_means.append(float(mean))
                ds_stds.append(float(std) if std is not None and math.isfinite(std) else 0.0)
        if any(math.isfinite(m) for m in ds_means):
            ds_order.append(data_source)
            means_matrix.append(ds_means)
            std_matrix.append(ds_stds)

    if not ds_order:
        return False

    configure_plot_style()
    palette = _color_hatch_cycle(len(ds_order))
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(components))
    bar_width = min(0.8 / max(len(ds_order), 1), 0.25)

    for idx, (data_source, means, stds) in enumerate(zip(ds_order, means_matrix, std_matrix)):
        color, hatch = palette[idx]
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        stds_arr[~np.isfinite(stds_arr)] = 0.0
        if np.all(~np.isfinite(means_arr)):
            continue
        offset = (idx - (len(ds_order) - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=data_source.title(),
        )
        for xpos, value, comp_label in zip(x + offset, means_arr, [lbl for _, lbl in components]):
            if comp_label not in {"Offload", "Reload"} or not math.isfinite(value):
                continue
            ax.text(
                xpos,
                value + 0.08 * max(value, 0.5),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONT_SIZE,
                rotation=90,
            )

    ax.set_ylabel("Time (s)", fontsize=10)
    # set y tick size to 10
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in components], rotation=20, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=1, fontsize=LEGEND_FONT_SIZE)

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
        palette = _color_hatch_cycle(len(valid_baselines))
        x = np.arange(len(group_labels))
        bar_width = min(0.75 / max(len(valid_baselines), 1), 0.18)
        fig, ax = plt.subplots(figsize=(3.8, 2.6))

        for idx, baseline_label in enumerate(valid_baselines):
            means_arr = np.asarray(values[baseline_label], dtype=float)
            stds_arr = np.asarray(errors[baseline_label], dtype=float)
            stds_arr[~np.isfinite(stds_arr)] = 0.0
            offset = (idx - (len(valid_baselines) - 1) / 2) * bar_width
            color, hatch = palette[idx]
            ax.bar(
                x + offset,
                means_arr,
                width=bar_width * 0.95,
                yerr=stds_arr,
                capsize=2.5,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.4,
                label=baseline_label,
            )

        ax.set_ylabel("Rebuffering Time (s)")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper left")
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
    palette = _color_hatch_cycle(len(ablation_configs))
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
        stds_arr[~np.isfinite(stds_arr)] = 0.0

        offset = (idx - (len(ablation_configs) - 1) / 2) * bar_width
        color, hatch = palette[idx]
        ax.bar(
            x_positions + offset,
            means_arr,
            width=bar_width * 0.95,
            yerr=stds_arr,
            capsize=2.5,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

    if not plotted:
        plt.close(fig)
        return False

    ax.set_ylabel("Rebuffering Time (s)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(n) for n in num_videos])
    ax.set_xlabel("Number of Users")
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE, ncol=2, loc="upper left")
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
    ablation_num_set: Set[int] = set(ablation_target_numbers)
    iterations: List[int] = list(args.iterations)
    data_sources: List[str] = list(dict.fromkeys(args.data_sources))
    baseline_comparison_nums: List[int] = (
        sorted(set(args.baseline_comparison_nums)) if args.baseline_comparison_nums else list(DEFAULT_NUM_VIDEOS)
    )
    comparison_base_dir: Optional[Path] = args.comparison_base_dir
    if comparison_base_dir is None:
        default_comparison = Path("../results-3090")
        if default_comparison.exists():
            comparison_base_dir = default_comparison.resolve()
    primary_label = args.primary_label or (base_dir.name or "primary")
    comparison_label = args.comparison_label
    per_video_targets: List[int] = sorted(set(args.per_video_numbers))
    overall_targets: List[int] = sorted(set(args.overall_video_numbers))

    all_metric_keys = (
        [key for key, *_ in METRIC_ORDER]
        + [key for key, *_ in LATENCY_COMPONENTS]
        + [f"{key}_per_frame" for key, *_ in LATENCY_COMPONENTS]
        + [key for key, *_ in DELAY_METRICS]
        + [key for key, *_ in SCHEDULING_COMPONENTS]
        + [key for key, *_ in MEMORY_COMPONENTS]
        + [
            KV_CACHE_MEMORY,
            GENERATION_SPEED_METRIC,
            FRACTION_RESPONSE_FRAMES_METRIC,
            GENERATION_LENGTHS,
            KV_OFFLOAD_SCATTER,
            KV_RELOAD_SCATTER,
            KV_SECONDARY_METRIC,
            KV_OFFLOAD_SLOPE,
            KV_RELOAD_SLOPE,
        ]
    )

    summary_stats_map: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]] = {}
    available_numbers_map: Dict[str, Dict[str, Set[int]]] = {}
    missing_paths_map: Dict[str, Dict[str, List[Path]]] = {}

    base_configs: List[Tuple[str, Path]] = [("primary", base_dir)]
    if comparison_base_dir:
        base_configs.append(("comparison", comparison_base_dir))

    for key, target_dir in base_configs:
        all_metrics_storage: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}
        available_numbers_local: Dict[str, Set[int]] = defaultdict(set)
        missing_paths_local: Dict[str, List[Path]] = defaultdict(list)

        for data_source in data_sources:
            candidate_dir = target_dir / data_source
            if candidate_dir.exists():
                ds_root = candidate_dir
            elif target_dir.name == data_source and target_dir.exists():
                ds_root = target_dir
            else:
                ds_root = candidate_dir
            ds_root = ds_root.resolve()

            if not ds_root.exists():
                print(
                    f"Warning: Data source directory not found for '{data_source}' in base '{target_dir}': {ds_root}"
                )
                all_metrics_storage[data_source] = {
                    metric: defaultdict(lambda: defaultdict(list)) for metric in all_metric_keys
                }
                continue

            if general_num_override:
                num_videos = list(general_num_override)
            else:
                detected = sorted(
                    int(p.name[1:])
                    for p in ds_root.glob("N*")
                    if p.is_dir() and len(p.name) > 1 and p.name[1:].isdigit()
                )
                num_videos = detected if detected else list(DEFAULT_NUM_VIDEOS)
            num_videos = sorted(set(num_videos) | set(ablation_target_numbers))

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
                            missing_paths_local[data_source].append(summary_path)
                            continue
                        metrics = load_iteration_metrics(summary_path, num)
                        available_numbers_local[data_source].add(num)
                        for metric_key in all_metric_keys:
                            value = metrics.get(metric_key, float("nan"))
                            storage_list = metrics_storage[metric_key][config_id][num]
                            if metric_key in EXTEND_METRICS:
                                if isinstance(value, list):
                                    storage_list.extend(value)
                                elif value is not None and value != float("nan"):
                                    storage_list.append(value)
                                continue
                            storage_list.append(value)

            all_metrics_storage[data_source] = metrics_storage

        summary_stats_map[key] = {}
        available_numbers_map[key] = available_numbers_local
        missing_paths_map[key] = missing_paths_local

        for data_source, ds_metrics in all_metrics_storage.items():
            summary_stats = {metric: {} for metric in all_metric_keys}
            for metric_key, configs in ds_metrics.items():
                for config_id, num_map in configs.items():
                    summary_stats[metric_key].setdefault(config_id, {})
                    for num, values in num_map.items():
                        if metric_key in EXTEND_METRICS:
                            summary_stats[metric_key][config_id][num] = {
                                "values": list(values),
                            }
                            continue
                        mean, std = safe_mean_std(values)
                        summary_stats[metric_key][config_id][num] = {
                            "mean": mean,
                            "std": std,
                            "values": list(values),
                        }
            summary_stats_map[key][data_source] = summary_stats

    primary_summary_stats = summary_stats_map.get("primary", {})
    if not primary_summary_stats:
        print("No data available to plot. Exiting.")
        return
    available_numbers = available_numbers_map.get("primary", defaultdict(set))

    # for data_source, paths in missing_paths_map.get("primary", {}).items():
    #     if paths:
    #         print(
    #             f"Warning: Missing summaries for data_source '{data_source}' in base '{base_dir}':"
    #         )
    #         for path in paths:
    #             print(f"  - {path}")

    comparison_summary_stats = summary_stats_map.get("comparison", {})
    # if comparison_summary_stats and comparison_base_dir:
    #     for data_source, paths in missing_paths_map.get("comparison", {}).items():
    #         if paths:
    #             print(
    #                 f"Warning: Missing summaries for data_source '{data_source}' in base '{comparison_base_dir}':"
    #             )
    #             for path in paths:
    #                 print(f"  - {path}")

    metric_meta = {key: (y_label, title, slug) for key, y_label, title, slug in METRIC_ORDER}
    all_summary_stats = primary_summary_stats

    distribution_stats: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, Iterable[float]]]]]] = {}

    for data_source, summary_stats in all_summary_stats.items():
        # Determine numbers and configs for plotting
        ds_numbers = sorted(available_numbers.get(data_source, []))
        if general_num_override:
            ds_numbers = sorted(set(ds_numbers) | set(general_num_override))
        if not ds_numbers:
            ds_numbers = list(DEFAULT_NUM_VIDEOS)

        distribution_stats[data_source] = summary_stats

        per_video_nums = [n for n in per_video_targets if n in ds_numbers]
        if not per_video_nums:
            per_video_nums = ds_numbers

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
                per_video_nums,
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
                allowed_nums=overall_targets,
            )
            if overall_saved:
                print(f"[{data_source}] Saved aggregated bar plot to {overall_path}")
            else:
                print(f"[{data_source}] Skipped aggregated bar plot for {metric_key}; no valid data.")

        # Base delay trends
        base_delay_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_base_delay_vs_videos.pdf"
        if plot_base_delay_trends(summary_stats, per_video_nums, base_delay_path):
            print(f"[{data_source}] Saved base delay trend plot to {base_delay_path}")
        else:
            print(f"[{data_source}] Skipped base delay trend plot; no valid data.")

        # Delay comparison across configs
        delay_compare_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_delay_by_config.pdf"
        if plot_delay_comparison_by_config(summary_stats, general_configs, delay_compare_path, allowed_nums=overall_targets):
            print(f"[{data_source}] Saved delay comparison plot to {delay_compare_path}")
        else:
            print(f"[{data_source}] Skipped delay comparison plot; no valid data.")

        # Scheduling components comparison across configs
        scheduling_config_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_scheduling_components_by_config.pdf"
        if plot_scheduling_components_by_config(summary_stats, general_configs, scheduling_config_path, allowed_nums=overall_targets):
            print(f"[{data_source}] Saved scheduling components by config plot to {scheduling_config_path}")
        else:
            print(f"[{data_source}] Skipped scheduling components by config plot; no valid data.")

        # Scheduling components vs number of videos (base config)
        scheduling_base_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_scheduling_components_base_vs_videos.pdf"
        if plot_scheduling_components_base_vs_videos(summary_stats, per_video_nums, scheduling_base_path):
            print(f"[{data_source}] Saved scheduling components (base) vs videos plot to {scheduling_base_path}")
        else:
            print(f"[{data_source}] Skipped scheduling components (base) vs videos plot; no valid data.")

        # Memory components across configs
        memory_breakdown_path = (
            plot_root_base.parent
            / f"{plot_root_base.name}_{data_source}_memory_breakdown_max_frames_memory_test.pdf"
        )
        if plot_memory_breakdown_multi_videos(
            summary_stats,
            "max_frames_memory_test",
            [3, 5, 8, 10],
            memory_breakdown_path,
        ):
            print(f"[{data_source}] Saved memory breakdown plot to {memory_breakdown_path}")
        else:
            print(f"[{data_source}] Skipped memory breakdown plot; no valid data.")

        memory_ratio_path = (
            plot_root_base.parent
            / f"{plot_root_base.name}_{data_source}_memory_speed_ratios_max_frames_memory_test.pdf"
        )
        if plot_memory_speed_ratios(
            summary_stats,
            "max_frames_memory_test",
            [3, 5, 8, 10],
            memory_ratio_path,
        ):
            print(f"[{data_source}] Saved memory/speed ratio plot to {memory_ratio_path}")
        else:
            print(f"[{data_source}] Skipped memory/speed ratio plot; no valid data.")

        if comparison_summary_stats:
            comp_stats = comparison_summary_stats.get(data_source)
            if comp_stats:
                server_compare_path = (
                    plot_root_base.parent
                    / f"{plot_root_base.name}_{data_source}_baseline_rebuffer_server_comparison.pdf"
                )
                if plot_rebuffer_baseline_comparison_across_bases(
                    summary_stats,
                    comp_stats,
                    list(general_config_ids),
                    baseline_comparison_nums,
                    primary_label,
                    comparison_label,
                    server_compare_path,
                ):
                    print(f"[{data_source}] Saved baseline rebuffer comparison to {server_compare_path}")
                else:
                    print(
                        f"[{data_source}] Skipped baseline rebuffer comparison plot; no valid overlapping data."
                    )

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
        if plot_latency_components_by_config(summary_stats, general_configs, latency_config_path, allowed_nums=overall_targets):
            print(f"[{data_source}] Saved latency components by config plot to {latency_config_path}")
        else:
            print(f"[{data_source}] Skipped latency components by config plot; no valid data.")

        # Generation speed for single-video experiments
        generation_speed_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_generation_speed_vs_listening.pdf"
        if plot_generation_speed_vs_listening(summary_stats, BASE_CONFIG_ID, generation_speed_path):
            print(f"[{data_source}] Saved generation speed vs listening plot to {generation_speed_path}")
        else:
            print(f"[{data_source}] Skipped generation speed vs listening plot; no valid data.")

        kv_scatter_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_kv_transfer_scatter.pdf"
        if plot_kv_transfer_scatter(summary_stats, "round_robin_m", kv_scatter_path):
            print(f"[{data_source}] Saved KV transfer scatter to {kv_scatter_path}")
        else:
            print(f"[{data_source}] Skipped KV transfer scatter plot; no valid data.")

        kv_slope_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_kv_transfer_slope.pdf"
        if plot_kv_slope_comparison(summary_stats, "round_robin_m", kv_slope_path):
            print(f"[{data_source}] Saved KV transfer slope plot to {kv_slope_path}")
        else:
            print(f"[{data_source}] Skipped KV transfer slope plot; no valid data.")

        buffering_compare_path = plot_root_base.parent / f"{plot_root_base.name}_{data_source}_buffering_components_comparison.pdf"
        if plot_buffering_components_comparison(
            summary_stats, ["round_robin_m", "round_robin_2"], 3, buffering_compare_path
        ):
            print(f"[{data_source}] Saved buffering components comparison to {buffering_compare_path}")
        else:
            print(f"[{data_source}] Skipped buffering components comparison plot; no valid data.")

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
                        mean = stats.get("mean", "")
                        std = stats.get("std", "")
                        raw_values = stats.get("values", [])
                        formatted_values = []
                        for v in raw_values:
                            if isinstance(v, (list, tuple)) and len(v) == 2:
                                size, time = v
                                formatted_values.append(f"{size:.6f},{time:.6f}")
                            elif isinstance(v, float):
                                formatted_values.append(f"{v:.6f}")
                            else:
                                formatted_values.append(str(v))
                        iteration_values = ";".join(formatted_values)
                        writer.writerow(
                            [
                                metric_key,
                                config_id,
                                num,
                                mean,
                                std,
                                iteration_values,
                            ]
                        )
        print(f"[{data_source}] Wrote summary table to {csv_path}")

    if distribution_stats:
        combined_dir = base_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        generation_length_path = combined_dir / "config_metrics_generation_length_cdf.pdf"
        if plot_generation_length_distribution(distribution_stats, DISTRIBUTION_CONFIG_ID, generation_length_path):
            print(f"[combined] Saved generation length CDF to {generation_length_path}")
        else:
            print("[combined] Skipped generation length CDF; no valid data.")

        fraction_path = combined_dir / "config_metrics_response_fraction_cdf.pdf"
        if plot_fraction_response_distribution(distribution_stats, DISTRIBUTION_CONFIG_ID, fraction_path):
            print(f"[combined] Saved response fraction CDF to {fraction_path}")
        else:
            print("[combined] Skipped response fraction CDF; no valid data.")

    combined_timing_path = combined_dir / "timing_breakdown_round_robin_m_combined.pdf"
    if plot_timing_breakdown(all_summary_stats, data_sources, combined_timing_path):
        print(f"[combined] Saved combined timing breakdown to {combined_timing_path}")
    else:
        print("[combined] Skipped combined timing breakdown plot; no valid data.")

    # Combined round_robin_m comparison across data sources
    if data_sources:
        combined_path = combined_dir / "roundrobin_m_rebuffer_across_sources.pdf"
        if plot_roundrobin_comparison(
            all_summary_stats,
            data_sources,
            available_numbers,
            general_num_override,
            combined_path,
        ):
            print(f"[combined] Saved round_robin_m comparison to {combined_path}")
        else:
            print("[combined] Skipped round_robin_m comparison plot; no valid data.")

    # Combined round_robin_2 / round_robin_m ratio comparison across data sources
    combined_ratio_path = (
        combined_dir / "roundrobin_2_over_m_ratio_goalstep.pdf"
    )
    if plot_roundrobin_ratio(
        all_summary_stats,
        ["goalstep"],
        available_numbers,
        list(range(1, 11)),
        combined_ratio_path,
    ):
        print(f"[combined] Saved goalstep round_robin_2 / round_robin_m ratio comparison to {combined_ratio_path}")
    else:
        print("[combined] Skipped round_robin_2 / round_robin_m ratio comparison plot; no valid data.")


if __name__ == "__main__":
    main()
