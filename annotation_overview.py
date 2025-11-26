#!/usr/bin/env python3
"""End-to-end annotation overview for Goalstep & Narration datasets."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

GOALSTEP_PATH = Path("datasets/ego4d/v2/annotations/goalstep_livechat_trainval_filtered_21k.json")
NARRATION_PATH = Path("datasets/ego4d/v2/annotations/refined_narration_stream_val.json")
DEFAULT_OUTPUT_DIR = Path("figures/combined")
plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class AggregateMetrics:
    response_lengths: List[float]
    responses_per_conv: List[int]
    conv_durations: List[float]
    responses_per_minute: List[float]
    response_gaps: List[float]
    words_per_minute: List[float]
    response_length_pairs: List[Tuple[float, float]]
    total_videos: int
    total_conversations: int
    total_responses: int


def load_goalstep() -> AggregateMetrics:
    data = json.loads(GOALSTEP_PATH.read_text())
    response_lengths: List[float] = []
    responses_per_conv: List[int] = []
    conv_durations: List[float] = []
    responses_per_minute: List[float] = []
    response_gaps: List[float] = []
    words_per_minute: List[float] = []
    video_ids: set[str] = set()
    response_pairs: List[Tuple[float, float]] = []

    for entry in data:
        video_uid = entry.get("video_uid", "")
        video_ids.add(video_uid)
        conversation = entry.get("conversation", []) or []
        responses = [turn for turn in conversation if turn.get("role") == "assistant"]
        responses_per_conv.append(len(responses))

        resp_times = []
        conv_word_total = 0.0
        prev_lengths: List[float] = []
        for turn in responses:
            text = turn.get("content") or ""
            words = len(text.split())
            conv_word_total += words
            response_lengths.append(words)
            if words > 0:
                for prev in prev_lengths:
                    response_pairs.append((prev, words))
                prev_lengths.append(words)
            if isinstance(turn.get("time"), (int, float)):
                resp_times.append(float(turn["time"]))

        if resp_times:
            resp_times = sorted(resp_times)
            response_gaps.extend(np.diff(resp_times))

        duration = entry.get("duration")
        if duration is None:
            if conversation and all(isinstance(turn.get("time"), (int, float)) for turn in conversation):
                times = [float(turn["time"]) for turn in conversation]
                duration = max(times) - min(times)
        if duration is not None and duration > 0:
            conv_durations.append(float(duration))
            responses_per_minute.append(len(responses) / (float(duration) / 60.0) if responses else 0.0)
            if conv_word_total > 0:
                words_per_minute.append(conv_word_total / (float(duration) / 60.0))

    return AggregateMetrics(
        response_lengths,
        responses_per_conv,
        conv_durations,
        responses_per_minute,
        response_gaps,
        words_per_minute,
        response_pairs,
        total_videos=len(video_ids),
        total_conversations=len(data),
        total_responses=len(response_lengths),
    )


def load_narration() -> AggregateMetrics:
    data = json.loads(NARRATION_PATH.read_text())
    response_lengths: List[float] = []
    responses_per_conv: List[int] = []
    conv_durations: List[float] = []
    responses_per_minute: List[float] = []
    response_gaps: List[float] = []
    words_per_minute: List[float] = []
    video_ids: set[str] = set()
    conv_count = 0
    response_pairs: List[Tuple[float, float]] = []

    for video_uid, conversations in data.items():
        video_ids.add(video_uid)
        for conv_id, entries in conversations.items():
            conv_count += 1
            entries = entries or []
            responses_per_conv.append(len(entries))
            times = []
            conv_word_total = 0.0
        prev_lengths: List[float] = []
        for entry in entries:
            text = entry.get("text") or ""
            words = len(text.split())
            conv_word_total += words
            response_lengths.append(words)
            if words > 0:
                for prev in prev_lengths:
                    response_pairs.append((prev, words))
                prev_lengths.append(words)
            if isinstance(entry.get("time"), (int, float)):
                times.append(float(entry["time"]))

            if times:
                times = sorted(times)
                duration = max(times) - min(times)
                if duration > 0:
                    conv_durations.append(duration)
                    responses_per_minute.append(len(entries) / (duration / 60.0) if entries else 0.0)
                    if conv_word_total > 0:
                        words_per_minute.append(conv_word_total / (duration / 60.0))
                if len(times) > 1:
                    response_gaps.extend(np.diff(times))

    return AggregateMetrics(
        response_lengths,
        responses_per_conv,
        conv_durations,
        responses_per_minute,
        response_gaps,
        words_per_minute,
        response_pairs,
        total_videos=len(video_ids),
        total_conversations=conv_count,
        total_responses=len(response_lengths),
    )


def analyze_source(data_source: str) -> AggregateMetrics:
    metrics = load_goalstep() if data_source == "goalstep" else load_narration()

    print(
        f"🔍 {data_source.title()} — Videos: {metrics.total_videos:,}, "
        f"Conversations: {metrics.total_conversations:,}, Responses: {metrics.total_responses:,}"
    )

    return metrics


def _prepare_hist(
    values_a: Sequence[float],
    values_b: Sequence[float],
    x_max: float,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_bins = max(4, num_bins)
    x_max = max(1e-6, x_max)

    def _sanitize(seq: Sequence[float]) -> np.ndarray:
        if not seq:
            return np.empty(0)
        arr = np.asarray(seq, dtype=float)
        arr = arr[np.isfinite(arr)]
        return arr[(arr >= 0) & (arr <= x_max)]

    arr_a = _sanitize(values_a)
    arr_b = _sanitize(values_b)

    bins = np.linspace(0, x_max, num_bins + 1)
    hist_a, _ = np.histogram(arr_a, bins=bins)
    hist_b, _ = np.histogram(arr_b, bins=bins)

    if hist_a.sum() > 0:
        hist_a = hist_a / hist_a.sum()
    if hist_b.sum() > 0:
        hist_b = hist_b / hist_b.sum()

    centers = (bins[:-1] + bins[1:]) / 2.0
    widths = np.diff(bins)
    return centers, hist_a, hist_b, widths


def render_combined_comparison(goal: AggregateMetrics, narration: AggregateMetrics, output_path: Path) -> None:
    metrics_pairs = [
        (
            goal.words_per_minute,
            narration.words_per_minute,
            "",
            "#Words per Min",
            120,
            400,
        ),
        (
            goal.response_lengths,
            narration.response_lengths,
            "",
            "Length (#Words)",
            80,
            100,
        ),
        (
            goal.responses_per_minute,
            narration.responses_per_minute,
            "",
            "#Resp. per Min",
            120,
            60,
        ),
    ]
    # set fontsize for the plot
    # "font.family": "serif",
    # "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],  # fallback list
    plt.rcParams.update({'font.size': 32, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
    # update tick size for x y ticks
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    # set fontsize for x y ticks
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=28)
    # set font size for legend
    legend_fontsize = 26
    colors = ("#4E79A7", "#F28E2B")
    # change to 100% on y axis
    for ax, (goal_values, narr_values, title, xlabel, bin_count, x_max) in zip(axes, metrics_pairs):
        if not goal_values and not narr_values:
            ax.axis("off")
            continue
        centers, goal_hist, narr_hist, widths = _prepare_hist(
            goal_values, narr_values, x_max=x_max, num_bins=bin_count
        )
        width = (widths.mean() if widths.size else x_max / bin_count) * 0.45
        # change to 100% on y axis
        goal_hist = goal_hist * 100
        narr_hist = narr_hist * 100
        ax.bar(
            centers - width / 2,
            goal_hist,
            width=width,
            color=colors[0],
            alpha=0.85,
            label="Goalstep",
        )
        ax.bar(
            centers + width / 2,
            narr_hist,
            width=width,
            color=colors[1],
            alpha=0.85,
            label="Narration",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if ax is axes[0]:
            ax.set_ylabel("Prob. (%)")
        else:
            ax.set_ylabel("")
        ax.set_xlim(0, x_max)
        ax.grid(alpha=0.3)
        if ax is axes[1]:
            ax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved combined comparison figure to {output_path}")


def plot_response_length_correlation(
    goal: AggregateMetrics,
    output_path: Path,
) -> None:
    pairs = goal.response_length_pairs
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if not pairs:
        ax.text(0.5, 0.5, "No response data", ha="center", va="center")
    else:
        arr = np.asarray(pairs, dtype=float)
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        prev = arr[:, 0]
        curr = arr[:, 1]
        sample_size = min(prev.size, 8000)
        if prev.size > sample_size:
            idx = np.random.default_rng(0).choice(prev.size, sample_size, replace=False)
            prev_plot = prev[idx]
            curr_plot = curr[idx]
        else:
            prev_plot = prev
            curr_plot = curr
        ax.scatter(prev_plot, curr_plot, alpha=0.2, s=12, color="#4E79A7")
        corr = float("nan")
        if prev.size >= 2:
            coeffs = np.polyfit(prev, curr, deg=1)
            x_line = np.linspace(prev.min(), prev.max(), 200)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, color="#E15759", linewidth=2)
            corr = np.corrcoef(prev, curr)[0, 1]
        ax.set_title(f"Goalstep (r={corr:.2f} / n={prev.size:,})")
        ax.set_xlabel("Previous response length (# words)")
        ax.set_ylabel("Current response length (# words)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved response length correlation figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotation overview for Goalstep/Narration datasets.")
    parser.add_argument(
        "--data-sources",
        nargs="+",
        choices=["goalstep", "narration", "both"],
        default=["goalstep", "narration"],
        help="Datasets to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store plots and summaries.",
    )
    args = parser.parse_args()

    data_sources = []
    for token in args.data_sources:
        if token == "both":
            data_sources.extend(["goalstep", "narration"])
        else:
            data_sources.append(token)
    # preserve order without duplicates
    ordered_sources = list(dict.fromkeys(data_sources))

    metrics_map: Dict[str, AggregateMetrics] = {}
    for source in ordered_sources:
        metrics_map[source] = analyze_source(source)

    if {"goalstep", "narration"}.issubset(metrics_map):
        combined_plot = args.output_dir / "annotation_overview_combined.pdf"
        render_combined_comparison(metrics_map["goalstep"], metrics_map["narration"], combined_plot)
    else:
        print("⚠️ Need both goalstep and narration data to render the combined comparison.")
    if "goalstep" in metrics_map:
        correlation_plot = args.output_dir / "goalstep_response_length_correlation.pdf"
        plot_response_length_correlation(metrics_map["goalstep"], correlation_plot)


if __name__ == "__main__":
    main()
