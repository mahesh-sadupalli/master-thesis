"""
Generate dataset-level exploratory plots for spatio-temporal flow data.

Outputs:
  - target_feature_histograms.png
  - spatial_mesh_first_timestep.png
  - temporal_target_trends.png
  - feature_correlation_heatmap.png
  - velocity_direction_dominance.png
  - dataset_feature_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "ML_test_loader_original_data.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "dataset_analysis"
MPLCONFIG_DIR = ROOT_DIR / ".matplotlib"

# Use a headless backend to avoid macOS GUI backend aborts in CLI runs.
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMNS = ["x", "y", "z", "t", "Vx", "Vy", "Pressure", "TKE"]
TARGET_COLUMNS = ["Vx", "Vy", "Pressure", "TKE"]
TARGET_INDICES = [4, 5, 6, 7]


def process_chunk(
    arr: np.ndarray,
    state: dict,
) -> None:
    """Update global aggregates for one numeric chunk."""
    n_chunk = arr.shape[0]
    state["n_rows"] += n_chunk

    state["sum_v"] += arr.sum(axis=0)
    state["sumsq_v"] += np.square(arr).sum(axis=0)
    state["xtx"] += arr.T @ arr

    # Histogram for target variables.
    for col_name, col_idx in zip(TARGET_COLUMNS, TARGET_INDICES):
        state["target_hist_counts"][col_name] += np.histogram(
            arr[:, col_idx], bins=state["target_bins"]
        )[0]

    # Velocity directional dominance counts.
    abs_vx = np.abs(arr[:, 4])
    abs_vy = np.abs(arr[:, 5])
    state["n_vx_gt_vy"] += int((abs_vx > abs_vy).sum())
    state["n_vy_gt_vx"] += int((abs_vy > abs_vx).sum())
    state["n_vx_eq_vy"] += int((abs_vx == abs_vy).sum())

    # Collect mesh points from the first timestep only.
    tvals = arr[:, 3]
    if state["first_timestep"] is None and len(tvals) > 0:
        state["first_timestep"] = float(tvals[0])

    if state["first_timestep"] is not None:
        mask_first_t = np.isclose(tvals, state["first_timestep"], atol=1e-12)
        if np.any(mask_first_t):
            state["first_timestep_xy"].append(arr[mask_first_t][:, [0, 1]])

    # Per-timestep aggregates for temporal plots.
    unique_t, inverse = np.unique(tvals, return_inverse=True)
    for i, t_val in enumerate(unique_t):
        mask = inverse == i
        target_block = arr[mask][:, 4:8]
        t_key = float(t_val)
        if t_key not in state["timestep_count"]:
            state["timestep_count"][t_key] = 0
            state["timestep_sum"][t_key] = np.zeros(4, dtype=np.float64)
            state["timestep_sumsq"][t_key] = np.zeros(4, dtype=np.float64)

        state["timestep_count"][t_key] += int(mask.sum())
        state["timestep_sum"][t_key] += target_block.sum(axis=0)
        state["timestep_sumsq"][t_key] += np.square(target_block).sum(axis=0)


def build_dataset_aggregates() -> dict:
    """Stream CSV once and collect all required statistics."""
    state = {
        "n_rows": 0,
        "sum_v": np.zeros(len(COLUMNS), dtype=np.float64),
        "sumsq_v": np.zeros(len(COLUMNS), dtype=np.float64),
        "xtx": np.zeros((len(COLUMNS), len(COLUMNS)), dtype=np.float64),
        "target_bins": np.linspace(0.0, 1.0, 51),
        "target_hist_counts": {
            name: np.zeros(50, dtype=np.int64) for name in TARGET_COLUMNS
        },
        "n_vx_gt_vy": 0,
        "n_vy_gt_vx": 0,
        "n_vx_eq_vy": 0,
        "first_timestep": None,
        "first_timestep_xy": [],
        "timestep_count": {},
        "timestep_sum": {},
        "timestep_sumsq": {},
    }

    chunk_rows = 100_000
    buffer = []

    chunk_idx = 0
    with DATA_FILE.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 8:
                continue
            try:
                values = [float(v) for v in row]
            except ValueError:
                continue
            buffer.append(values)

            if len(buffer) >= chunk_rows:
                arr = np.asarray(buffer, dtype=np.float64)
                process_chunk(arr, state)
                buffer.clear()
                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    print(
                        f"  processed ~{state['n_rows']:,} rows...",
                        flush=True,
                    )

    if buffer:
        arr = np.asarray(buffer, dtype=np.float64)
        process_chunk(arr, state)

    return state


def build_dataset_aggregates_with_limit(max_rows: int | None) -> dict:
    """Stream CSV with optional row cap and collect all required statistics."""
    state = {
        "n_rows": 0,
        "sum_v": np.zeros(len(COLUMNS), dtype=np.float64),
        "sumsq_v": np.zeros(len(COLUMNS), dtype=np.float64),
        "xtx": np.zeros((len(COLUMNS), len(COLUMNS)), dtype=np.float64),
        "target_bins": np.linspace(0.0, 1.0, 51),
        "target_hist_counts": {
            name: np.zeros(50, dtype=np.int64) for name in TARGET_COLUMNS
        },
        "n_vx_gt_vy": 0,
        "n_vy_gt_vx": 0,
        "n_vx_eq_vy": 0,
        "first_timestep": None,
        "first_timestep_xy": [],
        "timestep_count": {},
        "timestep_sum": {},
        "timestep_sumsq": {},
    }

    chunk_rows = 100_000
    chunk_idx = 0
    buffer = []
    n_read = 0

    with DATA_FILE.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 8:
                continue
            try:
                values = [float(v) for v in row]
            except ValueError:
                continue
            buffer.append(values)
            n_read += 1

            if len(buffer) >= chunk_rows:
                arr = np.asarray(buffer, dtype=np.float64)
                process_chunk(arr, state)
                buffer.clear()
                chunk_idx += 1
                if chunk_idx % 10 == 0:
                    print(
                        f"  processed ~{state['n_rows']:,} rows...",
                        flush=True,
                    )

            if max_rows is not None and n_read >= max_rows:
                break

    if buffer:
        arr = np.asarray(buffer, dtype=np.float64)
        process_chunk(arr, state)

    return state


def plot_target_histograms(state: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bin_edges = state["target_bins"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    for ax, target_name in zip(axes.ravel(), TARGET_COLUMNS):
        counts = state["target_hist_counts"][target_name]
        frac = counts / state["n_rows"]
        ax.bar(bin_centers, frac, width=bin_width * 0.95, color="#2b7a78")
        ax.set_title(f"{target_name} Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Fraction of samples")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.3)

    fig.suptitle("Target Feature Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "target_feature_histograms.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spatial_mesh(state: dict) -> Path:
    if not state["first_timestep_xy"]:
        raise RuntimeError("No points captured for first timestep.")

    xy = np.vstack(state["first_timestep_xy"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=0.6, alpha=0.7, color="#1f4e79")
    ax.set_title(
        f"Spatial Mesh at First Timestep (t={state['first_timestep']:.4f})",
        fontweight="bold",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    out_path = OUTPUT_DIR / "spatial_mesh_first_timestep.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_temporal_trends(state: dict) -> Path:
    sorted_t = np.array(sorted(state["timestep_count"].keys()), dtype=np.float64)
    counts = np.array([state["timestep_count"][t] for t in sorted_t], dtype=np.float64)

    means = np.zeros((len(sorted_t), 4), dtype=np.float64)
    stds = np.zeros((len(sorted_t), 4), dtype=np.float64)

    for i, t in enumerate(sorted_t):
        sums = state["timestep_sum"][float(t)]
        sumsqs = state["timestep_sumsq"][float(t)]
        means[i] = sums / counts[i]
        var = np.maximum(sumsqs / counts[i] - np.square(means[i]), 0.0)
        stds[i] = np.sqrt(var)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    for j, (ax, target_name) in enumerate(zip(axes.ravel(), TARGET_COLUMNS)):
        ax.plot(sorted_t, means[:, j], color=colors[j], linewidth=1.8, label="mean")
        ax.fill_between(
            sorted_t,
            means[:, j] - stds[:, j],
            means[:, j] + stds[:, j],
            color=colors[j],
            alpha=0.2,
            label="mean ± std",
        )
        ax.set_title(f"{target_name} vs Time")
        ax.set_xlabel("t")
        ax.set_ylabel(target_name)
        ax.grid(alpha=0.3)

    fig.suptitle("Per-Timestep Target Statistics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "temporal_target_trends.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(state: dict) -> Path:
    mean_v = state["sum_v"] / state["n_rows"]
    var_v = np.maximum(state["sumsq_v"] / state["n_rows"] - np.square(mean_v), 0.0)
    std_v = np.sqrt(var_v)

    cov = state["xtx"] / state["n_rows"] - np.outer(mean_v, mean_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std_v, std_v)
        corr[np.isnan(corr)] = 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_yticks(range(len(COLUMNS)))
    ax.set_xticklabels(COLUMNS, rotation=45, ha="right")
    ax.set_yticklabels(COLUMNS)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")

    # Keep text annotations minimal for readability.
    for i in range(len(COLUMNS)):
        for j in range(len(COLUMNS)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "feature_correlation_heatmap.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_velocity_dominance(state: dict) -> Path:
    labels = ["|Vy| > |Vx|", "|Vx| > |Vy|", "|Vx| = |Vy|"]
    values = np.array(
        [
            state["n_vy_gt_vx"],
            state["n_vx_gt_vy"],
            state["n_vx_eq_vy"],
        ],
        dtype=np.float64,
    )
    frac = values / state["n_rows"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, frac, color=["#1b9e77", "#7570b3", "#d95f02"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of rows")
    ax.set_title("Velocity Directional Dominance", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, frac):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    out_path = OUTPUT_DIR / "velocity_direction_dominance.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def export_summary(state: dict) -> Path:
    mean_v = state["sum_v"] / state["n_rows"]
    var_v = np.maximum(state["sumsq_v"] / state["n_rows"] - np.square(mean_v), 0.0)
    std_v = np.sqrt(var_v)

    sorted_t = np.array(sorted(state["timestep_count"].keys()), dtype=np.float64)
    counts = np.array([state["timestep_count"][t] for t in sorted_t], dtype=np.int64)

    summary = {
        "rows": int(state["n_rows"]),
        "columns": COLUMNS,
        "first_timestep": float(state["first_timestep"]),
        "n_timesteps": int(len(sorted_t)),
        "rows_per_timestep_min": int(counts.min()) if len(counts) > 0 else None,
        "rows_per_timestep_max": int(counts.max()) if len(counts) > 0 else None,
        "rows_per_timestep_mean": float(counts.mean()) if len(counts) > 0 else None,
        "feature_stats": {
            name: {
                "mean": float(mean_v[i]),
                "std": float(std_v[i]),
            }
            for i, name in enumerate(COLUMNS)
        },
        "velocity_directional_dominance": {
            "frac_vy_gt_vx": float(state["n_vy_gt_vx"] / state["n_rows"]),
            "frac_vx_gt_vy": float(state["n_vx_gt_vy"] / state["n_rows"]),
            "frac_equal": float(state["n_vx_eq_vy"] / state["n_rows"]),
        },
    }

    out_path = OUTPUT_DIR / "dataset_feature_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dataset feature analysis.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for processed rows (debug/testing).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading dataset: {DATA_FILE}")
    state = build_dataset_aggregates_with_limit(args.max_rows)
    print(f"Processed rows: {state['n_rows']:,}")

    print("Plotting target histograms...", flush=True)
    p1 = plot_target_histograms(state)
    print("Plotting spatial mesh...", flush=True)
    p2 = plot_spatial_mesh(state)
    print("Plotting temporal trends...", flush=True)
    p3 = plot_temporal_trends(state)
    print("Plotting correlation heatmap...", flush=True)
    p4 = plot_correlation_heatmap(state)
    print("Plotting velocity dominance...", flush=True)
    p5 = plot_velocity_dominance(state)
    print("Exporting summary JSON...", flush=True)
    p6 = export_summary(state)
    paths = [p1, p2, p3, p4, p5, p6]

    print("Generated files:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
