"""
Time-averaged velocity, pressure, and TKE profiles at multiple downstream
stations behind the cylinder, following the canonical CFD wake-profile
convention.

For each station x_i, a vertical y-line is sampled, the time-averaged
value of every variable is obtained at every y, and the profile is
plotted as value (horizontal axis) versus y-coordinate (vertical axis).

Cylinder geometry: centre (0, 0), radius 0.01, diameter D = 0.02.
Stations: 1D, 2D, 3D, 5D, 7D, 10D downstream of the cylinder centre.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pyarrow.csv as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(ROOT, "data/ML_test_loader_original_data.csv")
OUT_DIR = os.path.join(ROOT, "documents/official_notes/images")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TS = 300
N_PTS = 26397
D = 0.02
STATIONS_D = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
X_STATIONS = [s * D for s in STATIONS_D]
Y_RANGE = (-0.045, 0.045)
N_Y = 120

FEATURES = ["$V_x$", "$V_y$", "Pressure", "TKE"]


class LargeCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.network(x)


MODEL_SPECS = [
    ("Offline INR",
     "results/batch_learning/large_model_offline/large_model_minmax.pth"),
    ("Online Naive",
     "results/new/cl-boosted/naive_large/large_final.pth"),
    ("Online ER-Aggr.",
     "results/new/cl-boosted/er_aggressive_large/large_final.pth"),
]


def load_dataset():
    print(f"[Data] Loading {DATA_FILE}")
    table = pv.read_csv(
        DATA_FILE,
        read_options=pv.ReadOptions(
            column_names=["x", "y", "z", "t", "Vx", "Vy", "Pressure", "TKE"]),
    )
    data = table.to_pandas().values
    inputs = data[:, :4].astype(np.float32)
    targets = data[:, 4:].astype(np.float32)
    in_min, in_max = inputs.min(0), inputs.max(0)
    in_rng = np.where(in_max - in_min == 0, 1.0, in_max - in_min)
    inputs_n = (inputs - in_min) / in_rng
    print(f"[Data] {len(inputs):,} samples")
    return inputs, inputs_n, targets


def load_model(path):
    full = os.path.join(ROOT, path)
    print(f"[Model] {full}")
    m = LargeCompressor().to(device)
    m.load_state_dict(torch.load(full, map_location=device, weights_only=True))
    return m


def run_inference(model, inputs_n, batch=200_000):
    model.eval()
    preds = np.empty((len(inputs_n), 4), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(inputs_n), batch):
            chunk = torch.from_numpy(inputs_n[s:s + batch]).to(device)
            preds[s:s + batch] = model(chunk).cpu().numpy()
    return np.clip(preds, 0.0, 1.0)


def time_average(arr_flat):
    return arr_flat.reshape(N_TS, N_PTS, 4).mean(axis=0)


def sample_profile_at_x(xy_points, field_2d, x_target, y_line):
    """Interpolate time-averaged field onto vertical line (x_target, y_line)."""
    line_pts = np.column_stack([np.full_like(y_line, x_target), y_line])
    out = np.empty((len(y_line), field_2d.shape[1]), dtype=np.float32)
    for c in range(field_2d.shape[1]):
        v_lin = griddata(xy_points, field_2d[:, c], line_pts, method="linear")
        v_nn = griddata(xy_points, field_2d[:, c], line_pts, method="nearest")
        out[:, c] = np.where(np.isnan(v_lin), v_nn, v_lin)
    return out


def plot_multistation(profiles_by_model, y_line, x_stations_d, savepath):
    """
    profiles_by_model: dict label -> array (n_stations, n_y, 4)
    """
    n_stations = len(x_stations_d)
    n_vars = 4
    fig, axes = plt.subplots(n_vars, n_stations,
                             figsize=(2.5 * n_stations, 3.6 * n_vars),
                             sharey="row")
    colours = {"Ground Truth": "black",
               "Offline INR": "tab:blue",
               "Online Naive": "tab:orange",
               "Online ER-Aggr.": "tab:green"}
    linewidths = {"Ground Truth": 3.0,
                  "Offline INR": 2.2,
                  "Online Naive": 2.2,
                  "Online ER-Aggr.": 2.2}
    linestyles = {"Ground Truth": "-",
                  "Offline INR": "--",
                  "Online Naive": "--",
                  "Online ER-Aggr.": "--"}

    for r, name in enumerate(FEATURES):
        for c, xd in enumerate(x_stations_d):
            ax = axes[r, c]
            for label, profiles in profiles_by_model.items():
                ax.plot(profiles[c, :, r], y_line,
                        color=colours[label],
                        linestyle=linestyles[label],
                        linewidth=linewidths[label],
                        label=label, alpha=0.95)
            if r == 0:
                ax.set_title(f"$x = {xd:.1f}\\,D$", fontsize=15,
                             fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{name}\n$y$ (m)", fontsize=14,
                              fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)
        axes[-1, c].set_xlabel("normalised value", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=4, fontsize=14, frameon=True,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Time-Averaged Wake Profiles at Multiple Downstream Stations",
                 fontsize=17, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.985])
    plt.savefig(savepath, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[Plot] {savepath}")


def main():
    inputs, inputs_n, targets_raw = load_dataset()
    coords_xy = inputs[:N_PTS, :2]
    y_line = np.linspace(Y_RANGE[0], Y_RANGE[1], N_Y)

    gt_tavg = targets_raw.reshape(N_TS, N_PTS, 4).mean(axis=0)

    profiles_by_model = {}
    profiles_by_model["Ground Truth"] = np.stack([
        sample_profile_at_x(coords_xy, gt_tavg, xs, y_line)
        for xs in X_STATIONS
    ])

    for label, path in MODEL_SPECS:
        m = load_model(path)
        preds_n = run_inference(m, inputs_n)
        tavg = time_average(preds_n)
        profiles_by_model[label] = np.stack([
            sample_profile_at_x(coords_xy, tavg, xs, y_line)
            for xs in X_STATIONS
        ])
        del m

    plot_multistation(
        profiles_by_model, y_line, STATIONS_D,
        os.path.join(OUT_DIR, "time_averaged_multistation_profiles.png"),
    )


if __name__ == "__main__":
    main()
