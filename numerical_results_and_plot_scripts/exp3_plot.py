import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ======================================================
# Folder with result files
# ======================================================
folder_name = "Results_exp3"
files = sorted([f for f in os.listdir(folder_name) if f.endswith(".npz")])

if not files:
    raise FileNotFoundError(f"No .npz files found in folder: {folder_name}")

# ======================================================
# Matplotlib style (same visual configuration you liked)
# ======================================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 30,
    "axes.labelweight": "bold",
    "axes.titlesize": 30,
    "axes.titleweight": "bold",
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
})

colors = plt.cm.viridis
markers = ['o', 's', 'P', 'X', '^', 'v', '<', '>', 'D', '*']

# ======================================================
# Read and organize data
# Structure:
# grouped_data[medium][(Nqubit, Npar)][distance][Fth] = reliability
# ======================================================
grouped_data = {}

for filename in files:
    path = os.path.join(folder_name, filename)
    data = np.load(path, allow_pickle=True)

    medium = str(data["medium"])
    Nqubit = int(data["Nqubit"])
    Npar = int(data["Npar"])
    d = float(data["distance_km"])
    Fth = float(data["Fth"])
    R = float(data["reliability_simulation"])

    if medium not in grouped_data:
        grouped_data[medium] = {}

    key = (Nqubit, Npar)
    if key not in grouped_data[medium]:
        grouped_data[medium][key] = {}

    if d not in grouped_data[medium][key]:
        grouped_data[medium][key][d] = {}

    grouped_data[medium][key][d][Fth] = R

mediums = sorted(grouped_data.keys())
if "fiber" in mediums and "FSO" in mediums:
    mediums = ["fiber", "FSO"]

# Extract common parameter lists
all_keys = set()
for medium in grouped_data:
    all_keys.update(grouped_data[medium].keys())

N_qubits_list = sorted(list(set(k[0] for k in all_keys)))
N_parallel_list = sorted(list(set(k[1] for k in all_keys)))

all_distances = sorted(list(set(
    float(np.load(os.path.join(folder_name, f), allow_pickle=True)["distance_km"])
    for f in files
)))
all_fths = sorted(list(set(
    float(np.load(os.path.join(folder_name, f), allow_pickle=True)["Fth"])
    for f in files
)))

xticks = np.round(np.arange(min(all_fths), max(all_fths) + 0.001, 0.04), 2)
yticks = [0.00, 0.25, 0.50, 0.75, 1.00]


# ======================================================
# Helper to draw one medium figure
# ======================================================
def plot_medium(medium_name, output_prefix):
    fig, axs = plt.subplots(
        len(N_parallel_list),
        len(N_qubits_list),
        figsize=(24, 20),
        squeeze=False
    )

    plt.subplots_adjust(
        left=0.06,
        right=0.80,
        top=0.95,
        bottom=0.08,
        hspace=0.42,
        wspace=0.32
    )

    medium_data = grouped_data[medium_name]

    for (Nqubit, Npar), distance_dict in medium_data.items():
        i = N_parallel_list.index(Npar)
        j = N_qubits_list.index(Nqubit)
        ax = axs[i, j]

        distances_sorted = sorted(distance_dict.keys())

        for k, d in enumerate(distances_sorted):
            fths_sorted = sorted(distance_dict[d].keys())
            reliabilities = [distance_dict[d][fth] for fth in fths_sorted]

            color = colors(k / max(1, len(distances_sorted) - 1))
            marker = markers[k % len(markers)]

            ax.plot(
                fths_sorted,
                reliabilities,
                color=color,
                marker=marker,
                linestyle='-',
                linewidth=2.2,
                markersize=6.5,
                label=rf"$d={d:.2f}$ km"
            )

        ax.set_xlabel(r"Umbral de fidelidad ($F_{\mathrm{th}}$)")
        ax.set_ylabel(r"Fiabilidad ($R$)")
        ax.set_xlim(min(all_fths), max(all_fths))
        ax.set_xticks(xticks)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(yticks)
        ax.grid(True, alpha=0.3)

        rect_height = 0.13
        rect = patches.Rectangle(
            (0, 1.0), 1.0, rect_height,
            transform=ax.transAxes,
            facecolor='lightgray',
            edgecolor='black',
            linewidth=1.4,
            clip_on=False
        )
        ax.add_patch(rect)

        ax.text(
            0.5,
            1.0 + rect_height / 2,
            rf"$N_{{\mathrm{{qubit}}}}={Nqubit},\ N_{{\mathrm{{par}}}}={Npar}$",
            ha='center',
            va='center',
            fontsize=22,
            fontweight='bold',
            transform=ax.transAxes,
            clip_on=False
        )

    handles, labels = axs[0, -1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor="gray",
        ncol=1
    )

    png_name = f"{output_prefix}.png"
    pdf_name = f"{output_prefix}.pdf"

    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_name, bbox_inches="tight")
    plt.show()


# ======================================================
# Delta R = R_FSO - R_fiber
# Structure:
# delta_data[(Nqubit, Npar)][distance][Fth] = deltaR
# ======================================================
def build_delta_data():
    if "fiber" not in grouped_data or "FSO" not in grouped_data:
        print("Delta plot skipped: both 'fiber' and 'FSO' are required.")
        return None

    delta_data = {}

    common_keys = set(grouped_data["fiber"].keys()).intersection(set(grouped_data["FSO"].keys()))

    for key in common_keys:
        delta_data[key] = {}
        fiber_dist = grouped_data["fiber"][key]
        fso_dist = grouped_data["FSO"][key]

        common_distances = set(fiber_dist.keys()).intersection(set(fso_dist.keys()))

        for d in sorted(common_distances):
            delta_data[key][d] = {}
            common_fths = set(fiber_dist[d].keys()).intersection(set(fso_dist[d].keys()))

            for fth in sorted(common_fths):
                delta_data[key][d][fth] = fso_dist[d][fth] - fiber_dist[d][fth]

    return delta_data


def plot_delta(delta_data, output_prefix):
    fig, axs = plt.subplots(
        len(N_parallel_list),
        len(N_qubits_list),
        figsize=(24, 20),
        squeeze=False
    )

    plt.subplots_adjust(
        left=0.06,
        right=0.80,
        top=0.95,
        bottom=0.08,
        hspace=0.42,
        wspace=0.32
    )

    for (Nqubit, Npar), distance_dict in delta_data.items():
        i = N_parallel_list.index(Npar)
        j = N_qubits_list.index(Nqubit)
        ax = axs[i, j]

        distances_sorted = sorted(distance_dict.keys())

        for k, d in enumerate(distances_sorted):
            fths_sorted = sorted(distance_dict[d].keys())
            delta_vals = [distance_dict[d][fth] for fth in fths_sorted]

            color = colors(k / max(1, len(distances_sorted) - 1))
            marker = markers[k % len(markers)]

            ax.plot(
                fths_sorted,
                delta_vals,
                color=color,
                marker=marker,
                linestyle='-',
                linewidth=2.2,
                markersize=6.5,
                label=rf"$d={d:.2f}$ km"
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel(r"Umbral de fidelidad ($F_{\mathrm{th}}$)")
        ax.set_ylabel(r"$\Delta R = R_{\mathrm{FSO}} - R_{\mathrm{fiber}}$")
        ax.set_xlim(min(all_fths), max(all_fths))
        ax.set_xticks(xticks)
        ax.grid(True, alpha=0.3)

        rect_height = 0.13
        rect = patches.Rectangle(
            (0, 1.0), 1.0, rect_height,
            transform=ax.transAxes,
            facecolor='lightgray',
            edgecolor='black',
            linewidth=1.4,
            clip_on=False
        )
        ax.add_patch(rect)

        ax.text(
            0.5,
            1.0 + rect_height / 2,
            rf"$N_{{\mathrm{{qubit}}}}={Nqubit},\ N_{{\mathrm{{par}}}}={Npar}$",
            ha='center',
            va='center',
            fontsize=22,
            fontweight='bold',
            transform=ax.transAxes,
            clip_on=False
        )

    handles, labels = axs[0, -1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor="gray",
        ncol=1
    )

    png_name = f"{output_prefix}.png"
    pdf_name = f"{output_prefix}.pdf"

    plt.savefig(png_name, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_name, bbox_inches="tight")
    plt.show()


# ======================================================
# Run plots
# ======================================================


delta_data = build_delta_data()
if delta_data is not None:
    plot_delta(delta_data, "exp3_deltaR_vs_fth")