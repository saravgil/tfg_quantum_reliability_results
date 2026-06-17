import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ======================================================
# Folders containing the result files
# ======================================================
folder_name = "Results_exp5"
files = sorted([f for f in os.listdir(folder_name) if f.endswith(".npz")])

if not files:
    raise FileNotFoundError(f"No .npz files found in folder: {folder_name}")

# ======================================================
# Matplotlib style
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
# Data loading and organization
# Structure:
# realistic_data[(Nqubit, Npar)][distance][Fth] = R
# ideal_data[(Nqubit, Npar)][distance][Fth] = R
# ======================================================
realistic_data = {}
ideal_data = {}

for filename in files:
    path = os.path.join(folder_name, filename)
    data = np.load(path, allow_pickle=True)

    scenario = str(data["scenario"])
    Nqubit = int(data["Nqubit"])
    Npar = int(data["Npar"])
    d = float(data["distance_km"])
    Fth = float(data["Fth"])
    R = float(data["reliability_simulation"])

    key = (Nqubit, Npar)

    if scenario == "realistic":
        realistic_data.setdefault(key, {})
        realistic_data[key].setdefault(d, {})
        realistic_data[key][d][Fth] = R

    elif scenario == "ideal":
        ideal_data.setdefault(key, {})
        ideal_data[key].setdefault(d, {})
        ideal_data[key][d][Fth] = R

# ======================================================
# Common parameter sets
# ======================================================
common_keys = sorted(set(realistic_data.keys()).intersection(set(ideal_data.keys())))
if not common_keys:
    raise ValueError("No common (Nqubit, Npar) combinations between realistic and ideal data.")

N_qubits_list = sorted(list(set(k[0] for k in common_keys)))
N_parallel_list = sorted(list(set(k[1] for k in common_keys)))

# Common distances and fidelities
all_distances = sorted(list(set(
    d for key in common_keys for d in realistic_data[key].keys()
).intersection(set(
    d for key in common_keys for d in ideal_data[key].keys()
))))

all_fths = sorted(list(set(
    fth
    for key in common_keys
    for d in realistic_data[key]
    for fth in realistic_data[key][d].keys()
).intersection(set(
    fth
    for key in common_keys
    for d in ideal_data[key]
    for fth in ideal_data[key][d].keys()
))))

xticks = np.round(np.arange(min(all_fths), max(all_fths) + 0.001, 0.04), 2)

# ======================================================
# Creation of the subplot grid
# Rows = Npar, columns = Nqubit
# ======================================================
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

# ======================================================
# Representation of each block (Nqubit, Npar)
# ======================================================
for key in common_keys:
    Nqubit, Npar = key
    i = N_parallel_list.index(Npar)
    j = N_qubits_list.index(Nqubit)
    ax = axs[i, j]

    distances_sorted = sorted(set(realistic_data[key].keys()).intersection(set(ideal_data[key].keys())))

    for k, d in enumerate(distances_sorted):
        common_fths = sorted(
            set(realistic_data[key][d].keys()).intersection(set(ideal_data[key][d].keys()))
        )

        delta_R = [
            ideal_data[key][d][fth] - realistic_data[key][d][fth]
            for fth in common_fths
        ]

        color = colors(k / max(1, len(distances_sorted) - 1))
        marker = markers[k % len(markers)]

        ax.plot(
            common_fths,
            delta_R,
            color=color,
            marker=marker,
            linestyle='-',
            linewidth=2.2,
            markersize=6.5,
            label=rf"$d={d:.2f}$ km"
        )

    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.4)

    ax.set_xlabel(r"Umbral de fidelidad ($F_{\mathrm{th}}$)")
    ax.set_ylabel(r"$\Delta R = R_{\mathrm{ideal}} - R_{\mathrm{realistic}}$")
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


output_png = "exp5_deltaR_ideal_minus_realistic.png"
output_pdf = "exp5_deltaR_ideal_minus_realistic.pdf"

plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, bbox_inches="tight")
plt.show()