import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ======================================================
# Folders containing the result files
# Data with QR are reused from Experiment 1
# Data without QR are generated in Experiment 4
# ======================================================
folder_qr = "Results_exp1"
folder_noqr = "Results_exp4"

files_qr = sorted([f for f in os.listdir(folder_qr) if f.endswith(".npz")])
files_noqr = sorted([f for f in os.listdir(folder_noqr) if f.endswith(".npz")])

if not files_qr:
    raise FileNotFoundError(f"No .npz files found in folder: {folder_qr}")
if not files_noqr:
    raise FileNotFoundError(f"No .npz files found in folder: {folder_noqr}")

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
    "legend.fontsize": 20,
})

colors = plt.cm.viridis
markers_qr = ['o', 's', 'P', 'X', '^', 'v', '<', '>', 'D', '*']
markers_noqr = ['d', 'h', '8', 'p', '>', '<', '^', 'v', 'H', '*']

# ======================================================
# Data loading and organization
# Structure:
# qr_data[(Nqubit, Npar)][Fth][distance] = reliability
# noqr_data[(Nqubit, Npar)][Fth][distance] = reliability
# ======================================================
qr_data = {}
noqr_data = {}

for filename in files_qr:
    path = os.path.join(folder_qr, filename)
    data = np.load(path, allow_pickle=True)

    Nqubit = int(data["Nqubit"])
    Npar = int(data["Npar"])
    Fth = float(data["Fth"])
    d = float(data["distance_km"])
    R = float(data["reliability_simulation"])

    key = (Nqubit, Npar)
    qr_data.setdefault(key, {})
    qr_data[key].setdefault(Fth, {})
    qr_data[key][Fth][d] = R

for filename in files_noqr:
    path = os.path.join(folder_noqr, filename)
    data = np.load(path, allow_pickle=True)

    Nqubit = int(data["Nqubit"])
    Npar = int(data["Npar"])
    Fth = float(data["Fth"])
    d = float(data["distance_km"])
    R = float(data["reliability_simulation"])

    key = (Nqubit, Npar)
    noqr_data.setdefault(key, {})
    noqr_data[key].setdefault(Fth, {})
    noqr_data[key][Fth][d] = R

# Common parameter sets
common_keys = sorted(set(qr_data.keys()).intersection(set(noqr_data.keys())))
if not common_keys:
    raise ValueError("No common (Nqubit, Npar) combinations between Exp1 and Exp4 data.")

N_qubits_list = sorted(list(set(k[0] for k in common_keys)))
N_parallel_list = sorted(list(set(k[1] for k in common_keys)))

common_fths = sorted(list(
    set.intersection(
        *[set(qr_data[k].keys()).intersection(set(noqr_data[k].keys())) for k in common_keys]
    )
))
if not common_fths:
    raise ValueError("No common Fth values between Exp1 and Exp4 data.")

common_distances = sorted(list(
    set.intersection(
        *[
            set(d for fth in qr_data[k] for d in qr_data[k][fth].keys()).intersection(
                set(d for fth in noqr_data[k] for d in noqr_data[k][fth].keys())
            )
            for k in common_keys
        ]
    )
))
if not common_distances:
    raise ValueError("No common distance values between Exp1 and Exp4 data.")

xticks = np.round(np.arange(min(common_distances), max(common_distances) + 0.001, 0.3), 2)
yticks = [0.00, 0.25, 0.50, 0.75, 1.00]


# ======================================================
# Delta R = R_QR - R_noQR
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

for key in common_keys:
    Nqubit, Npar = key
    i = N_parallel_list.index(Npar)
    j = N_qubits_list.index(Nqubit)
    ax = axs[i, j]

    fths_sorted = sorted(set(qr_data[key].keys()).intersection(set(noqr_data[key].keys())))

    for k, Fth in enumerate(fths_sorted):
        d_common = sorted(set(qr_data[key][Fth].keys()).intersection(set(noqr_data[key][Fth].keys())))
        delta_R = [qr_data[key][Fth][d] - noqr_data[key][Fth][d] for d in d_common]

        color = colors(k / max(1, len(fths_sorted) - 1))
        marker = markers_qr[k % len(markers_qr)]

        ax.plot(
            d_common,
            delta_R,
            color=color,
            marker=marker,
            linestyle='-',
            linewidth=2.2,
            markersize=6.5,
            label=rf"$F_{{\mathrm{{th}}}}={Fth:.2f}$"
        )

    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.4)
    ax.set_xlabel(r"Distancia ($d$) [km]")
    ax.set_ylabel(r"$\Delta R = R_{\mathrm{QR}} - R_{\mathrm{noQR}}$")
    ax.set_xlim(min(common_distances), max(common_distances))
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

output_png = "exp4_deltaR_qr_minus_noqr.png"
output_pdf = "exp4_deltaR_qr_minus_noqr.pdf"

plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.savefig(output_pdf, bbox_inches="tight")
plt.show()