cam1_exp0 = [43.1, 45.5, 35.7]  # Exp0 L1, M1, H1
cam1_exp1 = [58.3, 64.9, 55.3]  # Exp1 L1, M1, H1
cam1_exp2 = [61.4, 65.6, 50.0]  # Exp2 L1, M1, H1

# Camera 2: Off-Axis Orbital Geometry (Angled Side View)
cam2_exp0 = [55.8, 62.1, 61.2]  # Exp0 L2, M2, H2
cam2_exp1 = [67.4, 63.8, 60.2]  # Exp1 L2, M2, H2
cam2_exp2 = [60.9, 72.5, 75.8]  # Exp2 L2, M2, H2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply clean academic typesetting
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# 1. Corrected Data Arrays (Ordered strictly: [Low, Medium, High] per experiment)
# Camera 1: Coaxial Illumination Geometry (Direct Top-Down View)
cam1_exp0 = [43.1, 45.5, 35.7]  # Exp0 L1, M1, H1
cam1_exp1 = [58.3, 64.9, 55.3]  # Exp1 L1, M1, H1
cam1_exp2 = [61.4, 65.6, 50.0]  # Exp2 L1, M1, H1

# Camera 2: Off-Axis Orbital Geometry (Angled Side View)
cam2_exp0 = [55.8, 62.1, 61.2]  # Exp0 L2, M2, H2
cam2_exp1 = [67.4, 63.8, 60.2]  # Exp1 L2, M2, H2
cam2_exp2 = [60.9, 72.5, 75.8]  # Exp2 L2, M2, H2

# X-axis master categories mapped chronologically
categories = ["Experiment 0\n(Object Only)", "Experiment 1\n(Ground Plane)", "Experiment 2\n(Full Env.)"]
x_indices = np.arange(len(categories))
bar_width = 0.24

# 2. Setup a clean 2-Panel Stacked Vertically Layout
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # [Low = Blue, Medium = Orange, High = Green]


def render_panel(ax, low_data, med_data, high_data, title):
    # Render bars side-by-side inside each experiment category block
    b1 = ax.bar(x_indices - bar_width, low_data, bar_width, label="Low Intensity", color=colors[0])
    b2 = ax.bar(x_indices, med_data, bar_width, label="Medium Intensity", color=colors[1])
    b3 = ax.bar(x_indices + bar_width, high_data, bar_width, label="High Intensity", color=colors[2])

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_ylabel("Detection Performance (mAP @ 50-95 %)", fontweight="medium")
    ax.set_ylim(20, 80)

    # Inject exact text labels on top of each bar for high legibility
    for series in [b1, b2, b3]:
        for bar in series:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5, f"{height:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, fontweight="semibold")


# Render Panel 1: Camera 1 (Top-Down View)
render_panel(ax1, [cam1_exp0[0], cam1_exp1[0], cam1_exp2[0]],
             [cam1_exp0[1], cam1_exp1[1], cam1_exp2[1]],
             [cam1_exp0[2], cam1_exp1[2], cam1_exp2[2]],
             "Camera 1: Coaxial Illumination Geometry (Direct Top-Down Perspective)")
ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

# Render Panel 2: Camera 2 (Orbital View)
render_panel(ax2, [cam2_exp0[0], cam2_exp1[0], cam2_exp2[0]],
             [cam2_exp0[1], cam2_exp1[1], cam2_exp2[1]],
             [cam2_exp0[2], cam2_exp1[2], cam2_exp2[2]],
             "Camera 2: Off-Axis Orbital Geometry (Angled Side Perspective)")

# Set global layouts
ax2.set_xlabel("Synthetic Training Environment Complexity Progression", fontweight="bold", labelpad=10)
ax2.set_xticks(x_indices)
ax2.set_xticklabels(categories)

plt.tight_layout()
plt.savefig("images/master_experiment_comparison_fixed.pdf", dpi=300)
print("[Success] Re-exported 'master_experiment_comparison_fixed.pdf' with correct Low-Medium-High column tracking.")