import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set clean academic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11
})

# 1. Hardcoded Class Names and Exact Percentage Metrics extracted from the data
# Metrics represent average mAP@50-95 across conditions for each class
classes = ["All", "BLid", "BTrailer", "Cabin", "FChassis", "Ggun", "Rchassis", "YTrailer", "Ylid"]

# Data points mapped to Exp0, Exp1, Exp2 based on the global evaluation patterns
exp0_data = [50.6, 50.2, 53.3, 65.7, 35.8, 59.1, 18.1, 61.2, 59.3]
exp1_data = [61.1, 66.9, 70.1, 71.9, 51.0, 71.9, 31.3, 72.1, 68.9]
exp2_data = [64.3, 67.8, 69.7, 54.4, 56.7, 78.3, 41.4, 74.8, 71.7]

y_indices = np.arange(len(classes))
bar_width = 0.25

# 2. Setup Plot Canvas (Wide layout gives horizontal bars perfect breathing room)
fig, ax = plt.subplots(figsize=(10, 7))

# Create the clustered horizontal bars
rects0 = ax.barh(y_indices + bar_width, exp0_data, bar_width, label="Experiment 0: Baseline", color="#1f77b4")
rects1 = ax.barh(y_indices, exp1_data, bar_width, label="Experiment 1: Ground Plane", color="#ff7f0e")
rects2 = ax.barh(y_indices - bar_width, exp2_data, bar_width, label="Experiment 2: Full Environment", color="#2ca02c")

# 3. Structural Polish for the Reviewer (Labels & Spacing)
ax.set_ylabel("Target Component Category", fontweight="bold")
ax.set_xlabel("Average mAP @[.50:.95]", fontweight="bold")
ax.set_title("Object Detection Performance Across Experiments", fontweight="bold", pad=15)
ax.set_yticks(y_indices)
ax.set_yticklabels(classes)
ax.set_xlim(0, 100)  # Percentage bounding box limits
ax.invert_yaxis()  # Places the crucial 'All' class summary cleanly at the top
ax.legend(
    loc="upper right",
    frameon=True,
    shadow=False,
    facecolor="white",
    edgecolor="none",
    fontsize=10,
    handlelength=1.2,
    handleheight=0.8,
    borderpad=0.5,
    labelspacing=0.3,
)

# 4. Local Text-Label Injection Framework (Solves the text overlapping completely)
def inject_local_labels(rects):
    for rect in rects:
        width = rect.get_width()
        # If the bar is too short, put text outside; if long enough, place it inside for visual layout stability
        text_padding = 1.5 if width < 10 else -5.5
        text_color = "black" if width < 10 else "white"

        ax.text(
            width + text_padding,
            rect.get_y() + rect.get_height() / 2,
            f"{width:.1f}%",
            ha="right" if width < 20 else "right",
            va="center",
            fontsize=9.5,
            color=text_color,
            fontweight="semibold"
        )


# Locally inject hardcoded percentages into individual bar profiles

inject_local_labels(rects0)
inject_local_labels(rects1)
inject_local_labels(rects2)


# Ensure tight axis fitting to avoid cropped margins in exported vector PDF
plt.tight_layout()

# Save the updated chart vector directly to your target paper repository
plt.savefig("images/classwise_comparison_fixed.pdf", dpi=300)
print("[Success] Re-rendered 'classwise_comparison_fixed.pdf' with optimized text placement profiles.")