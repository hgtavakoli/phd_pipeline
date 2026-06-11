# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
# # Set clean academic style matching the previous plot
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
#     "axes.labelsize": 11,
#     "axes.titlesize": 12,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10
# })
#
# # 1. Hardcoded Experiment Profiles and Exact Percentage Metrics from Figure 2
# experiments = ["Exp2 L", "Exp2 M", "Exp2 H", "Exp1 L", "Exp1 M", "Exp1 H", "Exp0 L", "Exp0 M", "Exp0 H"]
#
# # Camera 1 Data Profiles (Coaxial Illumination)
# cam1_domain_specific = [65.9, 68.9, 55.6, 65.8, 69.8, 59.8, 58.5, 47.9, 39.8]
# cam1_expanded_dataset = [61.0, 66.0, 50.0, 58.0, 64.8, 55.0, 43.0, 46.0, 35.7]
#
# # Camera 2 Data Profiles (Off-Axis Orbital Geometry)
# cam2_domain_specific = [66.9, 75.0, 78.0, 68.6, 66.9, 65.1, 60.4, 65.0, 66.0]
# cam2_expanded_dataset = [60.0, 73.0, 76.0, 67.4, 63.8, 60.0, 55.6, 62.0, 61.0]
#
# y_indices = np.arange(len(experiments))
# bar_width = 0.35
#
# # 2. Setup a 2-Panel Stacked Row Layout (Perfect for separating the two camera views)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
#
# # --- PANEL 1: CAMERA 1 (Top-Down View) ---
# rects1_ds = ax1.barh(y_indices + bar_width / 2, cam1_domain_specific, bar_width,
#                      label="Cam1: Domain-Specific Real Dataset", color="#4682b4")
# rects1_ex = ax1.barh(y_indices - bar_width / 2, cam1_expanded_dataset, bar_width,
#                      label="Cam1: Expanded Real Dataset (with Background Clutter)", color="#f08080")
#
# ax1.set_title("Camera 1: Coaxial Illumination Configurations (Top-Down Perspective)", fontweight="bold")
# ax1.set_yticks(y_indices)
# ax1.set_yticklabels(experiments)
# ax1.set_xlim(0, 100)
# ax1.invert_yaxis()  # Matches the sequential order of your text (Exp2 -> Exp1 -> Exp0)
# ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
#
# # --- PANEL 2: CAMERA 2 (Orbital View) ---
# rects2_ds = ax2.barh(y_indices + bar_width / 2, cam2_domain_specific, bar_width,
#                      label="Cam2: Domain-Specific Real Dataset", color="#000080")
# rects2_ex = ax2.barh(y_indices - bar_width / 2, cam2_expanded_dataset, bar_width,
#                      label="Cam2: Expanded Real Dataset (with Background Clutter)", color="#8b0000")
#
# ax2.set_title("Camera 2: Off-Axis Orbital Geometry Configurations (Angled Perspective)", fontweight="bold")
# ax2.set_yticks(y_indices)
# ax2.set_yticklabels(experiments)
# ax2.set_xlim(0, 100)
# ax2.invert_yaxis()
# ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
#
# # Set global shared X-axis label at the bottom panel
# ax2.set_xlabel("Model Detection Performance (mAP @ 50-95 %)", fontweight="bold", labelpad=10)
#
#
# # 3. Text Label Injection Engine (Ensures percentages never blur together)
# def inject_panel_labels(ax, rects):
#     for rect in rects:
#         width = rect.get_width()
#         # Shifts shorter values to the right of the bar edge, longer ones safely inside
#         text_padding = 1.2 if width < 25 else -4.8
#         text_color = "black" if width < 25 else "white"
#
#         ax.text(
#             width + text_padding,
#             rect.get_y() + rect.get_height() / 2,
#             f"{width:.1f}%",
#             ha="left" if width < 25 else "right",
#             va="center",
#             fontsize=9,
#             color=text_color,
#             fontweight="semibold"
#         )
#
#
# # Inject label layout configurations locally to all four data series
# inject_panel_labels(ax1, rects1_ds)
# inject_panel_labels(ax1, rects1_ex)
# inject_panel_labels(ax2, rects2_ds)
# inject_panel_labels(ax2, rects2_ex)
#
# # Adjust layouts to avoid clipping titles or boundaries
# plt.tight_layout()
#
# # Save output graphic file directly to your LaTeX workspace
# plt.savefig("images/ordered_chart2_fixed.pdf", dpi=300)
# print("[Success] Re-rendered 'ordered_chart2_fixed.pdf' using a split-panel landscape orientation.")

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
#     "axes.labelsize": 12,
#     "axes.titlesize": 13,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 11
# })
#
# experiments = ["Exp2 L", "Exp2 M", "Exp2 H", "Exp1 L", "Exp1 M", "Exp1 H", "Exp0 L", "Exp0 M", "Exp0 H"]
#
# # Calculate the actual drop (Domain-Specific minus Expanded)
# # Camera 1 drops significantly due to coaxial glare and clutter confusion
# cam1_drop = [4.9, 2.9, 5.6, 7.8, 5.0, 4.9, 15.5, 1.9, 4.1]
# # Camera 2 maintains minimal drops, proving structural robustness
# cam2_drop = [6.9, 2.0, 2.0, 1.2, 3.1, 5.1,  4.8, 3.0, 5.0]
#
# y_indices = np.arange(len(experiments))
# bar_width = 0.35
#
# fig, ax = plt.subplots(figsize=(9, 6))
#
# # Plot the drops side-by-side
# rects1 = ax.barh(y_indices + bar_width/2, cam1_drop, bar_width,
#                  label="Camera 1: Coaxial Illumination (Direct Top-Down)", color="#d95f02")
# rects2 = ax.barh(y_indices - bar_width/2, cam2_drop, bar_width,
#                  label="Camera 2: Off-Axis Orbital Geometry (Angled)", color="#1b9e77")
#
# ax.set_ylabel("Environmental Configuration", fontweight="bold")
# ax.set_xlabel("Performance Degradation / Drop in mAP @ 50-95 (%)", fontweight="bold")
# ax.set_title("Model Vulnerability to Clutter: Performance Drop When Backgrounds Are Added", fontweight="bold", pad=15)
# ax.set_yticks(y_indices)
# ax.set_yticklabels(experiments)
# ax.set_xlim(0, 20) # Gives plenty of room on the right side for a completely clean legend
# ax.invert_yaxis()
#
# # Clean, non-overlapping legend placement on the wide-open right half of the chart
# ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
#
# # Inject data labels to the right of the bars
# def inject_delta_labels(rects):
#     for rect in rects:
#         width = rect.get_width()
#         ax.text(
#             width + 0.3,
#             rect.get_y() + rect.get_height() / 2,
#             f"-{width:.1f}%",
#             ha="left", va="center", fontsize=9, fontweight="semibold"
#         )
#
# inject_delta_labels(rects1)
# inject_delta_labels(rects2)
#
# plt.tight_layout()
# plt.savefig("images/robustness_delta_chart.pdf", dpi=300)
# print("[Success] Generated 'robustness_delta_chart.pdf' displaying explicit clutter vulnerability.")


# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
#
# # Clean academic style
# sns.set_theme(style="whitegrid")
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
#     "axes.labelsize": 12,
#     "axes.titlesize": 13,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 11
# })
#
# experiments = ["Exp2 L", "Exp2 M", "Exp2 H", "Exp1 L", "Exp1 M", "Exp1 H", "Exp0 L", "Exp0 M", "Exp0 H"]
#
# # Raw performance drop percentages (Domain-Specific minus Expanded Real Dataset)
# cam1_drop = [4.9, 2.9, 5.6, 7.8, 5.0, 4.9, 15.5, 1.9, 4.1]
# cam2_drop = [6.9, 2.0, 2.0, 1.2, 3.1, 5.1,  4.8, 3.0, 5.0]
#
# # Calculate averages for each experiment block across all lighting parameters (L, M, H)
# # Formatted as: index slicing matching the 'experiments' layout array sequence
# cam1_avg_exp2 = np.mean(cam1_drop[0:3])  # Exp2 L, M, H
# cam1_avg_exp1 = np.mean(cam1_drop[3:6])  # Exp1 L, M, H
# cam1_avg_exp0 = np.mean(cam1_drop[6:9])  # Exp0 L, M, H
#
# cam2_avg_exp2 = np.mean(cam2_drop[0:3])  # Exp2 L, M, H
# cam2_avg_exp1 = np.mean(cam2_drop[3:6])  # Exp1 L, M, H
# cam2_avg_exp0 = np.mean(cam2_drop[6:9])  # Exp0 L, M, H
#
# y_indices = np.arange(len(experiments))
# bar_width = 0.35
#
# # Setup wide canvas to safely host the data and text summary box
# fig, ax = plt.subplots(figsize=(11, 6.5))
#
# rects1 = ax.barh(y_indices + bar_width/2, cam1_drop, bar_width,
#                  label="Camera 1: Coaxial Illumination (Direct View)", color="#d95f02")
# rects2 = ax.barh(y_indices - bar_width/2, cam2_drop, bar_width,
#                  label="Camera 2: Off-Axis Orbital (Angled View)", color="#1b9e77")
#
# ax.set_ylabel("Environmental Configuration", fontweight="bold")
# ax.set_xlabel("Performance Degradation / Drop in mAP @ 50-95 (%)", fontweight="bold")
# ax.set_title("Model Vulnerability to Clutter: Degradation Across Target Scenes", fontweight="bold", pad=15)
# ax.set_yticks(y_indices)
# ax.set_yticklabels(experiments)
# ax.set_xlim(0, 24) # Expanded horizontal ceiling to accommodate metadata text box securely
# ax.invert_yaxis()
#
# # Adjust primary legend location to clear bar space
# ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, facecolor="white", edgecolor="none")
#
# # Inject individual delta labels directly to the right edge of bars
# def inject_delta_labels(rects):
#     for rect in rects:
#         width = rect.get_width()
#         ax.text(
#             width + 0.3,
#             rect.get_y() + rect.get_height() / 2,
#             f"-{width:.1f}%",
#             ha="left", va="center", fontsize=9, fontweight="semibold"
#         )
#
# inject_delta_labels(rects1)
# inject_delta_labels(rects2)
#
# # Create an elegant academic text summary panel on the right side of the canvas
# summary_text = (
#     "  Averaged Experiment Vulnerability  \n"
#     "====================================\n"
#     f"Experiment 0 (Baseline Object-Only):\n"
#     f"  • Camera 1 Average Drop: {cam1_avg_exp0:.1f}%\n"
#     f"  • Camera 2 Average Drop: {cam2_avg_exp0:.1f}%\n\n"
#     f"Experiment 1 (Ground Plane Added):\n"
#     f"  • Camera 1 Average Drop: {cam1_avg_exp1:.1f}%\n"
#     f"  • Camera 2 Average Drop: {cam2_avg_exp1:.1f}%\n\n"
#     f"Experiment 2 (Full Complex Env.):\n"
#     f"  • Camera 1 Average Drop: {cam1_avg_exp2:.1f}%\n"
#     f"  • Camera 2 Average Drop: {cam2_avg_exp2:.1f}%\n"
#     "====================================\n"
#     "Conclusion: Camera 2 consistently yields\n"
#     "higher robustness against false positives\n"
#     "across all experimental progressions."
# )
#
# # Render the text box inside the plot coordinates safely out of the bar trajectories
# ax.text(
#     14.8, 4.5, summary_text,
#     fontsize=9.5, fontfamily="monospace", va="center", ha="left",
#     bbox=dict(boxstyle="round,pad=0.6", facecolor="#fdfdfd", edgecolor="#cccccc", alpha=0.95)
# )
#
# plt.tight_layout()
# plt.savefig("images/robustness_delta_chart_final.pdf", dpi=300)
# print("[Success] Exported 'robustness_delta_chart_final.pdf' featuring calculated summary metrics.")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set clean academic style matching your manuscript
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# 1. Calculated overall average performance drops across L, M, H
experiment_labels = [
    "Exp 0: Baseline",
    "Exp 1: Ground Plane",
    "Exp 2: Full Environment"
]

# Hardcoded calculated averages based on your raw experimental data
camera1_averages = [7.2, 5.9, 4.5]  # Lower numbers = less drop = more robust
camera2_averages = [4.3, 3.1, 3.6]

x_indices = np.arange(len(experiment_labels))
bar_width = 0.35

# 2. Setup standard vertical bar chart for overall perspective
fig, ax = plt.subplots(figsize=(8, 5.5))

rects1 = ax.bar(x_indices - bar_width/2, camera1_averages, bar_width,
                label="Camera 1: Coaxial Illumination (Direct View)", color="#d95f02")
rects2 = ax.bar(x_indices + bar_width/2, camera2_averages, bar_width,
                label="Camera 2: Off-Axis Orbital (Angled View)", color="#1b9e77")

# 3. Text and Label Polish
ax.set_ylabel("Average Performance Drop \n (Clutter Vulnerability)", fontweight="bold", color="black")
ax.set_xlabel("Synthetic Data Scene Complexity Progression", fontweight="bold", labelpad=10, color="black")
ax.set_title("Overall Model Vulnerability to Background Clutter", fontweight="bold", pad=15, color="black")
ax.set_xticks(x_indices)
ax.set_xticklabels(experiment_labels)
ax.set_ylim(0, 10)  # Clean upper ceiling leaving space for the legend

# Place legend in a highly readable, empty corner
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

# 4. Inject explicit data tags on top of each bar
def inject_bar_tags(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + 0.2,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="semibold"
        )

inject_bar_tags(rects1)
inject_bar_tags(rects2)

# Ensure tight layouts to avoid border clipping in final PDF export
plt.tight_layout()
plt.savefig("images/overall_robustness_summary.pdf", dpi=300)
print("[Success] Exported 'overall_robustness_summary.pdf' for a high-level project perspective.")