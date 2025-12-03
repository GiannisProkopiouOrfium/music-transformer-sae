# Save this as analyze_full_grid.py
import json
import numpy as np
import matplotlib.pyplot as plt

with open(
    "steering_interventions/dual_steering/outputs/full_grid_gram_schmidt_pitch/phase3_results.json"
) as f:
    data = json.load(f)

results = [r for r in data["results"] if r["valid_samples"] > 0]

alphas_pitch = sorted(list(set(r["alpha_pitch"] for r in results)))
alphas_duration = sorted(list(set(r["alpha_duration"] for r in results)))

# Create matrices
pitch_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)
duration_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)
quality_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)

for r in results:
    i = alphas_duration.index(r["alpha_duration"])
    j = alphas_pitch.index(r["alpha_pitch"])
    pitch_matrix[i, j] = r["pitch_control"]["mean"]
    duration_matrix[i, j] = r["duration_control"]["mean"]
    if not np.isnan(r["degradation"]["total_degradation"]["mean"]):
        quality_matrix[i, j] = r["degradation"]["total_degradation"]["mean"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im1 = axes[0].imshow(pitch_matrix, cmap="RdYlBu_r", aspect="auto")
axes[0].set_title("Pitch Control (MIDI)")
axes[0].set_xlabel("Alpha Pitch")
axes[0].set_ylabel("Alpha Duration")
axes[0].set_xticks(range(len(alphas_pitch)))
axes[0].set_xticklabels([f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right")
axes[0].set_yticks(range(len(alphas_duration)))
axes[0].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(duration_matrix, cmap="viridis", aspect="auto")
axes[1].set_title("Duration Control (ticks)")
axes[1].set_xlabel("Alpha Pitch")
axes[1].set_ylabel("Alpha Duration")
axes[1].set_xticks(range(len(alphas_pitch)))
axes[1].set_xticklabels([f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right")
axes[1].set_yticks(range(len(alphas_duration)))
axes[1].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(quality_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=10)
axes[2].set_title("Quality Degradation")
axes[2].set_xlabel("Alpha Pitch")
axes[2].set_ylabel("Alpha Duration")
axes[2].set_xticks(range(len(alphas_pitch)))
axes[2].set_xticklabels([f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right")
axes[2].set_yticks(range(len(alphas_duration)))
axes[2].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig("dual_steering_heatmaps.png", dpi=300)
print("Saved heatmaps!")
