"""Generate all paper and presentation figures from embedded experimental data.

No GPU or model loading needed — all data is hardcoded from completed experiments.
Generates PNG (300dpi) + PDF for each figure.

Usage:
    python pid_steering/plot_paper_figures.py [--output_dir pid_figs]

Figures generated:
    1. spatial_error_convergence     — ⟨e(0),e(k)⟩ across 12 layers (P/PI/PID)
    2. spatial_hook_ablation         — Grouped bars: PID δ vs P δ per config
    3. temporal_ablation_lambda      — Avg λ bars for P/PI/PID (temporal)
    4. spatial_fmd_comparison        — FMD at α=1.0 for both concepts
    5. temporal_fmd_comparison       — Temporal FMD: PID vs Static vs Baseline
    6. dual_steering_comparison      — Dual unconditioned (tm=2.0)
    7. dual_conditioned_comparison   — 4-scenario conditioned dual
    8. single_attribute_response     — Pitch/Duration vs α response curves
    9. gain_grid_search              — Concept-dependent gain visualization
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# EMBEDDED EXPERIMENTAL DATA
# ══════════════════════════════════════════════════════════════════════

# --- Error convergence ⟨e(0), e(k)⟩ ---
LAYERS = list(range(12))
ERROR_CONV_PITCH = {
    "P": [
        1.0,
        -0.2324,
        1.9136,
        0.3693,
        0.0641,
        0.0588,
        0.3492,
        0.0666,
        0.1556,
        0.1330,
        1.2441,
        -0.4823,
    ],
    "PI": [
        1.0,
        -0.2324,
        1.9293,
        0.3979,
        0.0646,
        0.1197,
        0.3582,
        0.1685,
        0.1872,
        0.4601,
        1.2788,
        -1.1416,
    ],
    "PID": [
        1.0,
        -0.2324,
        1.9272,
        0.4041,
        0.0642,
        0.1187,
        0.3579,
        0.1667,
        0.1873,
        0.4603,
        1.2786,
        -1.1500,
    ],
}
ERROR_CONV_DUR = {
    "P": [
        1.0,
        -0.1240,
        2.1418,
        0.4096,
        0.0658,
        0.1078,
        0.1286,
        0.2016,
        0.2994,
        0.2900,
        2.0349,
        -0.3101,
    ],
    "PI": [
        1.0,
        -0.1240,
        2.1443,
        0.4275,
        0.0668,
        0.1278,
        0.1305,
        0.2620,
        0.3086,
        0.4468,
        2.1138,
        -0.3664,
    ],
    "PID": [
        1.0,
        -0.1240,
        2.1432,
        0.4371,
        0.0667,
        0.1285,
        0.1304,
        0.2641,
        0.3088,
        0.4480,
        2.1154,
        -0.3725,
    ],
}

# --- Hook ablation (averaged across both concepts & α∈{1.0, 1.5}) ---
HOOK_CONFIGS = ["all_12", "mid_deep", "deep_only", "attn_only", "layer_10"]
HOOK_LAYERS = [12, 8, 4, 6, 1]
HOOK_PID_DELTA = [39.5, 39.8, 41.3, 38.1, 33.4]
HOOK_PID_DEG = [2.65, 3.07, 3.19, 3.59, 3.32]
HOOK_P_DEG = [3.21, 3.62, 4.46, 3.77, 6.23]
HOOK_DEG_REDUCTION = [17, 15, 28, 5, 47]  # percent

# --- Temporal ablation (avg λ) ---
TEMPORAL_ABLATION = {"P-only": 0.664, "PI": 1.136, "PID": 1.158}

# --- Spatial FMD ---
SPATIAL_FMD_ALPHAS = [0.5, 1.0, 1.5, 2.0]
SPATIAL_FMD_PITCH = {
    "Baseline": [412.4, 414.5, 413.8, 417.6],
    "P-only": [464.3, 490.0, 494.9, 516.1],
    "PI": [480.7, 508.9, 512.1, 518.4],
    "PID": [479.1, 506.3, 512.6, 518.6],
}
SPATIAL_FMD_DUR = {
    "Baseline": [387.3, 381.6, 382.1, 394.2],
    "P-only": [409.4, 471.9, 483.2, 474.6],
    "PI": [404.9, 462.9, 480.4, 475.4],
    "PID": [412.4, 456.0, 475.0, 474.8],
}

# --- Temporal FMD ---
TEMPORAL_FMD = {
    "Pitch": {"PID": 461.9, "Static SAS": 487.7, "Baseline": 381.5},
    "Duration": {"PID": 501.2, "Static SAS": 525.9, "Baseline": 385.3},
}

# --- Dual unconditioned (tm=2.0) ---
DUAL_UNCOND = {
    "PID": {"pitch": 71.77, "dur": 22.61, "delta": 0.47, "success": 75},
    "Static SAS": {"pitch": 77.05, "dur": 19.57, "delta": 2.19, "success": 95},
    "Baseline": {"pitch": 67.71, "dur": 7.47, "delta": 3.79, "success": 0},
}

# --- Dual conditioned (4 scenarios) ---
DUAL_COND_SCENARIOS = [
    "Low/Short→\nHigh/Long",
    "High/Long→\nLow/Short",
    "Low/Long→\nHigh/Short",
    "High/Short→\nLow/Long",
]
DUAL_COND_PID_SUCCESS = [80, 95, 80, 100]
DUAL_COND_PID_DELTA = [4.13, 5.21, 2.36, 1.92]
DUAL_COND_STATIC_SUCCESS = [75, 90, 85, 100]
DUAL_COND_STATIC_DELTA = [3.72, 3.61, 2.85, 4.30]

# --- Single-attribute response (pitch) ---
SINGLE_ALPHAS = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
SINGLE_PITCH_P = [37.83, 38.60, 39.17, 44.11, 81.93, 82.08, 79.24, 78.36]
SINGLE_PITCH_PI = [31.30, 35.35, 39.31, 43.33, 82.73, 82.47, 80.67, 79.22]
SINGLE_PITCH_PID = [33.23, 38.37, 39.55, 43.34, 82.76, 82.90, 80.88, 78.86]
SINGLE_PITCH_DEG_P = [6.6, 7.1, 5.2, 3.9, 4.6, 5.6, 7.7, 9.2]
SINGLE_PITCH_DEG_PI = [18.3, 9.2, 5.5, 6.5, 3.7, 5.2, 8.3, 9.1]
SINGLE_PITCH_DEG_PID = [14.3, 7.2, 6.4, 3.6, 3.7, 5.0, 9.1, 9.0]

# --- Single-attribute response (duration) ---
SINGLE_DUR_P = [3.07, 2.96, 3.13, 3.34, 17.45, 19.33, 20.13, 20.26]
SINGLE_DUR_PI = [3.06, 3.07, 3.13, 3.44, 18.01, 19.81, 19.97, 20.52]
SINGLE_DUR_PID = [3.06, 3.05, 3.08, 3.25, 18.00, 19.81, 20.09, 20.50]
SINGLE_DUR_DEG_P = [19.1, 17.4, 16.2, 6.3, 2.3, 3.9, 9.1, 12.6]
SINGLE_DUR_DEG_PI = [25.5, 23.1, 20.6, 6.0, 2.0, 6.4, 10.8, 12.2]
SINGLE_DUR_DEG_PID = [24.5, 25.1, 22.9, 6.0, 2.1, 4.7, 11.3, 13.2]

# --- Gain grid search ---
GAIN_SEARCH = {
    "Pitch": {
        "Kp": 1.5,
        "Ki": 0.2,
        "Kd": 0.01,
        "score_neg": -9.04,
        "score_pos": 74.76,
        "deg_neg": 60.67,
        "deg_pos": 7.91,
        "eff_neg": -14.52,
        "eff_pos": 80.67,
    },
    "Duration": {
        "Kp": 1.25,
        "Ki": 0.025,
        "Kd": 0.01,
        "score_neg": -1.92,
        "score_pos": 19.55,
        "deg_neg": 30.47,
        "deg_pos": 5.37,
        "eff_neg": -2.50,
        "eff_pos": 20.60,
    },
}

# ══════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    "P-only": "#3498db",
    "PI": "#e67e22",
    "PID": "#2ecc71",
    "Baseline": "#95a5a6",
    "Static SAS": "#e74c3c",
}


def save(fig, path, output_dir):
    """Save as both PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{path}.png"
    pdf = output_dir / f"{path}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {png}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: Error Convergence Across Layers
# ══════════════════════════════════════════════════════════════════════
def plot_error_convergence(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for name, data in ERROR_CONV_PITCH.items():
        ax1.plot(
            LAYERS,
            data,
            "o-",
            color=COLORS.get(name, COLORS["PID"]),
            linewidth=2,
            markersize=5,
            label=name,
        )
    ax1.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("⟨e(0), e(k)⟩", fontsize=12)
    ax1.set_title("Pitch (Kp=1.5, Ki=0.2, Kd=0.01)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks(LAYERS)
    # Highlight layer 11
    ax1.annotate(
        "PI/PID push\nerror negative",
        xy=(11, -1.15),
        xytext=(8.5, -1.4),
        fontsize=9,
        color="#27ae60",
        arrowprops=dict(arrowstyle="->", color="#27ae60"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", alpha=0.8),
    )

    for name, data in ERROR_CONV_DUR.items():
        ax2.plot(
            LAYERS,
            data,
            "o-",
            color=COLORS.get(name, COLORS["PID"]),
            linewidth=2,
            markersize=5,
            label=name,
        )
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_title(
        "Duration (Kp=1.25, Ki=0.025, Kd=0.01)", fontsize=13, fontweight="bold"
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.2)
    ax2.set_xticks(LAYERS)

    fig.suptitle(
        "Spatial PID Error Convergence ⟨e(0), e(k)⟩ Across Layers",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "spatial_error_convergence", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: Hook Ablation
# ══════════════════════════════════════════════════════════════════════
def plot_hook_ablation(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(HOOK_CONFIGS))
    w = 0.35

    # Panel A: Degradation comparison
    bars1 = ax1.bar(
        x - w / 2, HOOK_P_DEG, w, color=COLORS["P-only"], label="P-only δ", alpha=0.8
    )
    bars2 = ax1.bar(
        x + w / 2, HOOK_PID_DEG, w, color=COLORS["PID"], label="PID δ", alpha=0.8
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{c}\n({l}L)" for c, l in zip(HOOK_CONFIGS, HOOK_LAYERS)], fontsize=9
    )
    ax1.set_ylabel("Quality Degradation δ (↓ better)", fontsize=11)
    ax1.set_title(
        "PID Reduces Degradation at All Injection Loci", fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")

    # Add reduction % labels
    for i, pct in enumerate(HOOK_DEG_REDUCTION):
        y_max = max(HOOK_P_DEG[i], HOOK_PID_DEG[i])
        ax1.text(
            i,
            y_max + 0.15,
            f"−{pct}%",
            ha="center",
            fontsize=9,
            color="#27ae60",
            fontweight="bold",
        )

    # Panel B: Effectiveness
    ax2.bar(x, HOOK_PID_DELTA, 0.6, color=COLORS["PID"], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{c}\n({l}L)" for c, l in zip(HOOK_CONFIGS, HOOK_LAYERS)], fontsize=9
    )
    ax2.set_ylabel("Avg Steering |Δ| (↑ better)", fontsize=11)
    ax2.set_title(
        "Steering Effectiveness by Hook Configuration", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Spatial PID — Hook Ablation Study", fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, "spatial_hook_ablation", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: Temporal Ablation λ
# ══════════════════════════════════════════════════════════════════════
def plot_temporal_ablation(output_dir):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = list(TEMPORAL_ABLATION.keys())
    values = list(TEMPORAL_ABLATION.values())
    colors = [COLORS["P-only"], COLORS["PI"], COLORS["PID"]]

    bars = ax.bar(
        names,
        values,
        color=colors,
        width=0.55,
        alpha=0.9,
        edgecolor="white",
        linewidth=1.2,
    )

    # Add value labels above bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Average $\\lambda(t)$", fontsize=11)
    ax.set_xlabel("Controller Configuration", fontsize=11)
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(0, 1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save(fig, "temporal_ablation_lambda", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: Spatial FMD
# ══════════════════════════════════════════════════════════════════════
def plot_spatial_fmd(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # Pitch
    for method in ["Baseline", "P-only", "PI", "PID"]:
        ax1.plot(
            SPATIAL_FMD_ALPHAS,
            SPATIAL_FMD_PITCH[method],
            "o-",
            color=COLORS.get(method, "#333"),
            linewidth=2,
            markersize=6,
            label=method,
        )
    ax1.set_xlabel("α", fontsize=12)
    ax1.set_ylabel("FMD (↓ better)", fontsize=12)
    ax1.set_title("Pitch — Spatial FMD", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Duration
    for method in ["Baseline", "P-only", "PI", "PID"]:
        ax2.plot(
            SPATIAL_FMD_ALPHAS,
            SPATIAL_FMD_DUR[method],
            "o-",
            color=COLORS.get(method, "#333"),
            linewidth=2,
            markersize=6,
            label=method,
        )
    ax2.set_xlabel("α", fontsize=12)
    ax2.set_ylabel("FMD (↓ better)", fontsize=12)
    ax2.set_title("Duration — Spatial FMD", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    # Highlight PID advantage for duration
    ax2.annotate(
        "PID best FMD\nat α=1.0–1.5",
        xy=(1.0, 456.0),
        xytext=(1.5, 420),
        fontsize=9,
        color=COLORS["PID"],
        arrowprops=dict(arrowstyle="->", color=COLORS["PID"]),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", alpha=0.8),
    )

    fig.suptitle(
        "Spatial PID — Fréchet Music Distance vs Steering Strength",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "spatial_fmd_comparison", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5: Temporal FMD
# ══════════════════════════════════════════════════════════════════════
def plot_temporal_fmd(output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    concepts = list(TEMPORAL_FMD.keys())
    methods = ["Baseline", "Static SAS", "PID"]
    x = np.arange(len(concepts))
    w = 0.25

    for i, method in enumerate(methods):
        vals = [TEMPORAL_FMD[c][method] for c in concepts]
        bars = ax.bar(
            x + (i - 1) * w,
            vals,
            w,
            color=COLORS.get(method, "#333"),
            label=method,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{v:.1f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(concepts, fontsize=12)
    ax.set_ylabel("FMD (↓ better)", fontsize=12)
    ax.set_title(
        "Temporal PID — Fréchet Music Distance (40 samples/method)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")

    # Improvement annotations
    ax.annotate(
        "−5.3%", xy=(0.25, 465), fontsize=11, color=COLORS["PID"], fontweight="bold"
    )
    ax.annotate(
        "−4.7%", xy=(1.25, 505), fontsize=11, color=COLORS["PID"], fontweight="bold"
    )

    fig.tight_layout()
    save(fig, "temporal_fmd_comparison", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6: Dual Steering (Unconditioned)
# ══════════════════════════════════════════════════════════════════════
def plot_dual_steering(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    methods = ["Baseline", "Static SAS", "PID"]

    # Panel A: Attribute values (pitch + duration)
    x = np.arange(len(methods))
    w = 0.3
    pitch_vals = [DUAL_UNCOND[m]["pitch"] for m in methods]
    dur_vals = [DUAL_UNCOND[m]["dur"] for m in methods]

    bars1 = ax1.bar(
        x - w / 2, pitch_vals, w, color="#9b59b6", label="Avg Pitch", alpha=0.8
    )
    bars2 = ax1.bar(
        x + w / 2, dur_vals, w, color="#f39c12", label="Avg Duration", alpha=0.8
    )
    for bar, v in zip(bars1, pitch_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.1f}",
            ha="center",
            fontsize=9,
        )
    for bar, v in zip(bars2, dur_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v:.1f}",
            ha="center",
            fontsize=9,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylabel("Attribute Value", fontsize=12)
    ax1.set_title("Steering Effectiveness", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")

    # Panel B: Degradation + Success
    delta_vals = [DUAL_UNCOND[m]["delta"] for m in methods]
    colors = [COLORS.get(m, "#333") for m in methods]
    bars = ax2.bar(
        x, delta_vals, 0.5, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5
    )
    for bar, v, m in zip(bars, delta_vals, methods):
        success = DUAL_UNCOND[m]["success"]
        label = f"δ={v:.2f}"
        if success > 0:
            label += f"\n{success}% dual"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            label,
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylabel("Quality Degradation δ (↓ better)", fontsize=12)
    ax2.set_title("Quality Preservation", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="y")

    # Highlight
    ax2.annotate(
        "4.7× less\ndegradation",
        xy=(2, 0.47),
        xytext=(1.2, 1.5),
        fontsize=11,
        color=COLORS["PID"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["PID"], lw=2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", alpha=0.8),
    )

    fig.suptitle(
        "Dual Temporal PID — Unconditioned (tm=2.0, n=20)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "dual_steering_comparison", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7: Dual Conditioned
# ══════════════════════════════════════════════════════════════════════
def plot_dual_conditioned(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(DUAL_COND_SCENARIOS))
    w = 0.35

    # Panel A: Success rate
    ax1.bar(
        x - w / 2,
        DUAL_COND_PID_SUCCESS,
        w,
        color=COLORS["PID"],
        label="PID",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.bar(
        x + w / 2,
        DUAL_COND_STATIC_SUCCESS,
        w,
        color=COLORS["Static SAS"],
        label="Static SAS",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(DUAL_COND_SCENARIOS, fontsize=9)
    ax1.set_ylabel("Dual Success Rate (%)", fontsize=11)
    ax1.set_title("Steering Success", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.set_ylim(60, 105)
    # Highlight 100%
    ax1.text(
        3 - w / 2,
        101,
        "100%",
        ha="center",
        fontsize=9,
        color=COLORS["PID"],
        fontweight="bold",
    )

    # Panel B: Degradation
    bars1 = ax2.bar(
        x - w / 2,
        DUAL_COND_PID_DELTA,
        w,
        color=COLORS["PID"],
        label="PID δ",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = ax2.bar(
        x + w / 2,
        DUAL_COND_STATIC_DELTA,
        w,
        color=COLORS["Static SAS"],
        label="Static SAS δ",
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(DUAL_COND_SCENARIOS, fontsize=9)
    ax2.set_ylabel("Quality Degradation δ (↓ better)", fontsize=11)
    ax2.set_title("Quality Preservation", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")

    # Highlight best scenario
    ax2.annotate(
        "2.2× less δ",
        xy=(3 - w / 2, 1.92),
        xytext=(2.2, 0.5),
        fontsize=10,
        color=COLORS["PID"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["PID"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5f5e3", alpha=0.8),
    )

    fig.suptitle(
        "Dual Conditioned PID — 4 Scenarios (10 songs × 2 reps)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "dual_conditioned_comparison", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8: Single-Attribute Response Curves
# ══════════════════════════════════════════════════════════════════════
def plot_single_attribute(output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 1: Attribute values
    ax = axes[0, 0]
    ax.plot(
        SINGLE_ALPHAS,
        SINGLE_PITCH_P,
        "o-",
        color=COLORS["P-only"],
        lw=2,
        label="P-only",
    )
    ax.plot(SINGLE_ALPHAS, SINGLE_PITCH_PI, "s-", color=COLORS["PI"], lw=2, label="PI")
    ax.plot(
        SINGLE_ALPHAS, SINGLE_PITCH_PID, "^-", color=COLORS["PID"], lw=2, label="PID"
    )
    ax.axhline(
        y=69.57, color=COLORS["Baseline"], ls="--", lw=1.5, label="Baseline (69.57)"
    )
    ax.set_ylabel("Avg Pitch (semitones)", fontsize=11)
    ax.set_title("Pitch — Steering Response", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("α", fontsize=11)

    ax = axes[0, 1]
    ax.plot(
        SINGLE_ALPHAS, SINGLE_DUR_P, "o-", color=COLORS["P-only"], lw=2, label="P-only"
    )
    ax.plot(SINGLE_ALPHAS, SINGLE_DUR_PI, "s-", color=COLORS["PI"], lw=2, label="PI")
    ax.plot(SINGLE_ALPHAS, SINGLE_DUR_PID, "^-", color=COLORS["PID"], lw=2, label="PID")
    ax.axhline(
        y=6.21, color=COLORS["Baseline"], ls="--", lw=1.5, label="Baseline (6.21)"
    )
    ax.set_ylabel("Avg Duration (ticks)", fontsize=11)
    ax.set_title("Duration — Steering Response", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("α", fontsize=11)

    # Row 2: Degradation
    ax = axes[1, 0]
    ax.plot(
        SINGLE_ALPHAS,
        SINGLE_PITCH_DEG_P,
        "o-",
        color=COLORS["P-only"],
        lw=2,
        label="P-only",
    )
    ax.plot(
        SINGLE_ALPHAS, SINGLE_PITCH_DEG_PI, "s-", color=COLORS["PI"], lw=2, label="PI"
    )
    ax.plot(
        SINGLE_ALPHAS,
        SINGLE_PITCH_DEG_PID,
        "^-",
        color=COLORS["PID"],
        lw=2,
        label="PID",
    )
    ax.set_ylabel("Degradation δ (↓ better)", fontsize=11)
    ax.set_title("Pitch — Quality Degradation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("α", fontsize=11)

    ax = axes[1, 1]
    ax.plot(
        SINGLE_ALPHAS,
        SINGLE_DUR_DEG_P,
        "o-",
        color=COLORS["P-only"],
        lw=2,
        label="P-only",
    )
    ax.plot(
        SINGLE_ALPHAS, SINGLE_DUR_DEG_PI, "s-", color=COLORS["PI"], lw=2, label="PI"
    )
    ax.plot(
        SINGLE_ALPHAS, SINGLE_DUR_DEG_PID, "^-", color=COLORS["PID"], lw=2, label="PID"
    )
    ax.set_ylabel("Degradation δ (↓ better)", fontsize=11)
    ax.set_title("Duration — Quality Degradation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("α", fontsize=11)

    fig.suptitle(
        "Spatial PID — Single-Attribute Steering Response (n=25)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, "single_attribute_response", output_dir)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 9: Gain Grid Search
# ══════════════════════════════════════════════════════════════════════
def plot_gain_search(output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    concepts = ["Pitch", "Duration"]
    ki_vals = [GAIN_SEARCH[c]["Ki"] for c in concepts]
    kp_vals = [GAIN_SEARCH[c]["Kp"] for c in concepts]

    # Panel A: Ki comparison (the key insight)
    colors = ["#9b59b6", "#f39c12"]
    bars = ax1.bar(
        concepts,
        ki_vals,
        color=colors,
        width=0.5,
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, v in zip(bars, ki_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"Ki={v}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
    ax1.set_ylabel("Optimal Integral Gain (Ki)", fontsize=12)
    ax1.set_title("Concept-Dependent Ki: 8× Difference", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.annotate(
        "Pitch requires 8× more\nintegral action to override\nautoregressive priors",
        xy=(0, 0.2),
        xytext=(0.5, 0.15),
        fontsize=9,
        color="#9b59b6",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8daef", alpha=0.8),
    )

    # Panel B: Score at α=+1.0 (effectiveness)
    scores = [GAIN_SEARCH[c]["score_pos"] for c in concepts]
    degs = [GAIN_SEARCH[c]["deg_pos"] for c in concepts]
    effs = [GAIN_SEARCH[c]["eff_pos"] for c in concepts]

    x = np.arange(len(concepts))
    w = 0.25
    ax2.bar(x - w, scores, w, color=COLORS["PID"], label="Score", alpha=0.85)
    ax2.bar(x, effs, w, color="#3498db", label="Effectiveness", alpha=0.85)
    ax2.bar(
        x + w, degs, w, color=COLORS["Static SAS"], label="Degradation δ", alpha=0.85
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(concepts, fontsize=11)
    ax2.set_ylabel("Value (at α=+1.0)", fontsize=12)
    ax2.set_title(
        "Grid Search Best Configuration — α=+1.0", fontsize=13, fontweight="bold"
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "PID Gain Grid Search — Optimal Per-Concept Configuration",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "gain_grid_search", output_dir)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper/presentation figures"
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=pathlib.Path("pid_figs")
    )
    args = parser.parse_args()

    print(f"Generating figures to {args.output_dir}/\n")

    plot_error_convergence(args.output_dir)
    plot_hook_ablation(args.output_dir)
    plot_temporal_ablation(args.output_dir)
    plot_spatial_fmd(args.output_dir)
    plot_temporal_fmd(args.output_dir)
    plot_dual_steering(args.output_dir)
    plot_dual_conditioned(args.output_dir)
    plot_single_attribute(args.output_dir)
    plot_gain_search(args.output_dir)

    print(f"\nDone! {9} figures × 2 formats = 18 files in {args.output_dir}/")


if __name__ == "__main__":
    main()
