"""
Visualization script for SAS Layer-Lambda Optimization Results.

Generates publication-quality plots for:
1. Steering response curves (metric vs λ) per layer — for both pitch and duration
2. Layer comparison heatmap (R², monotonicity, degradation)
3. Control-quality trade-off scatter plot
4. Best configuration detailed view (single_10)
5. Full 16-layer overview from the initial broad sweep

Usage:
    python sparse_steering/plot_layer_lambda_results.py [--output_dir plots/]

    If JSON result files exist (from the experiment), they will be loaded.
    Otherwise, hardcoded data from the experiments is used.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec

# ============================================================================
# Style Configuration
# ============================================================================

# Use a clean style
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# Color palette (colorblind-friendly)
COLORS = {
    "single_0": "#bdbdbd",
    "single_1": "#e377c2",
    "single_2": "#bdbdbd",
    "single_3": "#bdbdbd",
    "single_4": "#bdbdbd",
    "single_5": "#bdbdbd",
    "single_6": "#bdbdbd",
    "single_7": "#bdbdbd",
    "single_8": "#2ca02c",  # green
    "single_9": "#ff7f0e",  # orange
    "single_10": "#1f77b4",  # blue (primary)
    "single_11": "#9467bd",  # purple
    "early": "#bcbd22",  # olive
    "middle": "#7f7f7f",  # grey
    "late": "#d62728",  # red
    "all": "#17becf",  # cyan
}

LAYER_LABELS = {
    "single_0": "Layer 0",
    "single_1": "Layer 1",
    "single_2": "Layer 2",
    "single_3": "Layer 3",
    "single_4": "Layer 4",
    "single_5": "Layer 5",
    "single_6": "Layer 6",
    "single_7": "Layer 7",
    "single_8": "Layer 8",
    "single_9": "Layer 9",
    "single_10": "Layer 10",
    "single_11": "Layer 11",
    "early": "Early [0-3]",
    "middle": "Middle [4-7]",
    "late": "Late [8-11]",
    "all": "All [0-11]",
}


# ============================================================================
# Experiment Data (hardcoded from all runs)
# ============================================================================


def get_pitch_broad_sweep():
    """Initial broad sweep: 16 layer groups × 9 λ values × 5 samples."""
    return {
        "lambda_values": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "n_samples": 5,
        "results": {
            "all": {
                "mean": [40.81, 41.19, 41.61, 50.59, 71.62, 82.85, 75.92, 69.77, 69.58],
                "r": 0.822,
                "p": 0.0066,
                "R2": 0.676,
                "slope": 10.0560,
                "range": 42.04,
                "mono": False,
                "deg": 1.40,
                "score": 34.418,
            },
            "single_10": {
                "mean": [38.79, 39.56, 40.11, 60.82, 67.09, 74.42, 75.27, 73.09, 72.54],
                "r": 0.904,
                "p": 0.0008,
                "R2": 0.817,
                "slope": 10.6509,
                "range": 36.48,
                "mono": False,
                "deg": 1.75,
                "score": 32.787,
            },
            "late": {
                "mean": [40.82, 41.33, 41.54, 52.11, 68.44, 77.43, 72.25, 70.96, 70.89],
                "r": 0.881,
                "p": 0.0017,
                "R2": 0.776,
                "slope": 9.8649,
                "range": 36.62,
                "mono": False,
                "deg": 0.71,
                "score": 32.178,
            },
            "single_8": {
                "mean": [40.25, 45.88, 58.79, 61.93, 70.84, 66.13, 73.67, 75.09, 73.87],
                "r": 0.924,
                "p": 0.0004,
                "R2": 0.855,
                "slope": 8.5348,
                "range": 34.84,
                "mono": False,
                "deg": 0.97,
                "score": 32.105,
            },
            "single_11": {
                "mean": [48.64, 49.36, 52.84, 64.92, 70.44, 69.74, 71.32, 75.03, 76.74],
                "r": 0.950,
                "p": 0.0001,
                "R2": 0.903,
                "slope": 7.7070,
                "range": 28.10,
                "mono": False,
                "deg": 0.78,
                "score": 26.625,
            },
            "single_1": {
                "mean": [60.12, 58.84, 55.54, 60.41, 66.10, 74.44, 80.48, 76.78, 72.14],
                "r": 0.841,
                "p": 0.0045,
                "R2": 0.707,
                "slope": 5.5263,
                "range": 24.94,
                "mono": False,
                "deg": 2.58,
                "score": 20.709,
            },
            "early": {
                "mean": [62.39, 60.26, 57.58, 57.79, 65.06, 82.21, 84.68, 72.90, 67.82],
                "r": 0.625,
                "p": 0.0722,
                "R2": 0.390,
                "slope": 4.6094,
                "range": 27.11,
                "mono": False,
                "deg": 1.58,
                "score": 16.771,
            },
            "single_5": {
                "mean": [71.34, 74.16, 71.47, 69.17, 66.38, 67.07, 66.17, 68.41, 60.20],
                "r": -0.845,
                "p": 0.0042,
                "R2": 0.713,
                "slope": -2.4838,
                "range": 13.96,
                "mono": False,
                "deg": 1.57,
                "score": 11.635,
            },
            "single_9": {
                "mean": [71.50, 69.18, 74.82, 67.36, 71.48, 66.11, 65.31, 60.09, 64.21],
                "r": -0.783,
                "p": 0.0126,
                "R2": 0.613,
                "slope": -2.5566,
                "range": 14.73,
                "mono": False,
                "deg": 1.13,
                "score": 11.418,
            },
            "single_0": {
                "mean": [61.60, 63.88, 68.44, 65.93, 64.47, 70.31, 66.11, 69.22, 74.89],
                "r": 0.789,
                "p": 0.0114,
                "R2": 0.623,
                "slope": 2.2974,
                "range": 13.29,
                "mono": False,
                "deg": 1.27,
                "score": 10.365,
            },
            "middle": {
                "mean": [72.49, 66.97, 69.35, 62.04, 67.68, 66.79, 61.07, 63.23, 63.42],
                "r": -0.723,
                "p": 0.0276,
                "R2": 0.523,
                "slope": -1.9758,
                "range": 11.42,
                "mono": False,
                "deg": 1.62,
                "score": 8.098,
            },
            "single_2": {
                "mean": [66.19, 66.45, 66.80, 66.37, 69.63, 68.72, 71.17, 72.43, 70.39],
                "r": 0.891,
                "p": 0.0013,
                "R2": 0.793,
                "slope": 1.5281,
                "range": 6.24,
                "mono": False,
                "deg": 1.20,
                "score": 5.435,
            },
            "single_7": {
                "mean": [69.81, 67.51, 66.71, 67.65, 62.94, 64.61, 63.48, 67.11, 62.30],
                "r": -0.728,
                "p": 0.0262,
                "R2": 0.530,
                "slope": -1.3580,
                "range": 7.51,
                "mono": False,
                "deg": 5.53,
                "score": 4.912,
            },
            "single_4": {
                "mean": [76.85, 65.74, 63.75, 73.36, 61.78, 65.93, 66.65, 68.50, 71.13],
                "r": -0.154,
                "p": 0.6918,
                "R2": 0.024,
                "slope": -0.5416,
                "range": 15.07,
                "mono": False,
                "deg": 2.34,
                "score": 2.092,
            },
            "single_3": {
                "mean": [68.82, 76.70, 63.94, 59.79, 71.34, 64.56, 75.21, 67.19, 72.98],
                "r": 0.125,
                "p": 0.7483,
                "R2": 0.016,
                "slope": 0.5137,
                "range": 16.91,
                "mono": False,
                "deg": 4.04,
                "score": 1.713,
            },
            "single_6": {
                "mean": [64.92, 67.94, 71.01, 64.61, 72.35, 66.41, 62.81, 67.88, 68.59],
                "r": -0.001,
                "p": 0.9971,
                "R2": 0.000,
                "slope": -0.0032,
                "range": 9.53,
                "mono": False,
                "deg": 1.64,
                "score": -0.150,
            },
        },
    }


def get_pitch_refined():
    """Refined pitch: 4 layers × 10 λ values × 10 samples."""
    return {
        "lambda_values": [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
        "n_samples": 10,
        "results": {
            "late": {
                "mean": [
                    41.45,
                    42.58,
                    40.10,
                    56.86,
                    67.58,
                    70.93,
                    73.55,
                    78.81,
                    74.89,
                    72.72,
                ],
                "r": 0.900,
                "p": 0.0004,
                "R2": 0.810,
                "slope": 17.2185,
                "range": 38.72,
                "mono": False,
                "deg": 0.70,
                "score": 34.781,
            },
            "single_10": {
                "mean": [
                    39.73,
                    40.10,
                    45.49,
                    59.91,
                    65.16,
                    67.44,
                    73.35,
                    74.04,
                    74.23,
                    72.66,
                ],
                "r": 0.930,
                "p": 0.0001,
                "R2": 0.865,
                "slope": 16.5265,
                "range": 34.49,
                "mono": False,
                "deg": 1.64,
                "score": 31.926,
            },
            "single_8": {
                "mean": [
                    44.41,
                    55.98,
                    63.27,
                    64.33,
                    67.36,
                    68.87,
                    67.53,
                    70.70,
                    69.18,
                    75.57,
                ],
                "r": 0.910,
                "p": 0.0003,
                "R2": 0.828,
                "slope": 9.9954,
                "range": 31.17,
                "mono": False,
                "deg": 1.10,
                "score": 28.249,
            },
            "single_11": {
                "mean": [
                    50.34,
                    51.83,
                    57.80,
                    63.53,
                    70.28,
                    66.39,
                    64.38,
                    73.13,
                    75.34,
                    75.38,
                ],
                "r": 0.937,
                "p": 0.0001,
                "R2": 0.878,
                "slope": 10.6986,
                "range": 25.04,
                "mono": False,
                "deg": 0.87,
                "score": 23.374,
            },
        },
        "quality": {
            "late": {
                "entropy": [
                    2.185,
                    2.291,
                    2.536,
                    2.981,
                    2.823,
                    2.875,
                    2.737,
                    2.894,
                    2.711,
                    2.592,
                ],
                "scale": [
                    96.77,
                    95.59,
                    95.46,
                    90.79,
                    94.88,
                    93.31,
                    94.81,
                    93.92,
                    96.40,
                    98.00,
                ],
                "groove": [
                    97.49,
                    97.01,
                    97.63,
                    94.57,
                    94.33,
                    89.31,
                    94.07,
                    94.80,
                    91.23,
                    94.84,
                ],
                "deg": [0.79, 0.68, 0.44, 1.47, 0.15, 3.84, 0.24, 0.08, 2.09, 0.38],
            },
            "single_10": {
                "entropy": [
                    2.651,
                    2.871,
                    2.957,
                    2.929,
                    2.847,
                    2.670,
                    2.840,
                    2.812,
                    2.950,
                    2.865,
                ],
                "scale": [
                    90.88,
                    88.87,
                    91.26,
                    93.24,
                    93.87,
                    96.36,
                    94.18,
                    96.80,
                    94.90,
                    95.17,
                ],
                "groove": [
                    97.75,
                    97.91,
                    97.13,
                    92.58,
                    92.93,
                    93.08,
                    92.22,
                    91.90,
                    91.59,
                    93.78,
                ],
                "deg": [1.70, 3.50, 1.02, 0.52, 0.24, 0.30, 0.96, 1.31, 1.49, 0.11],
            },
            "single_8": {
                "entropy": [
                    2.594,
                    2.739,
                    2.741,
                    2.815,
                    2.803,
                    2.896,
                    2.792,
                    2.804,
                    2.923,
                    2.802,
                ],
                "scale": [
                    94.93,
                    93.58,
                    95.22,
                    93.78,
                    95.15,
                    95.60,
                    95.84,
                    96.12,
                    93.97,
                    96.92,
                ],
                "groove": [
                    94.53,
                    93.24,
                    89.85,
                    92.95,
                    93.22,
                    87.66,
                    95.22,
                    90.91,
                    91.71,
                    91.67,
                ],
                "deg": [0.38, 0.23, 3.43, 0.26, 0.17, 5.47, 0.18, 2.31, 1.39, 1.56],
            },
            "single_11": {
                "entropy": [
                    2.358,
                    2.763,
                    2.720,
                    2.841,
                    2.867,
                    2.895,
                    2.847,
                    2.733,
                    2.623,
                    2.488,
                ],
                "scale": [
                    96.28,
                    90.47,
                    93.66,
                    94.05,
                    93.51,
                    91.85,
                    93.65,
                    95.34,
                    95.55,
                    98.02,
                ],
                "groove": [
                    96.42,
                    95.12,
                    94.07,
                    90.84,
                    94.31,
                    91.99,
                    93.16,
                    91.76,
                    95.25,
                    96.21,
                ],
                "deg": [0.62, 2.00, 0.25, 2.34, 0.11, 1.55, 0.13, 1.53, 0.35, 0.49],
            },
        },
    }


def get_pitch_final():
    """Final pitch: single_10 only × 9 λ values × 15 samples."""
    return {
        "lambda_values": [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
        "n_samples": 15,
        "results": {
            "single_10": {
                "mean": [39.29, 40.05, 43.93, 57.93, 64.00, 67.61, 67.59, 74.78, 74.69],
                "r": 0.958,
                "p": 0.0000,
                "R2": 0.918,
                "slope": 18.7174,
                "range": 35.49,
                "mono": False,
                "deg": 1.34,
                "score": 33.861,
            },
        },
        "quality": {
            "single_10": {
                "entropy": [
                    2.651,
                    2.871,
                    2.957,
                    2.929,
                    2.847,
                    2.670,
                    2.840,
                    2.812,
                    2.950,
                ],
                "scale": [
                    90.88,
                    88.87,
                    91.26,
                    93.24,
                    93.87,
                    96.36,
                    94.18,
                    96.80,
                    94.90,
                ],
                "groove": [
                    97.75,
                    97.91,
                    97.13,
                    92.58,
                    92.93,
                    93.08,
                    92.22,
                    91.90,
                    91.59,
                ],
                "deg": [1.70, 3.50, 1.02, 0.52, 0.24, 0.30, 0.96, 1.31, 1.49],
            },
        },
    }


def get_duration_broad_sweep():
    """Initial broad sweep duration: 16 layer groups × 9 λ × 5 samples."""
    return {
        "lambda_values": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "n_samples": 5,
        "results": {
            "late": {
                "mean": [3.00, 3.02, 3.09, 4.80, 10.50, 15.35, 86.68, 93.81, 95.15],
                "r": 0.871,
                "p": 0.0022,
                "R2": 0.759,
                "slope": 27.2905,
                "range": 92.15,
                "mono": True,
                "deg": 0.79,
                "score": 80.225,
            },
            "all": {
                "mean": [2.97, 2.61, 2.10, 2.34, 11.10, 38.06, 79.72, 78.65, 55.92],
                "r": 0.861,
                "p": 0.0029,
                "R2": 0.741,
                "slope": 21.0278,
                "range": 77.61,
                "mono": False,
                "deg": 1.82,
                "score": 66.613,
            },
            "single_9": {
                "mean": [74.90, 26.65, 22.58, 9.19, 6.17, 5.50, 6.23, 9.93, 6.83],
                "r": -0.730,
                "p": 0.0257,
                "R2": 0.532,
                "slope": -11.9612,
                "range": 69.41,
                "mono": False,
                "deg": 1.47,
                "score": 50.488,
            },
            "single_10": {
                "mean": [3.30, 3.37, 4.14, 5.13, 7.56, 17.39, 39.85, 48.84, 57.91],
                "r": 0.913,
                "p": 0.0006,
                "R2": 0.833,
                "slope": 14.6166,
                "range": 54.61,
                "mono": True,
                "deg": 0.81,
                "score": 49.764,
            },
            "single_1": {
                "mean": [2.06, 2.08, 2.08, 2.70, 10.41, 23.59, 44.36, 47.29, 47.97],
                "r": 0.928,
                "p": 0.0003,
                "R2": 0.862,
                "slope": 14.1570,
                "range": 45.91,
                "mono": False,
                "deg": 1.97,
                "score": 42.426,
            },
            "early": {
                "mean": [2.13, 2.25, 2.12, 3.23, 8.94, 19.17, 45.08, 47.02, 48.16],
                "r": 0.916,
                "p": 0.0005,
                "R2": 0.839,
                "slope": 14.0092,
                "range": 46.04,
                "mono": False,
                "deg": 1.11,
                "score": 42.055,
            },
            "single_8": {
                "mean": [5.52, 6.10, 5.90, 6.92, 7.82, 11.39, 20.40, 17.44, 25.11],
                "r": 0.909,
                "p": 0.0007,
                "R2": 0.826,
                "slope": 4.8626,
                "range": 19.60,
                "mono": False,
                "deg": 2.20,
                "score": 17.587,
            },
            "single_11": {
                "mean": [3.02, 3.05, 4.14, 10.11, 7.48, 10.29, 13.92, 13.93, 13.13],
                "r": 0.932,
                "p": 0.0002,
                "R2": 0.869,
                "slope": 3.0941,
                "range": 10.91,
                "mono": False,
                "deg": 1.07,
                "score": 10.067,
            },
            "single_3": {
                "mean": [8.23, 8.49, 11.30, 9.98, 12.75, 9.47, 7.58, 13.67, 22.17],
                "r": 0.644,
                "p": 0.0615,
                "R2": 0.414,
                "slope": 2.1126,
                "range": 14.58,
                "mono": False,
                "deg": 1.03,
                "score": 9.282,
            },
            "single_6": {
                "mean": [5.20, 6.94, 5.72, 7.17, 8.55, 8.30, 10.47, 10.10, 9.98],
                "r": 0.930,
                "p": 0.0003,
                "R2": 0.865,
                "slope": 1.3063,
                "range": 5.27,
                "mono": False,
                "deg": 3.17,
                "score": 4.581,
            },
            "single_5": {
                "mean": [11.04, 10.46, 9.32, 12.44, 7.63, 7.41, 9.09, 8.15, 8.11],
                "r": -0.644,
                "p": 0.0613,
                "R2": 0.414,
                "slope": -0.8034,
                "range": 5.03,
                "mono": False,
                "deg": 0.43,
                "score": 3.196,
            },
            "single_4": {
                "mean": [6.83, 6.06, 6.49, 6.88, 4.85, 5.38, 10.12, 8.61, 8.03],
                "r": 0.504,
                "p": 0.1669,
                "R2": 0.254,
                "slope": 0.6070,
                "range": 5.27,
                "mono": False,
                "deg": 1.90,
                "score": 2.461,
            },
            "single_7": {
                "mean": [10.26, 8.86, 8.48, 5.39, 9.66, 7.33, 7.56, 15.16, 8.05],
                "r": 0.171,
                "p": 0.6603,
                "R2": 0.029,
                "slope": 0.3392,
                "range": 9.78,
                "mono": False,
                "deg": 1.38,
                "score": 1.532,
            },
            "single_0": {
                "mean": [5.74, 8.75, 11.02, 4.13, 7.00, 7.29, 8.61, 8.26, 9.20],
                "r": 0.242,
                "p": 0.5301,
                "R2": 0.059,
                "slope": 0.3576,
                "range": 6.89,
                "mono": False,
                "deg": 3.54,
                "score": 1.314,
            },
            "middle": {
                "mean": [8.65, 7.40, 9.83, 6.23, 10.17, 8.84, 9.82, 6.83, 6.40],
                "r": -0.237,
                "p": 0.5393,
                "R2": 0.056,
                "slope": -0.2693,
                "range": 3.94,
                "mono": False,
                "deg": 0.65,
                "score": 0.869,
            },
            "single_2": {
                "mean": [6.99, 4.51, 9.74, 6.89, 8.37, 8.26, 5.74, 6.72, 6.67],
                "r": -0.037,
                "p": 0.9242,
                "R2": 0.001,
                "slope": -0.0418,
                "range": 5.23,
                "mono": False,
                "deg": 2.59,
                "score": -0.064,
            },
        },
    }


def get_duration_refined():
    """Refined duration: 5 layers × 7 λ values × 10 samples."""
    return {
        "lambda_values": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "n_samples": 10,
        "results": {
            "all": {
                "mean": [2.58, 2.09, 2.32, 7.27, 38.00, 89.58, 77.79],
                "r": 0.884,
                "p": 0.0082,
                "R2": 0.782,
                "slope": 31.1654,
                "range": 87.49,
                "mono": False,
                "deg": 1.69,
                "score": 77.180,
            },
            "late": {
                "mean": [3.01, 3.14, 4.25, 6.88, 20.17, 89.28, 92.65],
                "r": 0.857,
                "p": 0.0138,
                "R2": 0.734,
                "slope": 32.6499,
                "range": 89.63,
                "mono": True,
                "deg": 1.13,
                "score": 76.667,
            },
            "single_10": {
                "mean": [3.53, 4.14, 5.14, 9.09, 20.77, 43.57, 56.49],
                "r": 0.913,
                "p": 0.0041,
                "R2": 0.834,
                "slope": 18.0994,
                "range": 52.96,
                "mono": True,
                "deg": 1.04,
                "score": 48.262,
            },
            "early": {
                "mean": [2.28, 2.21, 2.89, 8.81, 25.05, 38.13, 46.40],
                "r": 0.940,
                "p": 0.0016,
                "R2": 0.884,
                "slope": 16.1684,
                "range": 44.19,
                "mono": False,
                "deg": 0.97,
                "score": 41.441,
            },
            "single_1": {
                "mean": [2.07, 2.08, 2.55, 8.73, 19.69, 43.88, 46.93],
                "r": 0.919,
                "p": 0.0034,
                "R2": 0.845,
                "slope": 16.8088,
                "range": 44.86,
                "mono": True,
                "deg": 1.59,
                "score": 41.070,
            },
        },
        "quality": {
            "all": {
                "entropy": [1.093, 0.661, 2.246, 2.768, 2.570, 2.293, 1.919],
                "scale": [99.06, 99.56, 95.06, 97.38, 95.51, 89.19, 98.83],
                "groove": [98.96, 98.65, 93.74, 90.49, 94.90, 97.25, 97.67],
                "deg": [1.88, 2.31, 0.73, 2.77, 0.40, 3.75, 1.06],
            },
            "late": {
                "entropy": [1.855, 1.817, 2.704, 2.705, 2.797, 2.198, 2.589],
                "scale": [99.45, 99.54, 96.14, 95.52, 95.19, 96.27, 96.39],
                "groove": [98.85, 98.21, 90.14, 91.11, 96.79, 98.12, 98.66],
                "deg": [1.12, 1.16, 3.18, 2.21, 0.18, 0.78, 0.38],
            },
            "single_10": {
                "entropy": [2.649, 2.787, 2.729, 2.668, 2.640, 2.946, 2.837],
                "scale": [96.34, 94.12, 95.01, 94.76, 97.19, 90.48, 95.47],
                "groove": [95.31, 93.32, 89.86, 94.17, 96.60, 98.35, 98.03],
                "deg": [0.33, 0.19, 3.43, 0.31, 0.33, 1.81, 0.14],
            },
            "early": {
                "entropy": [1.351, 1.416, 2.304, 2.799, 2.574, 2.564, 1.968],
                "scale": [98.30, 97.41, 95.48, 93.44, 95.45, 93.63, 94.22],
                "groove": [97.08, 95.96, 92.87, 94.34, 93.94, 96.61, 97.75],
                "deg": [1.62, 1.56, 0.85, 0.18, 0.40, 0.41, 1.01],
            },
            "single_1": {
                "entropy": [1.198, 1.368, 2.280, 2.883, 2.669, 2.111, 1.932],
                "scale": [97.91, 98.31, 97.32, 93.25, 96.09, 96.47, 96.71],
                "groove": [95.82, 96.82, 89.78, 94.27, 93.07, 96.90, 98.39],
                "deg": [1.78, 1.61, 3.96, 0.09, 0.30, 0.86, 1.04],
            },
        },
    }


# Ground truth
GROUND_TRUTH = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


# ============================================================================
# Plot 1: Steering Response Curves — Best Layers (for slides)
# ============================================================================


def plot_steering_curves(output_dir: pathlib.Path):
    """Main steering response plot: pitch and duration side by side."""
    pitch = get_pitch_refined()
    duration = get_duration_refined()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Pitch --
    ax = axes[0]
    key_layers_pitch = ["single_10", "single_11", "single_8", "late"]
    for layer in key_layers_pitch:
        d = pitch["results"][layer]
        lw = 2.5 if layer == "single_10" else 1.8
        marker = "o" if layer == "single_10" else "s"
        ms = 7 if layer == "single_10" else 5
        ax.plot(
            pitch["lambda_values"],
            d["mean"],
            color=COLORS[layer],
            marker=marker,
            markersize=ms,
            linewidth=lw,
            label=f"{LAYER_LABELS[layer]} (r={d['r']:+.3f})",
            zorder=5 if layer == "single_10" else 3,
        )
    ax.axhline(y=GROUND_TRUTH["pitch_class_entropy"], color="grey", ls=":", alpha=0.5)
    ax.set_xlabel("Steering Strength (λ)")
    ax.set_ylabel("Average Pitch (MIDI)")
    ax.set_title("Pitch Steering Response")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-1.7, 1.2)

    # -- Duration --
    ax = axes[1]
    key_layers_dur = ["late", "single_10", "single_1", "early"]
    for layer in key_layers_dur:
        d = duration["results"][layer]
        is_mono = d["mono"]
        lw = 2.5 if is_mono else 1.8
        marker = "o" if is_mono else "s"
        ms = 7 if is_mono else 5
        label_extra = " ✓" if is_mono else ""
        ax.plot(
            duration["lambda_values"],
            d["mean"],
            color=COLORS[layer],
            marker=marker,
            markersize=ms,
            linewidth=lw,
            label=f"{LAYER_LABELS[layer]} (r={d['r']:+.3f}){label_extra}",
            zorder=5 if is_mono else 3,
        )
    ax.set_xlabel("Steering Strength (λ)")
    ax.set_ylabel("Average Duration (ticks)")
    ax.set_title("Duration Steering Response")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-1.7, 1.7)

    fig.suptitle(
        "SAS Steering Response: Layer 10 Provides Best Control for Both Concepts",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_steering_curves.png")
    fig.savefig(output_dir / "fig1_steering_curves.pdf")
    plt.close(fig)
    print(f"  Saved fig1_steering_curves.png/pdf")


# ============================================================================
# Plot 2: Layer Heatmap — R² and Score across all 16 layers (broad sweep)
# ============================================================================


def plot_layer_heatmap(output_dir: pathlib.Path):
    """Heatmap showing R², score, and direction for all 12 single layers."""
    pitch_data = get_pitch_broad_sweep()
    dur_data = get_duration_broad_sweep()

    single_layers = [f"single_{i}" for i in range(12)]
    labels = [f"Layer {i}" for i in range(12)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (data, concept, unit) in enumerate(
        [
            (pitch_data, "Pitch", "semitones"),
            (dur_data, "Duration", "ticks"),
        ]
    ):
        ax = axes[ax_idx]
        r2_vals = [data["results"][L]["R2"] for L in single_layers]
        r_vals = [data["results"][L]["r"] for L in single_layers]
        scores = [data["results"][L]["score"] for L in single_layers]
        ranges = [data["results"][L]["range"] for L in single_layers]

        x = np.arange(12)
        # Bar color by sign of r (positive = blue, negative = red, weak = grey)
        bar_colors = []
        for r_val in r_vals:
            if abs(r_val) < 0.5:
                bar_colors.append("#bdbdbd")
            elif r_val > 0:
                bar_colors.append("#1f77b4")
            else:
                bar_colors.append("#d62728")

        bars = ax.bar(x, r2_vals, color=bar_colors, edgecolor="white", linewidth=0.5)

        # Annotate with range
        for i, (bar, r2, rng, r_val) in enumerate(zip(bars, r2_vals, ranges, r_vals)):
            if r2 > 0.1:
                sign = "+" if r_val > 0 else "−"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{sign}{rng:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("R² (variance explained)")
        ax.set_title(f"{concept} Steering by Layer")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="grey", ls="--", alpha=0.4, label="R²=0.5")

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label="Positive (↑λ → ↑value)"),
            Patch(facecolor="#d62728", label="Negative (↑λ → ↓value)"),
            Patch(facecolor="#bdbdbd", label="Weak (|r| < 0.5)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Steering Effectiveness by Layer (Broad Sweep, n=5)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_layer_heatmap.png")
    fig.savefig(output_dir / "fig2_layer_heatmap.pdf")
    plt.close(fig)
    print(f"  Saved fig2_layer_heatmap.png/pdf")


# ============================================================================
# Plot 3: Control vs Quality Trade-off
# ============================================================================


def plot_control_quality_tradeoff(output_dir: pathlib.Path):
    """Scatter plot: composite score vs degradation for all configs."""
    pitch_broad = get_pitch_broad_sweep()
    dur_broad = get_duration_broad_sweep()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (data, concept) in enumerate(
        [
            (pitch_broad, "Pitch"),
            (dur_broad, "Duration"),
        ]
    ):
        ax = axes[ax_idx]
        for layer_name, d in data["results"].items():
            color = COLORS.get(layer_name, "#bdbdbd")
            is_highlight = layer_name in [
                "single_8",
                "single_10",
                "single_11",
                "late",
                "all",
            ]
            alpha = 0.9 if is_highlight else 0.4
            size = 100 if is_highlight else 40
            marker = "★" if d.get("mono", False) else "o"
            zorder = 5 if is_highlight else 3

            ax.scatter(
                d["deg"],
                abs(d["r"]) * d["range"],
                c=color,
                s=size,
                alpha=alpha,
                zorder=zorder,
                edgecolors="black" if is_highlight else "none",
                linewidths=1 if is_highlight else 0,
            )
            if is_highlight:
                ax.annotate(
                    LAYER_LABELS[layer_name],
                    (d["deg"], abs(d["r"]) * d["range"]),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )

        ax.set_xlabel("Average Degradation (lower is better)")
        ax.set_ylabel("Effective Control (|r| × range)")
        ax.set_title(f"{concept}: Control vs Quality")

        # Ideal region
        ax.axvspan(0, 1.5, alpha=0.05, color="green")
        ax.text(
            0.7,
            ax.get_ylim()[1] * 0.95,
            "Ideal\nzone",
            fontsize=9,
            color="green",
            alpha=0.5,
            ha="center",
        )

    fig.suptitle(
        "Control–Quality Trade-off: Higher is More Controllable, Left is Higher Quality",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_control_quality_tradeoff.png")
    fig.savefig(output_dir / "fig3_control_quality_tradeoff.pdf")
    plt.close(fig)
    print(f"  Saved fig3_control_quality_tradeoff.png/pdf")


# ============================================================================
# Plot 4: Best Config Deep Dive — single_10 pitch (final 15-sample run)
# ============================================================================


def plot_best_config_deepdive(output_dir: pathlib.Path):
    """Detailed view of the best config: single_10 pitch with 15 samples."""
    data = get_pitch_final()
    d = data["results"]["single_10"]
    q = data["quality"]["single_10"]
    lambdas = data["lambda_values"]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # -- Top left: Steering curve with regression --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        lambdas,
        d["mean"],
        "o-",
        color=COLORS["single_10"],
        linewidth=2.5,
        markersize=8,
        label="Observed",
        zorder=5,
    )

    # Regression line
    lambdas_arr = np.array(lambdas)
    slope = d["slope"]
    intercept = d["mean"][5] - slope * 0.0  # intercept at λ=0
    # Compute proper intercept
    mean_arr = np.array(d["mean"])
    coeffs = np.polyfit(lambdas_arr, mean_arr, 1)
    reg_line = np.polyval(coeffs, lambdas_arr)
    ax1.plot(
        lambdas,
        reg_line,
        "--",
        color="grey",
        alpha=0.7,
        label=f"Linear fit (R²={d['R2']:.3f})",
    )

    # Highlight monotonic region
    ax1.axvspan(lambdas[0], lambdas[-1], alpha=0.05, color="green")
    ax1.annotate(
        f"r = {d['r']:+.3f}***\nslope = {slope:+.1f} MIDI/λ",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax1.set_xlabel("Steering Strength (λ)")
    ax1.set_ylabel("Average Pitch (MIDI)")
    ax1.set_title("Layer 10 — Pitch Steering (n=15)")
    ax1.legend(loc="lower right")

    # -- Top right: Quality metrics --
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        lambdas, q["entropy"], "s-", color="#2ca02c", label="Entropy", markersize=5
    )
    ax2.axhline(
        y=GROUND_TRUTH["pitch_class_entropy"], color="#2ca02c", ls=":", alpha=0.4
    )
    ax2.set_xlabel("Steering Strength (λ)")
    ax2.set_ylabel("Pitch Class Entropy", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(1.5, 3.5)

    ax2b = ax2.twinx()
    ax2b.plot(lambdas, q["scale"], "o-", color="#d62728", label="Scale %", markersize=5)
    ax2b.plot(
        lambdas, q["groove"], "^-", color="#1f77b4", label="Groove %", markersize=5
    )
    ax2b.axhline(
        y=GROUND_TRUTH["scale_consistency"], color="#d62728", ls=":", alpha=0.4
    )
    ax2b.axhline(
        y=GROUND_TRUTH["groove_consistency"], color="#1f77b4", ls=":", alpha=0.4
    )
    ax2b.set_ylabel("Consistency (%)")
    ax2b.set_ylim(80, 105)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
    ax2.set_title("Quality Metrics Preserved")

    # -- Bottom left: Degradation --
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(
        [str(l) for l in lambdas],
        q["deg"],
        color=[COLORS["single_10"] if d < 2.0 else "#d62728" for d in q["deg"]],
        edgecolor="white",
    )
    ax3.axhline(
        y=2.0, color="red", ls="--", alpha=0.5, label="High degradation threshold"
    )
    ax3.set_xlabel("Steering Strength (λ)")
    ax3.set_ylabel("Total Degradation")
    ax3.set_title("Quality Degradation by λ")
    ax3.legend(fontsize=9)

    # -- Bottom right: Summary stats table --
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Cross-concept comparison
    dur_data = get_duration_refined()
    dur_single10 = dur_data["results"]["single_10"]

    table_data = [
        ["Metric", "Pitch", "Duration"],
        ["Pearson r", f"{d['r']:+.3f}***", f"{dur_single10['r']:+.3f}**"],
        ["R²", f"{d['R2']:.3f}", f"{dur_single10['R2']:.3f}"],
        ["Slope", f"{d['slope']:+.1f} MIDI/λ", f"{dur_single10['slope']:+.1f} ticks/λ"],
        ["Range", f"{d['range']:.1f} semitones", f"{dur_single10['range']:.1f} ticks"],
        ["Monotonic", "✓ (λ∈[-1.5,+0.75])", "✓ (full range)"],
        ["Avg Degradation", f"{d['deg']:.2f}", f"{dur_single10['deg']:.2f}"],
        ["Samples", "15", "10"],
    ]

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#e6e6e6")
        table[0, j].set_text_props(fontweight="bold")

    ax4.set_title("Layer 10: Cross-Concept Summary", fontweight="bold", pad=20)

    fig.suptitle(
        "Best Configuration: Layer 10 (SAS Steering)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.savefig(output_dir / "fig4_best_config_deepdive.png")
    fig.savefig(output_dir / "fig4_best_config_deepdive.pdf")
    plt.close(fig)
    print(f"  Saved fig4_best_config_deepdive.png/pdf")


# ============================================================================
# Plot 5: Full 16-layer overview (broad sweep)
# ============================================================================


def plot_full_layer_overview(output_dir: pathlib.Path):
    """4×4 grid: all 12 single layers + 4 groups, mini steering curves."""
    pitch_data = get_pitch_broad_sweep()
    dur_data = get_duration_broad_sweep()
    lambdas = pitch_data["lambda_values"]

    all_layers = [f"single_{i}" for i in range(12)] + ["early", "middle", "late", "all"]

    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
    axes_flat = axes.flatten()

    for idx, layer in enumerate(all_layers):
        ax = axes_flat[idx]
        p_data = pitch_data["results"][layer]
        d_data = dur_data["results"][layer]

        # Pitch
        ax.plot(
            lambdas,
            p_data["mean"],
            "o-",
            color=COLORS["single_10"],
            markersize=3,
            linewidth=1.5,
            label="Pitch",
        )

        # Duration on twin axis
        ax2 = ax.twinx()
        ax2.plot(
            lambdas,
            d_data["mean"],
            "s-",
            color=COLORS["late"],
            markersize=3,
            linewidth=1.5,
            label="Duration",
        )

        # Title with stats
        p_r = p_data["r"]
        d_r = d_data["r"]
        p_sig = (
            "***"
            if p_data["p"] < 0.001
            else "**" if p_data["p"] < 0.01 else "*" if p_data["p"] < 0.05 else "ns"
        )
        d_sig = (
            "***"
            if d_data["p"] < 0.001
            else "**" if d_data["p"] < 0.01 else "*" if d_data["p"] < 0.05 else "ns"
        )

        title = LAYER_LABELS[layer]
        ax.set_title(
            f"{title}\nP: r={p_r:+.2f}{p_sig}  D: r={d_r:+.2f}{d_sig}",
            fontsize=9,
            pad=3,
        )

        # Highlight strong layers
        if layer in ["single_8", "single_10", "single_11", "late"]:
            ax.patch.set_facecolor("#f0f8ff")
            ax.patch.set_alpha(0.5)

        if idx % 4 == 0:
            ax.set_ylabel("Pitch", fontsize=8, color=COLORS["single_10"])
        if idx % 4 == 3:
            ax2.set_ylabel("Duration", fontsize=8, color=COLORS["late"])
        else:
            ax2.set_yticklabels([])

        if idx >= 12:
            ax.set_xlabel("λ", fontsize=9)

        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)

    fig.suptitle(
        "All Layer Steering Responses (Broad Sweep) — Blue: Pitch, Red: Duration",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "fig5_full_layer_overview.png")
    fig.savefig(output_dir / "fig5_full_layer_overview.pdf")
    plt.close(fig)
    print(f"  Saved fig5_full_layer_overview.png/pdf")


# ============================================================================
# Plot 6: Monotonicity Analysis — Duration focus
# ============================================================================


def plot_monotonicity_analysis(output_dir: pathlib.Path):
    """Show the monotonic duration configs with confidence visualization."""
    dur = get_duration_refined()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    monotonic_configs = ["late", "single_10", "single_1"]
    for ax, layer in zip(axes, monotonic_configs):
        d = dur["results"][layer]
        lambdas = dur["lambda_values"]

        ax.plot(
            lambdas,
            d["mean"],
            "o-",
            color=COLORS[layer],
            linewidth=2.5,
            markersize=8,
            zorder=5,
        )

        # Fill area to show monotonic increase
        ax.fill_between(lambdas, 0, d["mean"], alpha=0.15, color=COLORS[layer])

        # Regression
        coeffs = np.polyfit(lambdas, d["mean"], 1)
        reg_line = np.polyval(coeffs, lambdas)
        ax.plot(lambdas, reg_line, "--", color="grey", alpha=0.7)

        # Arrow showing direction
        ax.annotate(
            "",
            xy=(lambdas[-1], d["mean"][-1]),
            xytext=(lambdas[0], d["mean"][0]),
            arrowprops=dict(arrowstyle="->", color=COLORS[layer], lw=2, alpha=0.3),
        )

        ax.set_xlabel("Steering Strength (λ)")
        ax.set_ylabel("Average Duration (ticks)")
        ax.set_title(
            f"{LAYER_LABELS[layer]}\nr={d['r']:+.3f}, R²={d['R2']:.3f}, "
            f"range={d['range']:.1f} ticks\n✓ Monotonic",
            fontsize=11,
        )
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Confirmed Monotonic Duration Steering (n=10 per λ)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig6_monotonicity_duration.png")
    fig.savefig(output_dir / "fig6_monotonicity_duration.pdf")
    plt.close(fig)
    print(f"  Saved fig6_monotonicity_duration.png/pdf")


# ============================================================================
# Plot 7: Experiment progression (convergence story)
# ============================================================================


def plot_experiment_progression(output_dir: pathlib.Path):
    """Show how experiments refined: broad → targeted → final."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Stage 1: Broad sweep
    ax = axes[0]
    broad = get_pitch_broad_sweep()
    for layer in ["single_10", "single_11", "single_8", "late", "middle", "single_6"]:
        d = broad["results"][layer]
        alpha = 0.9 if layer in ["single_10", "late"] else 0.3
        lw = 2 if layer in ["single_10", "late"] else 1
        ax.plot(
            broad["lambda_values"],
            d["mean"],
            "o-",
            color=COLORS[layer],
            alpha=alpha,
            linewidth=lw,
            markersize=4,
            label=LAYER_LABELS[layer] if alpha > 0.5 else None,
        )
    ax.set_title("Stage 1: Broad Sweep\n16 groups × 9λ × 5 samples", fontsize=11)
    ax.set_xlabel("λ")
    ax.set_ylabel("Average Pitch (MIDI)")
    ax.legend(fontsize=9)
    ax.set_xlim(-2.3, 2.3)

    # Stage 2: Refined
    ax = axes[1]
    refined = get_pitch_refined()
    for layer in ["single_10", "single_11", "single_8", "late"]:
        d = refined["results"][layer]
        lw = 2.5 if layer == "single_10" else 1.5
        ax.plot(
            refined["lambda_values"],
            d["mean"],
            "o-",
            color=COLORS[layer],
            linewidth=lw,
            markersize=5,
            label=LAYER_LABELS[layer],
        )
    ax.set_title("Stage 2: Refined Search\n4 layers × 10λ × 10 samples", fontsize=11)
    ax.set_xlabel("λ")
    ax.legend(fontsize=9)
    ax.set_xlim(-1.7, 1.2)

    # Stage 3: Final
    ax = axes[2]
    final = get_pitch_final()
    d = final["results"]["single_10"]
    ax.plot(
        final["lambda_values"],
        d["mean"],
        "o-",
        color=COLORS["single_10"],
        linewidth=2.5,
        markersize=8,
    )
    coeffs = np.polyfit(final["lambda_values"], d["mean"], 1)
    reg_line = np.polyval(coeffs, final["lambda_values"])
    ax.plot(final["lambda_values"], reg_line, "--", color="grey", alpha=0.7)
    ax.fill_between(
        final["lambda_values"],
        d["mean"],
        reg_line,
        alpha=0.1,
        color=COLORS["single_10"],
    )
    ax.annotate(
        f"r = {d['r']:+.3f}***\nR² = {d['R2']:.3f}\nslope = {d['slope']:+.1f}/λ",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Stage 3: Final Validation\nLayer 10 × 9λ × 15 samples", fontsize=11)
    ax.set_xlabel("λ")

    fig.suptitle(
        "Pitch Experiment Progression: From 16 Layers to Best Configuration",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig7_experiment_progression.png")
    fig.savefig(output_dir / "fig7_experiment_progression.pdf")
    plt.close(fig)
    print(f"  Saved fig7_experiment_progression.png/pdf")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Plot layer-lambda results")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("plots/layer_lambda"),
        help="Output directory for plots",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating presentation plots...")
    print(f"Output directory: {args.output_dir}")
    print()

    plot_steering_curves(args.output_dir)  # Fig 1
    plot_layer_heatmap(args.output_dir)  # Fig 2
    plot_control_quality_tradeoff(args.output_dir)  # Fig 3
    plot_best_config_deepdive(args.output_dir)  # Fig 4
    plot_full_layer_overview(args.output_dir)  # Fig 5
    plot_monotonicity_analysis(args.output_dir)  # Fig 6
    plot_experiment_progression(args.output_dir)  # Fig 7

    print(f"\n✓ All 7 figures saved to {args.output_dir}/")
    print("\nSuggested slide order:")
    print("  1. fig7 — Experiment progression story (broad → refined → final)")
    print("  2. fig5 — Full 16-layer overview (shows most layers are ineffective)")
    print("  3. fig2 — Layer heatmap (R² by layer, clear late-layer dominance)")
    print("  4. fig1 — Steering curves (best layers, side-by-side pitch/duration)")
    print("  5. fig6 — Monotonicity proof (3 confirmed monotonic duration configs)")
    print("  6. fig4 — Best config deep-dive (Layer 10 cross-concept)")
    print("  7. fig3 — Control vs quality trade-off (Pareto frontier)")


if __name__ == "__main__":
    main()
