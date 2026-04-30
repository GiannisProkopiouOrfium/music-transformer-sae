"""Threshold-Aware Baseline for SAS Steering.

Implements a simple dynamic baseline requested by reviewers:
instead of PID, use a step-function controller that holds λ=0
until the generation reaches the ramp point, then immediately
steps to the target λ. This tests whether the PID's integral
accumulation provides benefit over a simpler bang-bang strategy.

Two baselines:
  1. step: λ=0 for t < T_ramp, then λ=λ_target instantly
  2. binary_search: per-step binary search for minimal λ that breaches Top-K

Compares against PID and static cosine ramp.
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.temporal_pid_sas_generator import (
    TemporalPIDSASGenerator,
    load_sae_model,
    load_sas_vector,
    get_adaptive_k,
    get_target_feature_indices,
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_single_attribute import compute_generation_metrics
from pid_steering.run_temporal_pid_sas import generate_static_smooth

import config_pid

logger = logging.getLogger(__name__)


class StepSASHook:
    """Step-function baseline: λ=0 during ramp, then λ=λ_target."""

    def __init__(
        self,
        sae_model,
        sas_vector,
        n_ramp_steps,
        lambda_target,
        target_feature_indices=None,
    ):
        self.sae_model = sae_model
        self.sas_vector_t = torch.tensor(sas_vector, dtype=torch.float32)
        self.n_ramp_steps = n_ramp_steps
        self.lambda_target = lambda_target
        self.target_feature_indices = target_feature_indices
        self.step = 0
        self.lambda_trajectory = []
        self.feature_trajectory = []

    def reset(self):
        self.step = 0
        self.lambda_trajectory = []
        self.feature_trajectory = []

    def __call__(self, module, input, output, layer_idx):
        actual = output[0] if isinstance(output, tuple) else output
        device = actual.device
        sas_vec = self.sas_vector_t.to(device)

        # Step function: 0 during ramp, target after
        lam = 0.0 if self.step < self.n_ramp_steps else self.lambda_target
        self.lambda_trajectory.append(lam)

        with torch.no_grad():
            a_flat = actual.reshape(-1, actual.shape[-1])
            f_a = self.sae_model.encode(a_flat)

            # Monitor features
            if self.target_feature_indices is not None:
                idx = torch.tensor(self.target_feature_indices, device=device)
                act = f_a[:, idx].mean().item()
                self.feature_trajectory.append(act)

            # Apply steering
            s = f_a + lam * sas_vec.unsqueeze(0)
            s = torch.relu(s)
            s = self.sae_model.topk(s)
            delta = a_flat - self.sae_model.decode(f_a)
            steered = self.sae_model.decode(s) + delta

            self.step += 1

        new_out = steered.reshape(actual.shape)
        if isinstance(output, tuple):
            return (new_out,) + output[1:]
        return new_out

    def get_diagnostics(self):
        return {
            "lambda_trajectory": self.lambda_trajectory,
            "feature_activation_trajectory": self.feature_trajectory,
        }


class MinimalLambdaSASHook:
    """Binary-search baseline: find minimal λ that breaches Top-K per step.

    At each step, binary-search for the smallest λ ∈ [0, λ_max] such that
    at least 1 target feature survives Top-K re-sparsification.
    Uses a cosine schedule for the DESIRED number of active features.
    """

    def __init__(
        self,
        sae_model,
        sas_vector,
        n_ramp_steps,
        lambda_max,
        target_feature_indices,
        n_search_steps=8,
    ):
        self.sae_model = sae_model
        self.sas_vector_t = torch.tensor(sas_vector, dtype=torch.float32)
        self.n_ramp_steps = n_ramp_steps
        self.lambda_max = lambda_max
        self.target_feature_indices = target_feature_indices
        self.n_search_steps = n_search_steps
        self.step = 0
        self.lambda_trajectory = []
        self.feature_trajectory = []

    def reset(self):
        self.step = 0
        self.lambda_trajectory = []
        self.feature_trajectory = []

    def _desired_n_active(self):
        """Cosine ramp for number of desired active target features."""
        n_total = len(self.target_feature_indices)
        if self.step >= self.n_ramp_steps:
            return n_total
        frac = 0.5 * (1 - np.cos(np.pi * self.step / self.n_ramp_steps))
        return max(1, int(frac * n_total))

    def __call__(self, module, input, output, layer_idx):
        actual = output[0] if isinstance(output, tuple) else output
        device = actual.device
        sas_vec = self.sas_vector_t.to(device)

        with torch.no_grad():
            a_flat = actual.reshape(-1, actual.shape[-1])
            f_a = self.sae_model.encode(a_flat)
            idx_t = torch.tensor(self.target_feature_indices, device=device)

            desired = self._desired_n_active()

            # Binary search for minimal λ
            lo, hi = 0.0, self.lambda_max
            best_lam = hi
            for _ in range(self.n_search_steps):
                mid = (lo + hi) / 2
                s_test = f_a + mid * sas_vec.unsqueeze(0)
                s_test = torch.relu(s_test)
                s_test = self.sae_model.topk(s_test)
                n_active = (s_test[:, idx_t] > 0).float().sum().item()
                if n_active >= desired:
                    best_lam = mid
                    hi = mid
                else:
                    lo = mid

            lam = best_lam
            self.lambda_trajectory.append(lam)

            # Apply with found λ
            s = f_a + lam * sas_vec.unsqueeze(0)
            s = torch.relu(s)
            s = self.sae_model.topk(s)

            act = s[:, idx_t].mean().item()
            self.feature_trajectory.append(act)

            delta = a_flat - self.sae_model.decode(f_a)
            steered = self.sae_model.decode(s) + delta

            self.step += 1

        new_out = steered.reshape(actual.shape)
        if isinstance(output, tuple):
            return (new_out,) + output[1:]
        return new_out

    def get_diagnostics(self):
        return {
            "lambda_trajectory": self.lambda_trajectory,
            "feature_activation_trajectory": self.feature_trajectory,
        }


def generate_with_hook(
    model, encoding, hook_obj, target_layer, n_samples, seq_len, device
):
    """Generate samples using a custom SAS hook."""
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    sequences = []
    diagnostics = []

    layers = model.decoder.net.attn_layers.layers
    layer_module = layers[target_layer]
    if isinstance(layer_module, torch.nn.ModuleList) and len(layer_module) > 1:
        target_module = layer_module[1]
    else:
        target_module = layer_module

    for i in range(n_samples):
        hook_obj.reset()
        handle = target_module.register_forward_hook(
            lambda mod, inp, out, h=hook_obj, idx=target_layer: h(mod, inp, out, idx)
        )
        try:
            start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start[:, 0, 0] = sos
            with torch.no_grad():
                generated = model.generate(
                    start,
                    seq_len,
                    eos_token=eos,
                    temperature=config_pid.TEMPERATURE,
                    filter_logits_fn="top_k",
                    filter_thres=config_pid.FILTER_THRESHOLD,
                    monotonicity_dim=("type", "beat"),
                )
            full_seq = torch.cat((start, generated), 1).cpu().numpy()[0]
            sequences.append(full_seq)
            diagnostics.append(hook_obj.get_diagnostics())
        finally:
            handle.remove()

        if (i + 1) % 5 == 0:
            logger.info(f"  Generated {i+1}/{n_samples}")

    return sequences, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Threshold-aware baselines vs PID for SAS steering"
    )
    parser.add_argument("--concept", type=str, nargs="+", default=["average_pitch"])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument(
        "--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "threshold_baselines",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--direction", type=str, choices=["positive", "negative"], default="positive"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    concepts = args.concept
    if concepts == ["all"]:
        concepts = ["average_pitch", "average_duration"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )
    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    for concept in concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Threshold baselines for: {concept} ({args.direction})")
        logger.info(f"{'='*70}")

        vector_path = config_pid.SAS_VECTORS_DIR / f"{concept}_sas_vectors.pt"
        sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)
        if args.direction == "negative":
            sas_vector = -sas_vector

        target_indices = get_target_feature_indices(sas_vector)
        results = {}

        # 1. Step baseline
        logger.info("Running Step baseline (λ=0 then λ=target)...")
        step_hook = StepSASHook(
            sae_model,
            sas_vector,
            args.n_ramp_steps,
            args.lambda_max,
            target_feature_indices=target_indices,
        )
        step_seqs, step_diags = generate_with_hook(
            model,
            encoding,
            step_hook,
            args.target_layer,
            args.n_samples,
            args.seq_len,
            device,
        )
        step_metrics = [compute_generation_metrics(s, encoding) for s in step_seqs]
        results["step"] = {
            "per_sample_metrics": step_metrics,
            "diagnostics": step_diags,
            "avg_lambda": float(
                np.mean([np.mean(d["lambda_trajectory"]) for d in step_diags])
            ),
        }

        # 2. Minimal-λ binary search baseline
        logger.info("Running Minimal-λ binary search baseline...")
        minlam_hook = MinimalLambdaSASHook(
            sae_model,
            sas_vector,
            args.n_ramp_steps,
            args.lambda_max,
            target_indices,
            n_search_steps=8,
        )
        minlam_seqs, minlam_diags = generate_with_hook(
            model,
            encoding,
            minlam_hook,
            args.target_layer,
            args.n_samples,
            args.seq_len,
            device,
        )
        minlam_metrics = [compute_generation_metrics(s, encoding) for s in minlam_seqs]
        results["minimal_lambda"] = {
            "per_sample_metrics": minlam_metrics,
            "diagnostics": minlam_diags,
            "avg_lambda": float(
                np.mean([np.mean(d["lambda_trajectory"]) for d in minlam_diags])
            ),
        }

        # 3. PID (for comparison)
        logger.info("Running Temporal PID...")
        pid_gen = TemporalPIDSASGenerator(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
        )
        pid_seqs, pid_diags = pid_gen.generate_with_temporal_pid(
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            target_magnitude=args.target_magnitude,
            Kp=args.Kp,
            Ki=args.Ki,
            Kd=args.Kd,
            max_I=args.max_I,
            lambda_max=args.lambda_max,
            n_ramp_beats=args.n_ramp_steps,
            device=device,
        )
        pid_metrics = [compute_generation_metrics(s, encoding) for s in pid_seqs]
        results["pid"] = {
            "per_sample_metrics": pid_metrics,
            "diagnostics": [d for d in pid_diags],
            "avg_lambda": float(
                np.mean(
                    [
                        np.mean(d["lambda_trajectory"])
                        for d in pid_diags
                        if d["lambda_trajectory"]
                    ]
                )
            ),
        }

        # 4. Unsteered baseline
        logger.info("Running unsteered baseline...")
        base_seqs, _ = generate_static_smooth(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
            lambda_max=0.0,
            n_ramp_steps=0,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            device=device,
        )
        base_metrics = [compute_generation_metrics(s, encoding) for s in base_seqs]
        results["baseline"] = {"per_sample_metrics": base_metrics}

        # Compute summary with CIs
        args.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Threshold Baseline Comparison — {concept} ({args.direction})")
        print(f"{'='*80}")
        print(
            f"{'Method':<18} {'Avg Pitch':>10} {'Avg Dur':>10} "
            f"{'PC Ent':>8} {'Scale%':>8} {'Groove%':>8} {'δ':>6} {'Avg λ':>7}"
        )
        print("-" * 80)

        gt_h, gt_s, gt_g = 2.974, 92.26, 93.05

        for method_name, method_data in results.items():
            metrics_list = method_data["per_sample_metrics"]
            pitch_vals = [m["average_pitch"] for m in metrics_list]
            dur_vals = [m["average_duration"] for m in metrics_list]
            h_vals = [m["pitch_class_entropy"] for m in metrics_list]
            s_vals = [m["scale_consistency"] for m in metrics_list]
            g_vals = [m["groove_consistency"] for m in metrics_list]

            avg_p = np.mean(pitch_vals)
            avg_d = np.mean(dur_vals)
            avg_h = np.mean(h_vals)
            avg_s = np.mean(s_vals)
            avg_g = np.mean(g_vals)
            delta = abs(avg_h - gt_h) + max(0, gt_s - avg_s) + max(0, gt_g - avg_g)

            avg_lam = method_data.get("avg_lambda", 0.0)
            print(
                f"{method_name:<18} {avg_p:>10.2f} {avg_d:>10.2f} "
                f"{avg_h:>8.3f} {avg_s:>8.1f} {avg_g:>8.1f} {delta:>6.2f} {avg_lam:>7.3f}"
            )

        # Save
        save_results = {}
        for method_name, method_data in results.items():
            save_results[method_name] = {
                "per_sample_metrics": method_data["per_sample_metrics"],
                "avg_lambda": method_data.get("avg_lambda", 0.0),
            }

        out_path = (
            args.output_dir / f"threshold_baselines_{concept}_{args.direction}.json"
        )
        with open(out_path, "w") as f:
            json.dump(save_results, f, indent=2, default=float)
        logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
