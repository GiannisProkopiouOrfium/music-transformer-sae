#!/usr/bin/env python3
"""
Batch Feature Extraction and Intervention Pipeline

This script automates the extraction of Linear Musical Features (LiMuFs) from SAE
columns and performs conditioned controlled interventions for systematic evaluation.

Process:
1. Extract LiMuFs for selected features using extract_limuf_sae_column.py
2. Generate conditioned interventions using conditioned_controlled_feature_intervention.py
3. Create organized output structure for evaluation
4. Track progress and handle errors/resumption

Features are selected based on:
- High activation frequency
- Clear musical interpretation
- Diverse musical categories
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import logging

# Configuration of selected features per layer
SELECTED_FEATURES = {
    1: [
        {
            "feature_id": 325,
            "name": "rhythmic_displacement",
            "description": "Rhythmic displacement with sixteenth note shifts creating syncopation",
            "category": "rhythmic_specific",
            "activation_freq": 0.184,  # From diversity report
        },
        {
            "feature_id": 256,
            "name": "dynamic_contrasts",
            "description": "Sudden accents followed by soft dynamics creating dramatic ebb and flow",
            "category": "dynamic_specific",
            "activation_freq": 0.019,
        },
        {
            "feature_id": 1323,
            "name": "wide_pitch_range",
            "description": "Wide pitch range utilization across multiple instruments for textural depth",
            "category": "textural_specific",
            "activation_freq": 0.088,
        },
    ],
    3: [
        {
            "feature_id": 997,
            "name": "antiphonal_texture",
            "description": "Call-and-response interactions between instrumental groups",
            "category": "textural_specific",
            "activation_freq": 0.205,
        },
        {
            "feature_id": 855,
            "name": "dynamic_contrasts",
            "description": "Sudden shifts between high and low velocity ranges creating dramatic tension",
            "category": "dynamic_specific",
            "activation_freq": 0.005,
        },
        {
            "feature_id": 182,
            "name": "steady_pulse",
            "description": "Steady pulse patterns with metronomic consistency and march-like rhythms",
            "category": "rhythmic_specific",
            "activation_freq": 0.027,
        },
    ],
    5: [
        {
            "feature_id": 471,
            "name": "unison_doubling",
            "description": "Multiple instruments playing same melodic line in octaves for reinforced sound",
            "category": "textural_specific",
            "activation_freq": 0.034,
        },
        {
            "feature_id": 1950,
            "name": "dynamic_contrasts_layer5",
            "description": "Abrupt shifts between soft and loud dynamics creating dramatic expression",
            "category": "dynamic_specific",
            "activation_freq": 0.012,
        },
        {
            "feature_id": 904,
            "name": "rhythmic_augmentation",
            "description": "Systematic lengthening of motifs creating expansion and tension over time",
            "category": "rhythmic_specific",
            "activation_freq": 0.057,
        },
    ],
}

# Default intervention parameters
DEFAULT_INTERVENTION_PARAMS = {
    "addition_strengths": [-2.0, -1.0, 1.0, 2.0],  # 0.0 is baseline
    "conditioning_length": 2,
    "seq_len": 512,
    "temperature": 0.1,
    "noise_scale": 1.2,
    "conditioning_seed": 24,
    "generation_seed": 24,
}


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up comprehensive logging."""
    log_file = output_dir / "batch_extraction.log"

    logger = logging.getLogger("batch_extraction")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_output_structure(base_output_dir: Path) -> Dict[str, Path]:
    """Create organized output directory structure."""

    structure = {
        "base": base_output_dir,
        "extractions": base_output_dir / "extractions",
        "interventions": base_output_dir / "interventions",
        "progress": base_output_dir / "progress",
        "logs": base_output_dir / "logs",
    }

    # Create all directories
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)

    # Create per-layer subdirectories
    for layer in [1, 3, 5]:
        (structure["extractions"] / f"layer{layer}").mkdir(exist_ok=True)
        (structure["interventions"] / f"layer{layer}").mkdir(exist_ok=True)

    return structure


def load_progress(progress_file: Path) -> Dict:
    """Load extraction progress from file."""
    if progress_file.exists():
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"completed_extractions": [], "completed_interventions": []}


def save_progress(progress_file: Path, progress: Dict):
    """Save extraction progress to file."""
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def run_extraction(
    layer: int, feature_id: int, output_dir: Path, logger: logging.Logger
) -> bool:
    """Run LiMuF extraction for a specific feature."""

    extraction_output_dir = output_dir / "extractions" / f"layer{layer}"

    cmd = [
        "python",
        "extract_limuf_sae_column.py",
        "--layer",
        str(layer),
        "--feature-id",
        str(feature_id),
        "--output-dir",
        str(extraction_output_dir),
    ]

    logger.info(f"🔄 Extracting Layer {layer}, Feature {feature_id}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=Path.cwd()
        )

        logger.info(f"✅ Extraction completed for Layer {layer}, Feature {feature_id}")
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Extraction failed for Layer {layer}, Feature {feature_id}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during extraction: {e}")
        return False


def run_intervention(
    layer: int,
    feature_info: Dict,
    output_dir: Path,
    intervention_params: Dict,
    logger: logging.Logger,
) -> bool:
    """Run conditioned intervention for a specific feature."""

    feature_id = feature_info["feature_id"]

    # Construct paths
    extraction_dir = output_dir / "extractions" / f"layer{layer}"
    intervention_output_dir = (
        output_dir
        / "interventions"
        / f"layer{layer}"
        / f"feature{feature_id}_{feature_info['name']}"
    )

    # Find the extracted LiMuF file
    limuf_pattern = f"limufs_layer{layer}_sae_columns"
    limuf_path = extraction_dir / limuf_pattern / "limufs.pt"

    if not limuf_path.exists():
        logger.error(f"❌ LiMuF file not found: {limuf_path}")
        return False

    # Prepare intervention command
    addition_strengths_str = ",".join(
        map(str, intervention_params["addition_strengths"])
    )

    cmd = [
        "python",
        "mmt/conditioned_controlled_feature_intervention.py",
        "--feature-limuf-path",
        str(limuf_path),
        "--output-dir",
        str(intervention_output_dir),
        "--intervention-layer",
        str(layer),
        "--addition-strengths",
        addition_strengths_str,
        "--conditioning-length",
        str(intervention_params["conditioning_length"]),
        "--seq-len",
        str(intervention_params["seq_len"]),
        "--temperature",
        str(intervention_params["temperature"]),
        "--noise-scale",
        str(intervention_params["noise_scale"]),
        "--conditioning-seed",
        str(intervention_params["conditioning_seed"]),
        "--generation-seed",
        str(intervention_params["generation_seed"]),
    ]

    logger.info(
        f"🎵 Running interventions for Layer {layer}, Feature {feature_id} ({feature_info['name']})"
    )
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, cwd=Path.cwd()
        )

        logger.info(
            f"✅ Interventions completed for Layer {layer}, Feature {feature_id}"
        )
        if result.stdout:
            # Log last few lines of output to avoid spam
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines[-5:]:
                logger.info(f"Output: {line}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Intervention failed for Layer {layer}, Feature {feature_id}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during intervention: {e}")
        return False


def create_experiment_summary(
    output_dir: Path, results: Dict, intervention_params: Dict
):
    """Create comprehensive experiment summary."""

    summary = {
        "experiment_info": {
            "type": "batch_feature_extraction_and_intervention",
            "timestamp": datetime.now().isoformat(),
            "total_features": sum(
                len(features) for features in SELECTED_FEATURES.values()
            ),
            "layers_processed": list(SELECTED_FEATURES.keys()),
            "intervention_parameters": intervention_params,
        },
        "feature_selection_criteria": {
            "high_activation_frequency": "Features with significant activation patterns",
            "clear_musical_interpretation": "Features with well-understood musical meaning",
            "diverse_categories": "Covering rhythmic, dynamic, textural, and structural aspects",
        },
        "selected_features": SELECTED_FEATURES,
        "processing_results": results,
        "output_structure": {
            "extractions/": "LiMuF extraction outputs per layer",
            "interventions/": "Conditioned intervention results per layer/feature",
            "progress/": "Processing progress and resume information",
            "logs/": "Detailed processing logs",
        },
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_file


def main():
    """Main batch extraction and intervention pipeline."""

    parser = argparse.ArgumentParser(
        description="Batch feature extraction and intervention pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="batch_extractions_interventions",
        help="Base output directory for all results",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        choices=[1, 3, 5],
        help="Layers to process",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step (use existing extractions)",
    )
    parser.add_argument(
        "--skip-intervention",
        action="store_true",
        help="Skip intervention step (only run extractions)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous progress"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Override intervention parameters
    parser.add_argument(
        "--conditioning-length", type=int, help="Override conditioning length"
    )
    parser.add_argument("--seq-len", type=int, help="Override sequence length")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--noise-scale", type=float, help="Override noise scale")

    args = parser.parse_args()

    # Create output structure
    output_structure = create_output_structure(args.output_dir)
    logger = setup_logging(args.output_dir)

    # Override intervention parameters if provided
    intervention_params = DEFAULT_INTERVENTION_PARAMS.copy()
    if args.conditioning_length is not None:
        intervention_params["conditioning_length"] = args.conditioning_length
    if args.seq_len is not None:
        intervention_params["seq_len"] = args.seq_len
    if args.temperature is not None:
        intervention_params["temperature"] = args.temperature
    if args.noise_scale is not None:
        intervention_params["noise_scale"] = args.noise_scale

    logger.info("🚀 STARTING BATCH FEATURE EXTRACTION AND INTERVENTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Processing layers: {args.layers}")
    logger.info(f"Skip extraction: {args.skip_extraction}")
    logger.info(f"Skip intervention: {args.skip_intervention}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Intervention parameters: {intervention_params}")

    # Load progress if resuming
    progress_file = output_structure["progress"] / "progress.json"
    progress = (
        load_progress(progress_file)
        if args.resume
        else {"completed_extractions": [], "completed_interventions": []}
    )

    results = {
        "successful_extractions": [],
        "failed_extractions": [],
        "successful_interventions": [],
        "failed_interventions": [],
    }

    total_features = sum(
        len(features)
        for layer in args.layers
        for features in [SELECTED_FEATURES[layer]]
    )
    current_feature = 0

    logger.info(f"\n📊 PROCESSING PLAN:")
    for layer in args.layers:
        features = SELECTED_FEATURES[layer]
        logger.info(f"Layer {layer}: {len(features)} features")
        for feature in features:
            logger.info(
                f"  - Feature {feature['feature_id']}: {feature['name']} ({feature['category']})"
            )

    logger.info(f"\n🎯 STARTING PROCESSING ({total_features} total features)")

    # Process each layer and feature
    for layer in args.layers:
        features = SELECTED_FEATURES[layer]

        logger.info(f"\n🔄 PROCESSING LAYER {layer} ({len(features)} features)")

        for feature_info in features:
            current_feature += 1
            feature_id = feature_info["feature_id"]
            feature_name = feature_info["name"]

            logger.info(
                f"\n📍 [{current_feature}/{total_features}] Layer {layer}, Feature {feature_id} ({feature_name})"
            )

            extraction_key = f"layer{layer}_feature{feature_id}"
            intervention_key = f"layer{layer}_feature{feature_id}"

            # Step 1: Extraction
            if not args.skip_extraction:
                if extraction_key not in progress["completed_extractions"]:
                    if args.dry_run:
                        logger.info(
                            f"🔍 [DRY RUN] Would extract Layer {layer}, Feature {feature_id}"
                        )
                    else:
                        if run_extraction(layer, feature_id, args.output_dir, logger):
                            results["successful_extractions"].append(extraction_key)
                            progress["completed_extractions"].append(extraction_key)
                            save_progress(progress_file, progress)
                        else:
                            results["failed_extractions"].append(extraction_key)
                            logger.error(
                                f"❌ Skipping intervention for failed extraction: {extraction_key}"
                            )
                            continue
                else:
                    logger.info(f"⏭️  Extraction already completed: {extraction_key}")

            # Step 2: Intervention
            if not args.skip_intervention:
                if intervention_key not in progress["completed_interventions"]:
                    if args.dry_run:
                        logger.info(
                            f"🎵 [DRY RUN] Would run interventions for Layer {layer}, Feature {feature_id}"
                        )
                    else:
                        if run_intervention(
                            layer,
                            feature_info,
                            args.output_dir,
                            intervention_params,
                            logger,
                        ):
                            results["successful_interventions"].append(intervention_key)
                            progress["completed_interventions"].append(intervention_key)
                            save_progress(progress_file, progress)
                        else:
                            results["failed_interventions"].append(intervention_key)
                else:
                    logger.info(
                        f"⏭️  Intervention already completed: {intervention_key}"
                    )

    # Create experiment summary
    if not args.dry_run:
        summary_file = create_experiment_summary(
            args.output_dir, results, intervention_params
        )
        logger.info(f"\n📊 Experiment summary saved: {summary_file}")

    # Final results
    logger.info("\n" + "=" * 80)
    logger.info("🎉 BATCH PROCESSING COMPLETE!")
    logger.info(f"✅ Successful extractions: {len(results['successful_extractions'])}")
    logger.info(f"❌ Failed extractions: {len(results['failed_extractions'])}")
    logger.info(
        f"✅ Successful interventions: {len(results['successful_interventions'])}"
    )
    logger.info(f"❌ Failed interventions: {len(results['failed_interventions'])}")

    if results["failed_extractions"]:
        logger.warning(f"Failed extractions: {results['failed_extractions']}")
    if results["failed_interventions"]:
        logger.warning(f"Failed interventions: {results['failed_interventions']}")

    logger.info(f"\n📁 Output structure created in: {args.output_dir}")
    logger.info("🎵 Ready for systematic evaluation!")

    return (
        len(results["failed_extractions"]) + len(results["failed_interventions"]) == 0
    )


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
