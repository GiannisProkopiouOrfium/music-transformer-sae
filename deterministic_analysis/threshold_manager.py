#!/usr/bin/env python3
"""
Threshold Manager

Manages loading and application of calibrated thresholds for quality assessment.
Provides both absolute and relative threshold checking based on training data distribution.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging


class ThresholdManager:
    """Manage calibrated thresholds for musical quality assessment."""

    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize threshold manager.

        Args:
            calibration_file: Path to calibrated_thresholds.json file
                             If None, looks for file in project root
        """
        self.logger = logging.getLogger(__name__)

        # Find calibration file
        if calibration_file is None:
            project_root = Path(__file__).parent.parent
            calibration_file = project_root / "calibrated_thresholds.json"

        self.calibration_file = Path(calibration_file)

        # Load thresholds
        self.thresholds = self.load_thresholds()

    def load_thresholds(self) -> Dict[str, Any]:
        """Load calibrated thresholds from JSON file."""
        if not self.calibration_file.exists():
            self.logger.warning(
                f"Calibration file not found: {self.calibration_file}. "
                "Using default thresholds."
            )
            return {"general_thresholds": {}, "feature_specific_thresholds": {}}

        try:
            with open(self.calibration_file, "r") as f:
                data = json.load(f)

            self.logger.info(
                f"Loaded calibrated thresholds from {self.calibration_file}"
            )
            return data

        except Exception as e:
            self.logger.error(f"Failed to load calibration file: {e}")
            return {"general_thresholds": {}, "feature_specific_thresholds": {}}

    def get_metric_percentile_rank(
        self, metric_path: str, value: float
    ) -> Optional[float]:
        """
        Calculate what percentile rank a given metric value falls into.

        Args:
            metric_path: Full metric path (e.g., "note_patterns.average_velocity")
            value: The metric value to evaluate

        Returns:
            Percentile rank (0-100) or None if metric not found
        """
        general = self.thresholds.get("general_thresholds", {})

        if metric_path not in general:
            return None

        percentiles = general[metric_path]["percentiles"]

        # Determine which percentile bracket the value falls into
        if value <= percentiles["p5"]:
            return 2.5  # Below P5
        elif value <= percentiles["p10"]:
            return 7.5  # P5-P10
        elif value <= percentiles["p25"]:
            return 17.5  # P10-P25
        elif value <= percentiles["p50"]:
            return 37.5  # P25-P50
        elif value <= percentiles["p75"]:
            return 62.5  # P50-P75
        elif value <= percentiles["p90"]:
            return 82.5  # P75-P90
        elif value <= percentiles["p95"]:
            return 92.5  # P90-P95
        else:
            return 97.5  # Above P95

    def get_quality_level_from_percentile(
        self, percentile_rank: float, higher_is_better: bool = True
    ) -> str:
        """
        Convert percentile rank to quality level.

        Args:
            percentile_rank: Percentile rank (0-100)
            higher_is_better: Whether higher values indicate better quality

        Returns:
            Quality level: "excellent", "good", "moderate", "acceptable", or "poor"
        """
        if not higher_is_better:
            percentile_rank = 100 - percentile_rank

        if percentile_rank >= 75:
            return "excellent"
        elif percentile_rank >= 50:
            return "good"
        elif percentile_rank >= 25:
            return "moderate"
        elif percentile_rank >= 10:
            return "acceptable"
        else:
            return "poor"

    def assess_metric_quality(
        self, metric_path: str, value: float, higher_is_better: bool = True
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Assess quality of a metric value using calibrated thresholds.

        Args:
            metric_path: Full metric path (e.g., "note_patterns.average_velocity")
            value: The metric value to assess
            higher_is_better: Whether higher values indicate better quality

        Returns:
            Tuple of (quality_level, score, details)
            - quality_level: "excellent", "good", "moderate", "acceptable", "poor"
            - score: Normalized score 0-1
            - details: Dict with percentile info and thresholds
        """
        general = self.thresholds.get("general_thresholds", {})

        if metric_path not in general:
            # Fallback for unknown metrics
            return "moderate", 0.5, {"error": "Metric not calibrated"}

        metric_data = general[metric_path]
        percentiles = metric_data["percentiles"]

        # Calculate percentile rank
        percentile_rank = self.get_metric_percentile_rank(metric_path, value)

        # Get quality level
        quality_level = self.get_quality_level_from_percentile(
            percentile_rank, higher_is_better
        )

        # Calculate normalized score (0-1)
        # Map percentile rank to score
        if higher_is_better:
            score = percentile_rank / 100.0
        else:
            score = (100 - percentile_rank) / 100.0

        details = {
            "percentile_rank": percentile_rank,
            "value": value,
            "mean": percentiles["mean"],
            "std": percentiles["std"],
            "median": percentiles["p50"],
            "good_range": [percentiles["p25"], percentiles["p75"]],
            "metric_path": metric_path,
        }

        return quality_level, score, details

    def assess_feature_metrics(
        self, feature_id: str, extracted_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess all metrics for a specific feature using calibrated thresholds.

        Args:
            feature_id: Feature ID (e.g., "325", "997")
            extracted_metrics: Dictionary of extracted metrics (nested structure)

        Returns:
            Assessment results with quality scores for each metric
        """
        feature_specific = self.thresholds.get("feature_specific_thresholds", {})

        if feature_id not in feature_specific:
            return {"error": f"Feature {feature_id} not found in calibrated thresholds"}

        feature_data = feature_specific[feature_id]
        metric_thresholds = feature_data.get("metric_thresholds", {})

        results = {
            "feature_id": feature_id,
            "feature_name": feature_data["name"],
            "layer": feature_data["layer"],
            "metric_assessments": {},
            "overall_quality_score": 0.0,
            "quality_level": "moderate",
        }

        scores = []

        # Assess each primary metric for this feature
        for metric_path in metric_thresholds.keys():
            # Parse metric path to access nested dict
            parts = metric_path.split(".")

            if len(parts) != 2:
                continue

            category, metric_name = parts

            # Get metric value from extracted_metrics
            if (
                category in extracted_metrics
                and metric_name in extracted_metrics[category]
            ):
                value = extracted_metrics[category][metric_name]

                # Assess quality (assume higher is better for most metrics)
                # TODO: Add per-metric direction configuration
                quality_level, score, details = self.assess_metric_quality(
                    metric_path, value, higher_is_better=True
                )

                results["metric_assessments"][metric_path] = {
                    "value": value,
                    "quality_level": quality_level,
                    "score": score,
                    "details": details,
                }

                scores.append(score)

        # Calculate overall quality
        if scores:
            results["overall_quality_score"] = sum(scores) / len(scores)
            results["quality_level"] = self.get_quality_level_from_percentile(
                results["overall_quality_score"] * 100, higher_is_better=True
            )

        return results

    def get_feature_metric_paths(self, feature_id: str) -> list:
        """Get list of primary metric paths for a feature."""
        feature_specific = self.thresholds.get("feature_specific_thresholds", {})

        if feature_id not in feature_specific:
            return []

        return list(feature_specific[feature_id].get("metric_thresholds", {}).keys())

    def get_all_calibrated_metrics(self) -> list:
        """Get list of all calibrated metric paths."""
        general = self.thresholds.get("general_thresholds", {})
        return list(general.keys())

    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get summary of loaded thresholds."""
        general = self.thresholds.get("general_thresholds", {})
        feature_specific = self.thresholds.get("feature_specific_thresholds", {})
        metadata = self.thresholds.get("metadata", {})

        return {
            "calibration_file": str(self.calibration_file),
            "metadata": metadata,
            "total_metrics": len(general),
            "features_covered": len(feature_specific),
            "metrics_by_feature": {
                fid: len(data.get("metric_thresholds", {}))
                for fid, data in feature_specific.items()
            },
        }
