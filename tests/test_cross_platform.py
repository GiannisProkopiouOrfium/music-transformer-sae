#!/usr/bin/env python3
"""
Cross-Platform SAE Testing Suite
Tests the unified SAE pipeline on both macOS and AWS EC2/Ubuntu platforms.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import yaml
import json
from datetime import datetime


class CrossPlatformTester:
    """Test suite for cross-platform SAE pipeline."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.platform_name = platform.system().lower()
        self.is_macos = self.platform_name == "darwin"
        self.is_linux = self.platform_name == "linux"

        # Test configurations
        self.test_configs = {
            "macos": {
                "config_file": "configs/macos_test_config.yaml",
                "output_dir": "test_results_macos",
                "experiment_name": "macos_cross_platform_test",
            },
            "aws_ec2": {
                "config_file": "configs/aws_ec2_config.yaml",
                "output_dir": "test_results_aws_ec2",
                "experiment_name": "aws_ec2_cross_platform_test",
            },
        }

    def detect_environment(self):
        """Detect the current environment."""
        env_info = {
            "platform": self.platform_name,
            "is_macos": self.is_macos,
            "is_linux": self.is_linux,
            "is_aws_ec2": self._is_aws_ec2(),
            "python_version": platform.python_version(),
            "working_directory": str(self.base_dir),
        }

        return env_info

    def _is_aws_ec2(self):
        """Check if running on AWS EC2."""
        if not self.is_linux:
            return False

        try:
            import urllib.request

            req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"User-Agent": "aws-ec2-metadata/1.0"},
            )
            urllib.request.urlopen(req, timeout=2)
            return True
        except:
            return False

    def test_dependencies(self):
        """Test if all required dependencies are available."""
        dependencies = {
            "torch": "PyTorch",
            "numpy": "NumPy",
            "h5py": "HDF5",
            "matplotlib": "Matplotlib",
            "yaml": "PyYAML",
            "tqdm": "TQDM",
            "psutil": "psutil",
        }

        results = {}
        print("🔍 Testing Dependencies...")

        for module, name in dependencies.items():
            try:
                __import__(module)
                results[module] = {"status": "available", "name": name}
                print(f"  ✅ {name}: Available")
            except ImportError as e:
                results[module] = {"status": "missing", "name": name, "error": str(e)}
                print(f"  ❌ {name}: Missing ({e})")

        return results

    def validate_configs(self):
        """Validate configuration files."""
        results = {}
        print("\n📋 Validating Configurations...")

        for platform_name, config_info in self.test_configs.items():
            config_path = self.base_dir / config_info["config_file"]

            if not config_path.exists():
                results[platform_name] = {"status": "missing", "path": str(config_path)}
                print(f"  ❌ {platform_name}: Config file missing at {config_path}")
                continue

            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # Validate required sections
                required_sections = ["model", "data", "extraction", "sae"]
                missing_sections = [s for s in required_sections if s not in config]

                if missing_sections:
                    results[platform_name] = {
                        "status": "invalid",
                        "missing_sections": missing_sections,
                        "path": str(config_path),
                    }
                    print(f"  ❌ {platform_name}: Missing sections {missing_sections}")
                else:
                    results[platform_name] = {
                        "status": "valid",
                        "config": config,
                        "path": str(config_path),
                    }
                    print(f"  ✅ {platform_name}: Valid configuration")

            except Exception as e:
                results[platform_name] = {
                    "status": "error",
                    "error": str(e),
                    "path": str(config_path),
                }
                print(f"  ❌ {platform_name}: Error loading config ({e})")

        return results

    def run_pipeline_test(self, platform_name):
        """Run pipeline test for specific platform."""
        print(f"\n🚀 Testing Pipeline for {platform_name.upper()}...")

        if platform_name not in self.test_configs:
            print(f"  ❌ Unknown platform: {platform_name}")
            return {"status": "error", "error": "Unknown platform"}

        config_info = self.test_configs[platform_name]
        config_path = self.base_dir / config_info["config_file"]
        output_dir = self.base_dir / config_info["output_dir"]
        experiment_name = config_info["experiment_name"]

        # Ensure output directory exists
        output_dir.mkdir(exist_ok=True)

        # Build command
        pipeline_script = self.base_dir / "pipeline" / "extract.py"
        cmd = [
            sys.executable,
            str(pipeline_script),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--experiment-name",
            experiment_name,
            "--stage",
            "all",
        ]

        print(f"  🔄 Running: {' '.join(cmd)}")

        start_time = datetime.now()

        try:
            # Run the pipeline
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            end_time = datetime.now()
            duration = end_time - start_time

            # Check results
            expected_files = [
                output_dir / "activations.h5",
                output_dir / "sae_model.pt",
                output_dir / "analysis_results.h5",
                output_dir / "sae_analysis.png",
            ]

            missing_files = [f for f in expected_files if not f.exists()]

            test_result = {
                "status": (
                    "success"
                    if result.returncode == 0 and not missing_files
                    else "failed"
                ),
                "return_code": result.returncode,
                "duration": str(duration),
                "output_dir": str(output_dir),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "expected_files": [str(f) for f in expected_files],
                "missing_files": [str(f) for f in missing_files],
            }

            if test_result["status"] == "success":
                print(f"  ✅ Pipeline completed successfully in {duration}")
            else:
                print(f"  ❌ Pipeline failed (return code: {result.returncode})")
                if missing_files:
                    print(f"      Missing files: {[f.name for f in missing_files]}")

            return test_result

        except subprocess.TimeoutExpired:
            print(f"  ❌ Pipeline timed out after 5 minutes")
            return {
                "status": "timeout",
                "duration": "300+ seconds",
                "error": "Pipeline execution timed out",
            }
        except Exception as e:
            print(f"  ❌ Pipeline execution failed: {e}")
            return {"status": "error", "error": str(e)}

    def run_all_tests(self):
        """Run complete test suite."""
        print("🧪 Cross-Platform SAE Testing Suite")
        print("=" * 50)

        # Environment detection
        env_info = self.detect_environment()
        print(
            f"Environment: {env_info['platform']} (Python {env_info['python_version']})"
        )
        print(f"Working Directory: {env_info['working_directory']}")

        # Test results
        results = {
            "environment": env_info,
            "timestamp": datetime.now().isoformat(),
            "dependencies": self.test_dependencies(),
            "configurations": self.validate_configs(),
            "pipeline_tests": {},
        }

        # Determine which platforms to test based on current environment
        if env_info["is_macos"]:
            print("\n🍎 Running on macOS - testing macOS configuration")
            results["pipeline_tests"]["macos"] = self.run_pipeline_test("macos")
        elif env_info["is_aws_ec2"]:
            print("\n☁️  Running on AWS EC2 - testing AWS configuration")
            results["pipeline_tests"]["aws_ec2"] = self.run_pipeline_test("aws_ec2")
        elif env_info["is_linux"]:
            print("\n🐧 Running on Linux - testing AWS configuration")
            results["pipeline_tests"]["aws_ec2"] = self.run_pipeline_test("aws_ec2")
        else:
            print(f"\n❓ Unknown platform: {env_info['platform']}")

        # Generate report
        self.generate_report(results)

        return results

    def generate_report(self, results):
        """Generate test report."""
        print("\n📊 Test Report")
        print("-" * 30)

        # Dependencies summary
        dep_results = results["dependencies"]
        available_deps = sum(
            1 for dep in dep_results.values() if dep["status"] == "available"
        )
        total_deps = len(dep_results)
        print(f"Dependencies: {available_deps}/{total_deps} available")

        # Configuration summary
        config_results = results["configurations"]
        valid_configs = sum(
            1 for cfg in config_results.values() if cfg["status"] == "valid"
        )
        total_configs = len(config_results)
        print(f"Configurations: {valid_configs}/{total_configs} valid")

        # Pipeline tests summary
        pipeline_results = results["pipeline_tests"]
        successful_tests = sum(
            1 for test in pipeline_results.values() if test["status"] == "success"
        )
        total_tests = len(pipeline_results)
        print(f"Pipeline Tests: {successful_tests}/{total_tests} successful")

        # Overall status
        all_passed = (
            available_deps == total_deps
            and valid_configs == total_configs
            and successful_tests == total_tests
        )

        if all_passed:
            print("\n🎉 All tests passed! Cross-platform setup is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please check the details above.")

        # Save detailed results
        results_file = self.base_dir / "cross_platform_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

        return all_passed


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Platform SAE Testing Suite")
    parser.add_argument(
        "--base-dir", help="Base directory (default: current directory)"
    )
    parser.add_argument(
        "--platform",
        choices=["macos", "aws_ec2", "auto"],
        default="auto",
        help="Platform to test (default: auto-detect)",
    )

    args = parser.parse_args()

    # Create tester
    tester = CrossPlatformTester(args.base_dir)

    # Run tests
    if args.platform == "auto":
        results = tester.run_all_tests()
    else:
        # Run specific platform test
        print(f"🧪 Testing specific platform: {args.platform}")
        results = {
            "environment": tester.detect_environment(),
            "dependencies": tester.test_dependencies(),
            "configurations": tester.validate_configs(),
            "pipeline_tests": {args.platform: tester.run_pipeline_test(args.platform)},
        }
        tester.generate_report(results)

    # Exit with appropriate code
    pipeline_results = results.get("pipeline_tests", {})
    all_successful = all(
        test["status"] == "success" for test in pipeline_results.values()
    )
    sys.exit(0 if all_successful else 1)


if __name__ == "__main__":
    main()
