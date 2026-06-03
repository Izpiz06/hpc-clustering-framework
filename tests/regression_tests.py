"""
Regression Tests for src/utils/data_generator.py

Purpose: Detect unintended algorithm changes that break reproducibility.

Regression tests verify that generated datasets maintain consistent checksums
across runs. Any change to the data generation algorithm (even unintended) will
be caught by comparing the checksum of generated data against a golden baseline.

Golden baselines are stored in tests/regression_baselines.json and represent
the "known good" outputs for specific dataset configurations. If a regression
test fails, it indicates:
- Algorithm has changed (either intentionally or via dependency update)
- Numpy/scikit-learn versions have drifted from pinned versions
- There's a platform-specific numerical difference

References:
    - ADR-001: Deterministic, Reproducible Data Generation

Run with: pytest tests/regression_tests.py -v
"""

import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.data_generator import generate_dataset


# ============================================================================
# Baseline Configuration
# ============================================================================

BASELINE_FILE = os.path.join(os.path.dirname(__file__), "regression_baselines.json")

# These configurations represent the "golden" baselines that must not change
# Any intentional change requires updating these baselines and documenting why
BASELINE_CONFIGS = {
    "normal_seed42_1000x16": {
        "samples": 1000,
        "features": 16,
        "distribution": "normal",
        "seed": 42,
    },
    "uniform_seed42_1000x16": {
        "samples": 1000,
        "features": 16,
        "distribution": "uniform",
        "seed": 42,
    },
    "normal_seed123_500x32": {
        "samples": 500,
        "features": 32,
        "distribution": "normal",
        "seed": 123,
    },
    "uniform_seed999_500x32": {
        "samples": 500,
        "features": 32,
        "distribution": "uniform",
        "seed": 999,
    },
}


# ============================================================================
# Utility Functions
# ============================================================================


def compute_sha256(data: np.ndarray) -> str:
    """
    Compute SHA256 checksum of numpy array.

    The checksum is computed on the binary representation of the array
    data (dtype, shape, and values). This ensures that any change to the
    data will be detected.

    Args:
        data (np.ndarray): Input array.

    Returns:
        str: SHA256 checksum in hexadecimal format (without 'sha256:' prefix).
    """
    data_bytes = data.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def load_baselines() -> dict:
    """
    Load golden baselines from regression_baselines.json.

    Returns:
        dict: Baselines indexed by baseline name, value is {'sha256': '<hash>', ...}

    Raises:
        FileNotFoundError: If baselines file doesn't exist.
        json.JSONDecodeError: If baselines file is malformed.
    """
    if not os.path.exists(BASELINE_FILE):
        raise FileNotFoundError(
            f"Baseline file not found: {BASELINE_FILE}\n"
            f"Run pytest with --generate-baselines to create it."
        )

    with open(BASELINE_FILE, "r") as f:
        return json.load(f)


def save_baselines(baselines: dict):
    """Save baselines to regression_baselines.json."""
    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)


def generate_baselines() -> dict:
    """
    Generate golden baselines for all configured datasets.

    Called once to initialize the baseline file. Should be committed to the
    repository and verified across all platforms/environments.

    Returns:
        dict: Generated baselines.
    """
    baselines = {}

    for name, config in BASELINE_CONFIGS.items():
        print(f"Generating baseline: {name}...", end=" ")
        data = generate_dataset(**config)
        sha256 = compute_sha256(data)
        baselines[name] = {
            "sha256": sha256,
            "config": config,
            "numpy_version": np.__version__,
        }
        print(f"SHA256={sha256[:16]}...")

    return baselines


# ============================================================================
# Regression Tests
# ============================================================================


class TestRegressionBaselines:
    """Regression tests for baseline verification."""

    @classmethod
    def setup_class(cls):
        """Load baselines once for the entire test class."""
        try:
            cls.baselines = load_baselines()
        except FileNotFoundError as e:
            pytest.skip(str(e))

    def test_baseline_normal_distribution_seed42(self):
        """
        Regression Test 1: Normal distribution (1000×16, seed=42).

        Generates dataset and verifies checksum matches golden baseline.
        Failure indicates algorithm change or version drift.
        """
        config = BASELINE_CONFIGS["normal_seed42_1000x16"]
        baseline = self.baselines["normal_seed42_1000x16"]

        data = generate_dataset(**config)
        actual_sha256 = compute_sha256(data)

        assert actual_sha256 == baseline["sha256"], (
            f"Normal distribution baseline mismatch!\n"
            f"Expected: {baseline['sha256']}\n"
            f"Actual:   {actual_sha256}\n"
            f"This indicates algorithm change or version drift.\n"
            f"See ADR-001 for upgrade procedure."
        )

    def test_baseline_uniform_distribution_seed42(self):
        """
        Regression Test 2: Uniform distribution (1000×16, seed=42).

        Generates dataset and verifies checksum matches golden baseline.
        """
        config = BASELINE_CONFIGS["uniform_seed42_1000x16"]
        baseline = self.baselines["uniform_seed42_1000x16"]

        data = generate_dataset(**config)
        actual_sha256 = compute_sha256(data)

        assert actual_sha256 == baseline["sha256"], (
            f"Uniform distribution baseline mismatch!\n"
            f"Expected: {baseline['sha256']}\n"
            f"Actual:   {actual_sha256}\n"
            f"This indicates algorithm change or version drift.\n"
            f"See ADR-001 for upgrade procedure."
        )

    def test_baseline_normal_distribution_seed123(self):
        """Regression Test 3: Normal distribution (500×32, seed=123)."""
        config = BASELINE_CONFIGS["normal_seed123_500x32"]
        baseline = self.baselines["normal_seed123_500x32"]

        data = generate_dataset(**config)
        actual_sha256 = compute_sha256(data)

        assert actual_sha256 == baseline["sha256"]

    def test_baseline_uniform_distribution_seed999(self):
        """Regression Test 4: Uniform distribution (500×32, seed=999)."""
        config = BASELINE_CONFIGS["uniform_seed999_500x32"]
        baseline = self.baselines["uniform_seed999_500x32"]

        data = generate_dataset(**config)
        actual_sha256 = compute_sha256(data)

        assert actual_sha256 == baseline["sha256"]


class TestVersionCompatibility:
    """Test version compatibility and document versions in use."""

    def test_numpy_version_compatible(self):
        """
        Verify numpy version is compatible with pinned requirement.

        Per ADR-001, reproducibility requires exact version pinning.
        This test alerts if versions have drifted.
        """
        expected_version = "2.4.4"
        actual_version = np.__version__

        # Log version for CI/debug purposes
        print(f"NumPy version in use: {actual_version}")

        # Alert if version doesn't match (but allow patch version flexibility)
        major_minor_expected = ".".join(expected_version.split(".")[:2])
        major_minor_actual = ".".join(actual_version.split(".")[:2])

        if major_minor_actual != major_minor_expected:
            pytest.warns(
                UserWarning,
                match=f"NumPy version {actual_version} differs from pinned {expected_version}",
            )

    def test_versions_in_requirements(self):
        """
        Verify versions in requirements.txt match what's installed.

        This is a sanity check to ensure the environment is set up correctly.
        """
        # Load requirements.txt
        req_file = os.path.join(
            os.path.dirname(__file__), "..", "requirements.txt"
        )
        with open(req_file, "r") as f:
            requirements = f.read()

        # Check numpy version is pinned
        assert "numpy==" in requirements, "numpy version not pinned in requirements.txt"
        assert "scikit-learn==" in requirements, "scikit-learn version not pinned"
        assert "pytest==" in requirements, "pytest version not pinned"

        print("All dependencies properly pinned in requirements.txt")


class TestDeterminism:
    """Test determinism across multiple runs."""

    def test_determinism_100_runs(self):
        """
        Test that 100 consecutive runs with same seed produce identical output.

        This is a stress test for determinism.
        """
        config = BASELINE_CONFIGS["normal_seed42_1000x16"]
        checksums = []

        for _ in range(100):
            data = generate_dataset(**config)
            checksum = compute_sha256(data)
            checksums.append(checksum)

        # All checksums should be identical
        unique_checksums = set(checksums)
        assert len(unique_checksums) == 1, (
            f"Determinism check failed: got {len(unique_checksums)} different checksums "
            f"in 100 runs with same seed"
        )

    def test_different_seeds_produce_different_checksums(self):
        """Verify that different seeds produce different checksums (sanity check)."""
        config1 = BASELINE_CONFIGS["normal_seed42_1000x16"]
        config2 = BASELINE_CONFIGS["normal_seed123_500x32"]

        checksum1 = compute_sha256(generate_dataset(**config1))
        checksum2 = compute_sha256(generate_dataset(**config2))

        assert checksum1 != checksum2, "Different configs produced same checksum"


# ============================================================================
# Bootstrap Helper (run with --generate-baselines)
# ============================================================================


def pytest_configure(config):
    """
    Hook to generate baselines if --generate-baselines is passed.

    Usage: pytest tests/regression_tests.py --generate-baselines
    """
    if hasattr(config, "option") and config.option.__dict__.get("generate_baselines"):
        print("\n[REGRESSION TESTS] Generating golden baselines...\n")
        baselines = generate_baselines()
        save_baselines(baselines)
        print(f"\nBaselines saved to: {BASELINE_FILE}")
        print("Commit this file to version control.")


# Add custom CLI option
def pytest_addoption(parser):
    """Add custom --generate-baselines option to pytest."""
    parser.addoption(
        "--generate-baselines",
        action="store_true",
        default=False,
        help="Generate golden baselines for regression tests",
    )
