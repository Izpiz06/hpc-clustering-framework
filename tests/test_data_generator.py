"""
Unit Tests for src/utils/data_generator.py

Comprehensive test suite covering:
- Dataset generation (normal and uniform distributions)
- Reproducibility (bit-identical output with same seed)
- Determinism (different seeds produce different output)
- File I/O (CSV and NPY format serialization)
- CLI argument parsing and validation
- Edge cases (boundary conditions, error handling)

Test Coverage Target: ≥90% of data_generator.py

Run with: pytest tests/test_data_generator.py -v
Run with coverage: pytest tests/test_data_generator.py --cov=src.utils.data_generator --cov-report=html
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the functions under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.data_generator import generate_dataset, save_dataset, main


class TestGenerateDataset:
    """Test suite for generate_dataset() function."""

    def test_generate_dataset_normal_distribution(self):
        """
        Test 1: Generate dataset with normal distribution.

        Verifies:
        - Shape is correct (samples, features)
        - Dtype is float64
        - Data distribution approximates normal (mean ~0, std ~1)
        """
        samples, features = 1000, 32
        data = generate_dataset(samples=samples, features=features,
                                distribution="normal", seed=42)

        # Verify shape
        assert data.shape == (samples, features), f"Expected shape {(samples, features)}, got {data.shape}"

        # Verify dtype
        assert data.dtype == np.float64, f"Expected dtype float64, got {data.dtype}"

        # Verify distribution characteristics (within tolerance for sample size)
        mean = data.mean()
        std = data.std()
        assert abs(mean) < 0.1, f"Mean {mean} is far from 0"
        assert abs(std - 1.0) < 0.1, f"Std {std} is far from 1"

    def test_generate_dataset_uniform_distribution(self):
        """
        Test 2: Generate dataset with uniform distribution.

        Verifies:
        - Shape is correct
        - Dtype is float64
        - All values in expected range [-1, 1]
        """
        samples, features = 1000, 32
        data = generate_dataset(samples=samples, features=features,
                                distribution="uniform", seed=42)

        # Verify shape and dtype
        assert data.shape == (samples, features)
        assert data.dtype == np.float64

        # Verify range: uniform should be in [-1, 1]
        assert data.min() >= -1.0, f"Min value {data.min()} is below -1"
        assert data.max() <= 1.0, f"Max value {data.max()} is above 1"

    def test_reproducibility_same_seed(self):
        """
        Test 3: Reproducibility with same seed (CRITICAL for ADR-001).

        Verifies that identical parameters produce bit-identical output.
        This is the core requirement for scientific benchmarking.
        """
        # Generate two datasets with same parameters
        data1 = generate_dataset(samples=500, features=16,
                                 distribution="normal", seed=42)
        data2 = generate_dataset(samples=500, features=16,
                                 distribution="normal", seed=42)

        # They should be bit-identical
        assert np.allclose(data1, data2), "Same seed produced different output"
        assert np.array_equal(data1, data2), "Same seed did not produce bit-identical output"

    def test_reproducibility_different_seed(self):
        """
        Test 4: Different seeds produce different output.

        Verifies that changing the seed produces a different dataset
        (with very high probability).
        """
        data_seed42 = generate_dataset(samples=500, features=16,
                                       distribution="normal", seed=42)
        data_seed123 = generate_dataset(samples=500, features=16,
                                        distribution="normal", seed=123)

        # Should be different (essentially guaranteed for random data)
        assert not np.allclose(data_seed42, data_seed123), \
            "Different seeds produced identical output (extremely unlikely)"

    def test_invalid_samples(self):
        """Test that negative/zero samples raises ValueError."""
        with pytest.raises(ValueError, match="samples must be > 0"):
            generate_dataset(samples=0, features=10)

        with pytest.raises(ValueError, match="samples must be > 0"):
            generate_dataset(samples=-5, features=10)

    def test_invalid_features(self):
        """Test that negative/zero features raises ValueError."""
        with pytest.raises(ValueError, match="features must be > 0"):
            generate_dataset(samples=100, features=0)

        with pytest.raises(ValueError, match="features must be > 0"):
            generate_dataset(samples=100, features=-3)

    def test_invalid_distribution(self):
        """Test that invalid distribution raises ValueError."""
        with pytest.raises(ValueError, match="distribution must be 'normal' or 'uniform'"):
            generate_dataset(samples=100, features=10, distribution="poisson")

    def test_edge_case_single_sample(self):
        """Test generation with single sample (edge case)."""
        data = generate_dataset(samples=1, features=10, seed=42)
        assert data.shape == (1, 10)
        assert data.dtype == np.float64

    def test_edge_case_single_feature(self):
        """Test generation with single feature (edge case)."""
        data = generate_dataset(samples=100, features=1, seed=42)
        assert data.shape == (100, 1)
        assert data.dtype == np.float64

    def test_large_dataset(self):
        """Test generation of larger dataset (sanity check, not performance)."""
        # Generate 100K × 128 dataset (reasonable size)
        data = generate_dataset(samples=100000, features=128, seed=42)
        assert data.shape == (100000, 128)
        assert data.dtype == np.float64


class TestSaveDataset:
    """Test suite for save_dataset() function."""

    def test_csv_save(self):
        """
        Test 5: Save dataset in CSV format.

        Verifies:
        - File is created
        - File can be read back
        - Data matches original (within float precision)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data = generate_dataset(samples=100, features=10, seed=42)

            # Save as CSV
            output_path = os.path.join(tmpdir, "test_data.csv")
            saved_path = save_dataset(data, 42, output_path, output_format="csv")

            # Verify file exists
            assert os.path.exists(saved_path), f"File not created: {saved_path}"

            # Read back and verify
            loaded = np.loadtxt(saved_path, delimiter=",")
            assert loaded.shape == data.shape, f"Shape mismatch: {loaded.shape} vs {data.shape}"
            assert np.allclose(loaded, data, rtol=1e-10), "Data mismatch after CSV round-trip"

    def test_npy_save(self):
        """
        Test 6: Save dataset in NPY format.

        Verifies:
        - File is created
        - File can be read back
        - Data matches exactly (bit-identical)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data = generate_dataset(samples=100, features=10, seed=42)

            # Save as NPY
            output_path = os.path.join(tmpdir, "test_data.npy")
            saved_path = save_dataset(data, 42, output_path, output_format="npy")

            # Verify file exists
            assert os.path.exists(saved_path), f"File not created: {saved_path}"

            # Read back and verify (NPY should be bit-identical)
            loaded = np.load(saved_path)
            assert loaded.shape == data.shape
            assert np.array_equal(loaded, data), "NPY round-trip not bit-identical"
            assert loaded.dtype == data.dtype

    def test_output_directory_creation(self):
        """
        Test 7: Output directory is created if missing.

        Verifies that save_dataset() creates parent directories
        as needed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create path with non-existent nested directories
            output_path = os.path.join(tmpdir, "nested", "dirs", "test_data.csv")

            data = generate_dataset(samples=50, features=5, seed=42)
            saved_path = save_dataset(data, 42, output_path, output_format="csv")

            # Verify all directories were created
            assert os.path.exists(saved_path), f"File not saved: {saved_path}"
            assert os.path.exists(os.path.dirname(saved_path)), "Directory not created"

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        data = generate_dataset(samples=10, features=5, seed=42)
        with pytest.raises(ValueError, match="output_format must be 'csv' or 'npy'"):
            save_dataset(data, 42, "test.txt", output_format="txt")

    def test_csv_and_npy_equivalence(self):
        """
        Test that CSV and NPY formats preserve the same data.

        Verifies that both formats can be read back to produce
        equivalent results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data = generate_dataset(samples=100, features=10, seed=42)

            # Save both formats
            csv_path = save_dataset(data, 42, os.path.join(tmpdir, "test.csv"), output_format="csv")
            npy_path = save_dataset(data, 42, os.path.join(tmpdir, "test.npy"), output_format="npy")

            # Load both back
            csv_loaded = np.loadtxt(csv_path, delimiter=",")
            npy_loaded = np.load(npy_path)

            # Should be equivalent
            assert np.allclose(csv_loaded, npy_loaded, rtol=1e-10), \
                "CSV and NPY formats produced different results"


class TestCLIInterface:
    """Test suite for CLI argument parsing and main() function."""

    def test_cli_basic_usage(self):
        """
        Test CLI with basic arguments (--samples and --features).

        Verifies that main() processes arguments and generates output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate CLI call
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--samples", "100",
                    "--features", "10",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0, f"main() returned {result}, expected 0"

                # Verify output file was created
                files = os.listdir(tmpdir)
                assert len(files) > 0, "No output files created"
                assert any(f.endswith(".csv") for f in files), "No CSV file found"
            finally:
                sys.argv = original_argv

    def test_cli_npy_format(self):
        """Test CLI with --format npy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--samples", "100",
                    "--features", "10",
                    "--output-format", "npy",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0

                files = os.listdir(tmpdir)
                assert any(f.endswith(".npy") for f in files), "No NPY file found"
            finally:
                sys.argv = original_argv

    def test_cli_uniform_distribution(self):
        """Test CLI with --distribution uniform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--samples", "100",
                    "--features", "10",
                    "--distribution", "uniform",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0
            finally:
                sys.argv = original_argv

    def test_cli_custom_seed(self):
        """Test CLI with --seed argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--samples", "100",
                    "--features", "10",
                    "--seed", "12345",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0
            finally:
                sys.argv = original_argv

    def test_cli_random_state_alias(self):
        """Test CLI with --random-state (alias for --seed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--samples", "100",
                    "--features", "10",
                    "--random-state", "54321",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0
            finally:
                sys.argv = original_argv

    def test_cli_missing_required_args(self):
        """Test CLI with default arguments when none provided.

        Note: With SPRINT-002, --samples and --features now have defaults
        (when not in load mode). This test verifies defaults are used.
        """
        import sys
        original_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                sys.argv = ["data_generator.py", "--output-dir", tmpdir]
                result = main()
                assert result == 0, "Should succeed with default args"
                # Verify output file was created with defaults
                files = os.listdir(tmpdir)
                assert len(files) > 0, "No output files created with defaults"
        finally:
            sys.argv = original_argv

    def test_cli_invalid_samples(self):
        """Test CLI rejects negative samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--samples", "-100",
                    "--features", "10",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 1, "Should fail with negative samples"
            finally:
                sys.argv = original_argv


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_end_to_end_csv(self):
        """
        End-to-end test: generate, save as CSV, and verify.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate dataset
            data = generate_dataset(samples=500, features=20,
                                    distribution="normal", seed=999)

            # Save as CSV
            output_path = os.path.join(tmpdir, "data.csv")
            save_dataset(data, 999, output_path, output_format="csv")

            # Load and verify
            loaded = np.loadtxt(output_path, delimiter=",")
            assert np.allclose(loaded, data), "End-to-end CSV test failed"

    def test_end_to_end_npy(self):
        """
        End-to-end test: generate, save as NPY, and verify.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate dataset
            data = generate_dataset(samples=500, features=20,
                                    distribution="uniform", seed=999)

            # Save as NPY
            output_path = os.path.join(tmpdir, "data.npy")
            save_dataset(data, 999, output_path, output_format="npy")

            # Load and verify
            loaded = np.load(output_path)
            assert np.array_equal(loaded, data), "End-to-end NPY test failed"

    def test_reproducibility_across_formats(self):
        """
        Test that the same dataset can be saved in both formats
        and loaded to produce equivalent results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            data = generate_dataset(samples=200, features=15, seed=777)

            # Save both ways
            csv_path = os.path.join(tmpdir, "data.csv")
            npy_path = os.path.join(tmpdir, "data.npy")
            save_dataset(data, 777, csv_path, output_format="csv")
            save_dataset(data, 777, npy_path, output_format="npy")

            # Load both
            csv_loaded = np.loadtxt(csv_path, delimiter=",")
            npy_loaded = np.load(npy_path)

            # Should match
            assert np.allclose(csv_loaded, npy_loaded), \
                "CSV and NPY format round-trips differ"


# ============================================================================
# Coverage Report Target: ≥90%
# ============================================================================
# This test suite covers:
# - generate_dataset: All code paths (normal, uniform, validation, edge cases)
# - save_dataset: Both formats (CSV, NPY), directory creation, validation
# - main: CLI parsing, all argument combinations, error handling
# ============================================================================
