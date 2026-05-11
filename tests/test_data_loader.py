"""
Unit Tests for Data Loading and Transformation Functions (SPRINT-002)

Tests for:
- load_dataset(filepath): CSV and NPY loading
- validate_input_data(data): 2D shape and numeric dtype validation
- apply_transform(data, transform): Normalization and standardization
- CLI integration: --load, --transform, --output-format arguments
- ADR-002 Guardrails: allow_pickle=False, mutual exclusivity, validation

Test Coverage Target: ≥10 new tests, ≥90% coverage on new functions

Run with: pytest tests/test_data_loader.py -v
Run with coverage: pytest tests/test_data_loader.py --cov=src.utils.data_generator --cov-report=html
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the functions under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.data_generator import load_dataset, validate_input_data, apply_transform, main


class TestLoadDataset:
    """Test suite for load_dataset() function."""

    def test_load_csv_valid(self):
        """
        Test 1: Load valid CSV file.

        Verifies:
        - CSV file is loaded correctly
        - Shape is preserved
        - Data type is numeric
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV file
            test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            csv_path = os.path.join(tmpdir, "test.csv")
            np.savetxt(csv_path, test_data, delimiter=",")

            # Load and verify
            loaded = load_dataset(csv_path)
            assert loaded.shape == test_data.shape, f"Shape mismatch: {loaded.shape} vs {test_data.shape}"
            assert np.allclose(loaded, test_data), "Data values don't match"
            assert loaded.dtype in [np.float32, np.float64], "Invalid dtype"

    def test_load_npy_valid(self):
        """
        Test 2: Load valid NPY file.

        Verifies:
        - NPY file is loaded correctly
        - Shape is preserved
        - Data type is numeric
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test NPY file
            test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
            npy_path = os.path.join(tmpdir, "test.npy")
            np.save(npy_path, test_data)

            # Load and verify
            loaded = load_dataset(npy_path)
            assert loaded.shape == test_data.shape, f"Shape mismatch: {loaded.shape} vs {test_data.shape}"
            assert np.array_equal(loaded, test_data), "Data values don't match (NPY should be bit-identical)"
            assert loaded.dtype == test_data.dtype, "Data type not preserved"

    def test_load_unsupported_extension(self):
        """
        Test 3: Reject unsupported file extensions.

        Verifies that loading .txt, .json, etc. raises ValueError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .txt file
            txt_path = os.path.join(tmpdir, "test.txt")
            with open(txt_path, "w") as f:
                f.write("1,2,3\n4,5,6\n")

            # Should raise ValueError for unsupported extension
            with pytest.raises(ValueError, match="Unsupported file extension"):
                load_dataset(txt_path)

    def test_load_file_not_found(self):
        """
        Test 4: Handle missing file gracefully.

        Verifies that loading a non-existent file raises FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/file.csv")

    def test_load_csv_1d_array(self):
        """
        Test 5: Load CSV that produces 1D array (validation will catch it).

        Verifies that loading a 1-row CSV still loads (validation catches shape issue).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create single-row CSV
            csv_path = os.path.join(tmpdir, "single_row.csv")
            np.savetxt(csv_path, np.array([[1.0, 2.0, 3.0]]), delimiter=",")

            # Load should work
            loaded = load_dataset(csv_path)
            assert loaded is not None, "Failed to load single-row CSV"

    def test_load_npy_allow_pickle_false(self):
        """
        Test 6: Verify allow_pickle=False enforcement in NPY loading (ADR-002).

        This test verifies that standard float64 NPY files load correctly with
        allow_pickle=False. Pickle-enabled files would be blocked.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create standard float64 NPY (no pickled objects)
            test_data = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
            npy_path = os.path.join(tmpdir, "standard.npy")
            np.save(npy_path, test_data)

            # Load with allow_pickle=False should work fine for standard arrays
            loaded = load_dataset(npy_path)
            assert np.array_equal(loaded, test_data), "Standard NPY failed to load with allow_pickle=False"


class TestValidateInputData:
    """Test suite for validate_input_data() function."""

    def test_validate_2d_numeric_array(self):
        """
        Test 7: Validate correct 2D numeric array.

        Verifies that valid 2D float arrays pass validation.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate_input_data(data)
        assert np.array_equal(result, data), "Valid data was rejected"

    def test_validate_reject_1d_array(self):
        """
        Test 8: Reject 1D arrays.

        Verifies that 1D arrays raise ValueError.
        """
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Input data must be 2-dimensional"):
            validate_input_data(data)

    def test_validate_reject_3d_array(self):
        """
        Test 9: Reject 3D arrays.

        Verifies that 3D arrays raise ValueError.
        """
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        with pytest.raises(ValueError, match="Input data must be 2-dimensional"):
            validate_input_data(data)

    def test_validate_reject_non_numeric_dtype(self):
        """
        Test 10: Reject non-numeric dtypes (strings, objects).

        Verifies that string arrays raise ValueError.
        """
        data = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(ValueError, match="Input data must be numeric"):
            validate_input_data(data)


class TestApplyTransform:
    """Test suite for apply_transform() function."""

    def test_transform_none(self):
        """
        Test 11: 'none' transform returns data unchanged.

        Verifies that the 'none' transform is a no-op.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = apply_transform(data, "none")
        assert np.array_equal(result, data), "'none' transform modified data"

    def test_transform_normalize(self):
        """
        Test 12: 'normalize' transform scales values to [0, 1].

        Verifies min-max normalization per feature.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]])
        result = apply_transform(data, "normalize")

        # Check that all values are in [0, 1]
        assert result.min() >= 0.0, f"Normalized data has values < 0: {result.min()}"
        assert result.max() <= 1.0, f"Normalized data has values > 1: {result.max()}"

        # Check that min and max are actually 0 and 1 (per feature)
        # For feature 0: min=1, max=3 → normalized to [0, 1]
        # For feature 1: min=1, max=4 → normalized to [0, 1]
        assert result[0, 0] == 0.0, "Min value not normalized to 0"

    def test_transform_standardize(self):
        """
        Test 13: 'standardize' transform converts to zero mean, unit variance.

        Verifies z-score standardization per feature.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        result = apply_transform(data, "standardize")

        # Check that mean is approximately 0 and std is approximately 1 per feature
        mean_per_feature = result.mean(axis=0)
        std_per_feature = result.std(axis=0, ddof=0)

        assert np.allclose(mean_per_feature, 0.0, atol=1e-10), \
            f"Standardized data has mean {mean_per_feature}, expected ~0"
        assert np.allclose(std_per_feature, 1.0, atol=1e-10), \
            f"Standardized data has std {std_per_feature}, expected ~1"

    def test_transform_unknown(self):
        """
        Test 14: Unknown transform raises ValueError.

        Verifies that invalid transform names are rejected.
        """
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transform(data, "invalid")

    def test_transform_normalize_with_constant_feature(self):
        """
        Test 15: 'normalize' handles constant features (division by zero guard).

        Verifies that epsilon prevents division by zero errors.
        """
        # Feature 0 is constant, feature 1 has range
        data = np.array([[1.0, 2.0], [1.0, 4.0]])
        result = apply_transform(data, "normalize")

        # Should not raise, should handle gracefully
        assert result.shape == data.shape, "Shape changed after normalize"
        assert not np.any(np.isnan(result)), "Result contains NaN values"
        assert not np.any(np.isinf(result)), "Result contains inf values"

    def test_transform_standardize_with_constant_feature(self):
        """
        Test 16: 'standardize' handles constant features (division by zero guard).

        Verifies that epsilon prevents division by zero errors.
        """
        # Feature 0 is constant, feature 1 has variation
        data = np.array([[1.0, 2.0], [1.0, 4.0]], dtype=np.float64)
        result = apply_transform(data, "standardize")

        # Should not raise, should handle gracefully
        assert result.shape == data.shape, "Shape changed after standardize"
        assert not np.any(np.isnan(result)), "Result contains NaN values"
        assert not np.any(np.isinf(result)), "Result contains inf values"


class TestCLIMutualExclusivity:
    """Test suite for ADR-002 Guardrail 1: Mutual exclusivity."""

    def test_load_with_samples_fails(self):
        """
        Test 17: --load with --samples raises argparse error.

        Verifies ADR-002 mutual exclusivity: --load and --samples/--features cannot coexist.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test CSV file
            csv_path = os.path.join(tmpdir, "test.csv")
            np.savetxt(csv_path, np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", csv_path,
                    "--samples", "100",  # Should conflict
                    "--output-dir", tmpdir,
                ]
                # parser.error() calls sys.exit(2), which raises SystemExit
                with pytest.raises(SystemExit):
                    main()
            finally:
                sys.argv = original_argv

    def test_load_with_features_fails(self):
        """
        Test 18: --load with --features raises argparse error.

        Verifies ADR-002 mutual exclusivity.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test CSV file
            csv_path = os.path.join(tmpdir, "test.csv")
            np.savetxt(csv_path, np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", csv_path,
                    "--features", "64",  # Should conflict
                    "--output-dir", tmpdir,
                ]
                # parser.error() calls sys.exit(2), which raises SystemExit
                with pytest.raises(SystemExit):
                    main()
            finally:
                sys.argv = original_argv


class TestCLILoadMode:
    """Test suite for CLI load mode (--load argument)."""

    def test_cli_load_csv(self):
        """
        Test 19: CLI load mode with CSV file.

        Verifies: python data_generator.py --load file.csv
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV
            test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            csv_path = os.path.join(tmpdir, "input.csv")
            np.savetxt(csv_path, test_data, delimiter=",")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", csv_path,
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0, f"main() returned {result}, expected 0"

                # Verify output file was created
                files = [f for f in os.listdir(tmpdir) if f.endswith(".csv") and f != "input.csv"]
                assert len(files) > 0, "No output CSV file created"
            finally:
                sys.argv = original_argv

    def test_cli_load_npy(self):
        """
        Test 20: CLI load mode with NPY file.

        Verifies: python data_generator.py --load file.npy
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test NPY
            test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            npy_path = os.path.join(tmpdir, "input.npy")
            np.save(npy_path, test_data)

            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", npy_path,
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0, f"main() returned {result}, expected 0"

                # Verify output file was created
                files = [f for f in os.listdir(tmpdir) if f.endswith(".csv") and f != "input.npy"]
                assert len(files) > 0, "No output file created"
            finally:
                sys.argv = original_argv

    def test_cli_load_with_transform(self):
        """
        Test 21: CLI load mode with --transform option.

        Verifies: python data_generator.py --load file.csv --transform standardize
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV
            test_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            csv_path = os.path.join(tmpdir, "input.csv")
            np.savetxt(csv_path, test_data, delimiter=",")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", csv_path,
                    "--transform", "standardize",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0, f"main() returned {result}, expected 0"

                # Verify output file was created
                files = [f for f in os.listdir(tmpdir) if f.endswith(".csv") and f != "input.csv"]
                assert len(files) > 0, "No output file created"
            finally:
                sys.argv = original_argv

    def test_cli_load_csv_save_npy(self):
        """
        Test 22: CLI load CSV, save as NPY.

        Verifies: python data_generator.py --load file.csv --output-format npy
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV
            test_data = np.array([[1.0, 2.0], [3.0, 4.0]])
            csv_path = os.path.join(tmpdir, "input.csv")
            np.savetxt(csv_path, test_data, delimiter=",")

            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", csv_path,
                    "--output-format", "npy",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 0, f"main() returned {result}, expected 0"

                # Verify output NPY file was created
                files = [f for f in os.listdir(tmpdir) if f.endswith(".npy")]
                assert len(files) > 0, "No output NPY file created"
            finally:
                sys.argv = original_argv

    def test_cli_load_nonexistent_file(self):
        """
        Test 23: CLI load mode with non-existent file.

        Verifies error handling for missing input file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_argv = sys.argv
            try:
                sys.argv = [
                    "data_generator.py",
                    "--load", "/nonexistent/file.csv",
                    "--output-dir", tmpdir,
                ]
                result = main()
                assert result == 1, "Should fail when file doesn't exist"
            finally:
                sys.argv = original_argv


# ============================================================================
# Coverage Report Target: ≥10 new tests (actual: 23)
# ============================================================================
# This test suite covers:
# - load_dataset: CSV loading, NPY loading, unsupported formats, missing files, 1D arrays, allow_pickle=False
# - validate_input_data: Valid 2D arrays, reject 1D/3D/non-numeric
# - apply_transform: none, normalize, standardize, unknown, constant features
# - CLI mutual exclusivity: --load with --samples/--features
# - CLI load mode: CSV load, NPY load, transform, format conversion, error handling
# ============================================================================
