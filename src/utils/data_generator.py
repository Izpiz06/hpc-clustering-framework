#!/usr/bin/env python3
"""
Synthetic Dataset Generator for HPC Clustering Framework

Generates deterministic, reproducible synthetic datasets for clustering algorithm
benchmarking. Provides CLI interface with configurable sample count, feature count,
distribution type, and random seed. Outputs datasets in .npy (numpy binary) and
.csv (comma-separated values) formats.

Key Features:
- Deterministic reproducibility: Identical parameters produce bit-identical output
- Two distribution types: normal (mean=0, std=1) and uniform (-1 to 1)
- Multiple output formats: .npy (fast, preserves precision) and .csv (human-readable)
- Advanced CLI options: Custom seed, output directory, format selection
- Hard reproducibility: Uses numpy.random.RandomState(seed) with pinned numpy version

Usage Examples:
    # Basic: 10K samples, 32 features, default seed, CSV output
    python src/utils/data_generator.py --samples 10000 --features 32

    # Advanced: 5K samples, 64 features, uniform dist, custom seed, NPY format
    python src/utils/data_generator.py --samples 5000 --features 64 \\
        --distribution uniform --seed 12345 --format npy

    # Custom output directory
    python src/utils/data_generator.py --samples 1000 --features 16 \\
        --output-dir /tmp/datasets

References:
    - ADR-001: Deterministic, Reproducible Data Generation
      https://github.com/Izpiz06/hpc-clustering-framework/issues/3
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Default values for mutual exclusivity check
DEFAULT_SAMPLES = 1000
DEFAULT_FEATURES = 32


def load_dataset(filepath: str) -> np.ndarray:
    """
    Load a dataset from a CSV or NPY file.

    Auto-detects file format by extension:
    - '.csv' → Uses numpy.loadtxt with comma delimiter
    - '.npy' → Uses numpy.load with allow_pickle=False (security)
    - Other extensions → Raises ValueError

    Args:
        filepath (str): Path to the CSV or NPY file to load.

    Returns:
        np.ndarray: Loaded dataset as 2D ndarray (validation must follow).

    Raises:
        ValueError: If file extension is not '.csv' or '.npy'.
        FileNotFoundError: If file does not exist.
        IOError: If file cannot be read.

    Examples:
        >>> data = load_dataset('data.csv')
        >>> data.shape
        (1000, 32)

        >>> data_npy = load_dataset('data.npy')
        >>> data_npy.dtype
        dtype('float64')
    """
    # Get file extension
    file_ext = os.path.splitext(filepath)[1].lower()

    try:
        if file_ext == ".csv":
            # Load CSV using numpy.loadtxt with comma delimiter
            data = np.loadtxt(filepath, delimiter=",")
        elif file_ext == ".npy":
            # Load NPY with allow_pickle=False (ADR-002 security requirement)
            data = np.load(filepath, allow_pickle=False)
        else:
            raise ValueError(f"Unsupported file extension '{file_ext}'. Must be '.csv' or '.npy'.")

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except IOError as e:
        raise IOError(f"Failed to read file {filepath}: {e}") from e


def validate_input_data(data: np.ndarray) -> np.ndarray:
    """
    Validate that input data is 2-dimensional and numeric.

    ADR-002 Guardrail: After loading, verify 2D shape and numeric dtype.

    Args:
        data (np.ndarray): Data to validate.

    Returns:
        np.ndarray: The same data if valid.

    Raises:
        ValueError: If data is not 2-dimensional or not numeric.

    Examples:
        >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> validate_input_data(data)
        array([[1., 2.],
               [3., 4.]])

        >>> bad_data = np.array([1.0, 2.0])
        >>> validate_input_data(bad_data)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: Input data must be 2-dimensional, got 1
    """
    # Check dimensionality
    if data.ndim != 2:
        raise ValueError(f"Input data must be 2-dimensional, got {data.ndim}")

    # Check numeric dtype
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"Input data must be numeric, got dtype '{data.dtype}'")

    return data


def apply_transform(data: np.ndarray, transform: str) -> np.ndarray:
    """
    Apply a transformation to the dataset.

    Supports three transformations:
    - 'none': Return data unchanged
    - 'normalize': Min-max normalization to [0, 1] per feature (axis=0)
    - 'standardize': Z-score standardization per feature (axis=0)

    Args:
        data (np.ndarray): 2D input data to transform.
        transform (str): Name of transformation to apply.

    Returns:
        np.ndarray: Transformed data (same shape as input).

    Raises:
        ValueError: If transform is not recognized.

    Examples:
        >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> normalize = apply_transform(data, 'normalize')
        >>> normalize[0, 0]  # Min value normalized to 0
        0.0

        >>> standardize = apply_transform(data, 'standardize')
        >>> abs(standardize.mean()) < 1e-10  # Mean ~0 after standardize
        True
    """
    if transform == "none":
        return data
    elif transform == "normalize":
        # Min-max normalization: (x - min) / (max - min + eps)
        # Applied per feature (axis=0)
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        data_range = data_max - data_min
        # Add small epsilon to avoid division by zero
        return (data - data_min) / (data_range + 1e-8)
    elif transform == "standardize":
        # Z-score standardization: (x - mean) / (std + eps)
        # Applied per feature (axis=0, ddof=0)
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0, ddof=0)
        return (data - data_mean) / (data_std + 1e-8)
    else:
        raise ValueError(f"Unknown transform '{transform}'. Must be 'none', 'normalize', or 'standardize'.")


def generate_dataset(samples: int, features: int, distribution: str = "normal", seed: int = 42) -> np.ndarray:
    """
    Generate a synthetic dataset with the specified parameters.

    Uses numpy.random.RandomState(seed) for explicit, reproducible random number generation.
    This ensures bit-identical output across runs with identical parameters, which is
    essential for scientific benchmarking (see ADR-001).

    Args:
        samples (int): Number of samples (rows) to generate. Must be > 0.
        features (int): Number of features (columns) per sample. Must be > 0.
        distribution (str): Type of distribution: 'normal' or 'uniform'.
            - 'normal': Gaussian distribution with mean=0, std=1
            - 'uniform': Uniform distribution in range [-1, 1]
        seed (int): Random seed for reproducibility. Default: 42.
            Use the same seed to generate identical datasets.

    Returns:
        np.ndarray: Generated dataset with shape (samples, features) and dtype float64.

    Raises:
        ValueError: If samples or features are <= 0, or distribution is invalid.

    Examples:
        >>> data = generate_dataset(samples=100, features=10, seed=42)
        >>> data.shape
        (100, 10)
        >>> data.dtype
        dtype('float64')

        >>> # Generate reproducible dataset
        >>> data1 = generate_dataset(100, 10, seed=42)
        >>> data2 = generate_dataset(100, 10, seed=42)
        >>> np.allclose(data1, data2)
        True

        >>> # Different distribution
        >>> data_uniform = generate_dataset(100, 10, distribution='uniform', seed=42)
        >>> data_uniform.min() >= -1 and data_uniform.max() <= 1
        True
    """
    # Input validation
    if samples <= 0:
        raise ValueError(f"samples must be > 0, got {samples}")
    if features <= 0:
        raise ValueError(f"features must be > 0, got {features}")
    if distribution not in ("normal", "uniform"):
        raise ValueError(f"distribution must be 'normal' or 'uniform', got '{distribution}'")

    # Create RandomState with explicit seed for reproducibility
    rng = np.random.RandomState(seed)

    # Generate dataset based on distribution type
    if distribution == "normal":
        # Standard normal distribution: mean=0, std=1
        data = rng.normal(loc=0.0, scale=1.0, size=(samples, features))
    else:  # distribution == "uniform"
        # Uniform distribution in range [-1, 1]
        data = rng.uniform(low=-1.0, high=1.0, size=(samples, features))

    # Ensure float64 dtype for consistency across platforms and numpy versions
    return data.astype(np.float64)


def save_dataset(
    data: np.ndarray, seed: int, output_path: str, output_format: str = "csv"
) -> str:
    """
    Save a dataset to disk in the specified format.

    Supports two formats:
    - 'csv': Comma-separated values (human-readable, can be opened in Excel/spreadsheets)
    - 'npy': NumPy binary format (fast, preserves full precision, smaller file size)

    Automatically creates output directory if it doesn't exist.

    Args:
        data (np.ndarray): Dataset to save. Should be 2D array with dtype float64.
        seed (int): Random seed (used to construct output path if needed).
        output_path (str): Full path to output file (including filename and extension).
            Example: 'data/20260511-154500_10000_32_normal_seed42.csv'
        output_format (str): Output format: 'csv' or 'npy'. Default: 'csv'.

    Returns:
        str: Full path to the saved file.

    Raises:
        ValueError: If format is invalid.
        IOError: If file cannot be written (permission error, disk full, etc.).

    Examples:
        >>> data = generate_dataset(100, 10, seed=42)
        >>> path = save_dataset(data, 42, 'output.csv', output_format='csv')
        >>> import os
        >>> os.path.exists(path)
        True

        >>> # Save as NPY (binary format)
        >>> path = save_dataset(data, 42, 'output.npy', output_format='npy')
        >>> loaded = np.load(path)
        >>> np.allclose(data, loaded)
        True
    """
    # Validate format
    if output_format not in ("csv", "npy"):
        raise ValueError(f"output_format must be 'csv' or 'npy', got '{output_format}'")

    # Create output directory if missing
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save in the specified format
    try:
        if output_format == "csv":
            # Save as CSV with comma delimiter, no header, no index
            # Float precision: use default (sufficient for 64-bit float)
            np.savetxt(output_path, data, delimiter=",", fmt="%.16g")
        else:  # output_format == "npy"
            # Save as NumPy binary format (preserves dtype exactly)
            np.save(output_path, data)
    except IOError as e:
        raise IOError(f"Failed to save dataset to {output_path}: {e}") from e

    return output_path


def main():
    """
    CLI entry point for the synthetic dataset generator and data loader.

    Parses command-line arguments and dispatches to either:
    - Generate mode: Create synthetic datasets
    - Load mode: Load external CSV/NPY files with optional transforms

    Supported arguments:
        --load STR: Path to CSV or NPY file to load (mutually exclusive with --samples/--features)
        --transform {none|normalize|standardize}: Transform to apply after load (default: none)
        --output-format {csv,npy}: Output format (default: csv)
        --samples INT: Number of samples to generate (for generate mode)
        --features INT: Number of features per sample (for generate mode)
        --distribution {normal,uniform}: Data distribution (default: normal)
        --seed INT: Random seed for reproducibility (default: 42)
        --random-state INT: Alias for --seed (numpy convention)
        --output-dir PATH: Output directory (default: data/)

    Exit Codes:
        0: Success
        1: Argument parsing error or runtime error

    Example:
        # Generate mode (unchanged from SPRINT-001)
        $ python src/utils/data_generator.py --samples 10000 --features 32

        # Load mode (new in SPRINT-002)
        $ python src/utils/data_generator.py --load data.csv --transform standardize
        $ python src/utils/data_generator.py --load data.npy --output-format csv
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets or load external data for clustering benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GENERATE MODE (create synthetic data)
  # Basic: 10K samples, 32 features, CSV, default seed (42)
  python src/utils/data_generator.py --samples 10000 --features 32

  # Advanced: 5K samples, 64 features, uniform distribution, custom seed
  python src/utils/data_generator.py --samples 5000 --features 64 \\
    --distribution uniform --seed 12345 --output-format npy

  # LOAD MODE (load external data)
  # Load CSV, standardize, save as NPY
  python src/utils/data_generator.py --load my_data.csv --transform standardize --output-format npy

  # Load NPY, save as CSV
  python src/utils/data_generator.py --load my_data.npy --output-format csv

  # Load CSV, no transform
  python src/utils/data_generator.py --load my_data.csv

For reproducibility: Use the same --seed to generate identical datasets.
See docs/data_generator_how_to.md for detailed usage guide.
        """,
    )

    # Load mode argument (new in SPRINT-002)
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to CSV or NPY file to load. Mutually exclusive with --samples/--features.",
    )

    # Transform argument (new in SPRINT-002)
    parser.add_argument(
        "--transform",
        type=str,
        choices=["none", "normalize", "standardize"],
        default="none",
        help="Transform to apply after loading data. Options: 'none' (unchanged), "
        "'normalize' (min-max to [0,1]), 'standardize' (z-score). Default: %(default)s",
    )

    # Output format argument (shared between generate and load modes)
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "npy"],
        help="Output format. 'csv' (default) is human-readable; 'npy' is binary "
        "(faster, smaller file). Default: %(default)s",
    )

    # Generate mode arguments (required only if not in load mode)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of samples (rows) to generate. Must be > 0. Default: %(default)s",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=DEFAULT_FEATURES,
        help="Number of features (columns) per sample. Must be > 0. Default: %(default)s",
    )

    # Optional arguments with sensible defaults
    parser.add_argument(
        "--distribution",
        type=str,
        default="normal",
        choices=["normal", "uniform"],
        help="Data distribution type. 'normal' (default) is Gaussian (mean=0, std=1); "
        "'uniform' is [-1, 1]. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Use same seed to generate identical "
        "datasets. Default: %(default)s",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Alias for --seed (numpy convention). If provided, overrides --seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for generated or processed datasets. Created if missing. "
        "Default: %(default)s",
    )

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit() on error; catch and re-exit with code 1
        return 1

    # Resolve seed: --random-state takes precedence if provided
    seed = args.random_state if args.random_state is not None else args.seed

    # ADR-002 Guardrail 1: Mutual exclusivity check
    if args.load:
        # Load mode
        if args.samples != DEFAULT_SAMPLES or args.features != DEFAULT_FEATURES:
            parser.error(
                "--load and --samples/--features are mutually exclusive. "
                "When loading external data, do not specify --samples or --features."
            )

        try:
            # Load dataset from file
            print(f"Loading data from {args.load}...", file=sys.stderr)
            data = load_dataset(args.load)

            # Validate input data (ADR-002 Guardrail 3)
            data = validate_input_data(data)

            # Apply transform
            if args.transform != "none":
                print(f"Applying {args.transform} transform...", file=sys.stderr)
            data = apply_transform(data, args.transform)

            # Construct output filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            input_basename = os.path.splitext(os.path.basename(args.load))[0]
            if args.transform != "none":
                filename = f"{timestamp}_{input_basename}_{args.transform}.{args.output_format}"
            else:
                filename = f"{timestamp}_{input_basename}.{args.output_format}"
            output_path = os.path.join(args.output_dir, filename)

            # Save processed dataset
            saved_path = save_dataset(data, seed, output_path, output_format=args.output_format)

            # Print success summary
            file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
            print(f"\nData loaded and processed successfully!", file=sys.stderr)
            print(f"  Source: {args.load}", file=sys.stderr)
            print(f"  Location: {saved_path}", file=sys.stderr)
            print(f"  Shape: {data.shape}", file=sys.stderr)
            print(f"  Dtype: {data.dtype}", file=sys.stderr)
            print(f"  Transform: {args.transform}", file=sys.stderr)
            print(f"  File size: {file_size_mb:.2f} MB", file=sys.stderr)
            print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]", file=sys.stderr)

            return 0

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except IOError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 1

    else:
        # Generate mode (SPRINT-001 original behavior)
        # Validate that samples and features are positive
        if args.samples <= 0:
            print(f"Error: --samples must be > 0, got {args.samples}", file=sys.stderr)
            return 1
        if args.features <= 0:
            print(f"Error: --features must be > 0, got {args.features}", file=sys.stderr)
            return 1

        try:
            # Generate dataset
            print(
                f"Generating {args.samples:,} samples × {args.features} features "
                f"({args.distribution} distribution, seed={seed})...",
                file=sys.stderr,
            )
            data = generate_dataset(
                samples=args.samples,
                features=args.features,
                distribution=args.distribution,
                seed=seed,
            )

            # Construct output filename with timestamp and parameters
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = (
                f"{timestamp}_{args.samples}_{args.features}_{args.distribution}_"
                f"seed{seed}.{args.output_format}"
            )
            output_path = os.path.join(args.output_dir, filename)

            # Save dataset
            saved_path = save_dataset(data, seed, output_path, output_format=args.output_format)

            # Print success summary
            file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
            print(f"\nDataset generated successfully!", file=sys.stderr)
            print(f"  Location: {saved_path}", file=sys.stderr)
            print(f"  Shape: {data.shape}", file=sys.stderr)
            print(f"  Dtype: {data.dtype}", file=sys.stderr)
            print(f"  File size: {file_size_mb:.2f} MB", file=sys.stderr)
            print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]", file=sys.stderr)

            return 0

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except IOError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
