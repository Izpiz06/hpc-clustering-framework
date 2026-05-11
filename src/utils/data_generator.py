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
    data: np.ndarray, output_path: str, format: str = "csv"
) -> str:
    """
    Save a dataset to disk in the specified format.

    Supports two formats:
    - 'csv': Comma-separated values (human-readable, can be opened in Excel/spreadsheets)
    - 'npy': NumPy binary format (fast, preserves full precision, smaller file size)

    Automatically creates output directory if it doesn't exist.

    Args:
        data (np.ndarray): Dataset to save. Should be 2D array with dtype float64.
        output_path (str): Full path to output file (including filename and extension).
            Example: 'data/20260511-154500_10000_32_normal_seed42.csv'
        format (str): Output format: 'csv' or 'npy'. Default: 'csv'.

    Returns:
        str: Full path to the saved file.

    Raises:
        ValueError: If format is invalid.
        IOError: If file cannot be written (permission error, disk full, etc.).

    Examples:
        >>> data = generate_dataset(100, 10, seed=42)
        >>> path = save_dataset(data, 'output.csv', format='csv')
        >>> import os
        >>> os.path.exists(path)
        True

        >>> # Save as NPY (binary format)
        >>> path = save_dataset(data, 'output.npy', format='npy')
        >>> loaded = np.load(path)
        >>> np.allclose(data, loaded)
        True
    """
    # Validate format
    if format not in ("csv", "npy"):
        raise ValueError(f"format must be 'csv' or 'npy', got '{format}'")

    # Create output directory if missing
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save in the specified format
    try:
        if format == "csv":
            # Save as CSV with comma delimiter, no header, no index
            # Float precision: use default (sufficient for 64-bit float)
            np.savetxt(output_path, data, delimiter=",", fmt="%.16g")
        else:  # format == "npy"
            # Save as NumPy binary format (preserves dtype exactly)
            np.save(output_path, data)
    except IOError as e:
        raise IOError(f"Failed to save dataset to {output_path}: {e}") from e

    return output_path


def main():
    """
    CLI entry point for the synthetic dataset generator.

    Parses command-line arguments, generates a dataset, saves it to disk,
    and prints a summary of the operation.

    Supported arguments:
        --samples INT (required): Number of samples to generate
        --features INT (required): Number of features per sample
        --format {csv,npy}: Output format (default: csv)
        --distribution {normal,uniform}: Data distribution (default: normal)
        --seed INT: Random seed for reproducibility (default: 42)
        --random-state INT: Alias for --seed (numpy convention)
        --output-dir PATH: Output directory (default: data/)

    Exit Codes:
        0: Success
        1: Argument parsing error or runtime error

    Example:
        $ python src/utils/data_generator.py --samples 10000 --features 32
        $ python src/utils/data_generator.py --samples 5000 --features 64 \\
            --format npy --distribution uniform --seed 12345
    """
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic datasets for clustering benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: 10K samples, 32 features, CSV, default seed (42)
  python src/utils/data_generator.py --samples 10000 --features 32

  # Advanced: 5K samples, 64 features, uniform distribution, custom seed
  python src/utils/data_generator.py --samples 5000 --features 64 \\
    --distribution uniform --seed 12345 --format npy

  # Custom output directory
  python src/utils/data_generator.py --samples 1000 --features 16 \\
    --output-dir /tmp/datasets

For reproducibility: Use the same --seed to generate identical datasets.
See docs/data_generator_how_to.md for detailed usage guide.
        """,
    )

    # Required arguments
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="Number of samples (rows) to generate. Must be > 0.",
    )
    parser.add_argument(
        "--features",
        type=int,
        required=True,
        help="Number of features (columns) per sample. Must be > 0.",
    )

    # Optional arguments with sensible defaults
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "npy"],
        help="Output format. 'csv' (default) is human-readable; 'npy' is binary "
        "(faster, smaller file). Default: %(default)s",
    )
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
        help="Output directory for generated datasets. Created if missing. "
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

    # Validate arguments
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
            f"seed{seed}.{args.format}"
        )
        output_path = os.path.join(args.output_dir, filename)

        # Save dataset
        saved_path = save_dataset(data, output_path, format=args.format)

        # Print success summary
        file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
        print(
            f"\nDataset saved successfully!",
            file=sys.stderr,
        )
        print(
            f"  Location: {saved_path}",
            file=sys.stderr,
        )
        print(
            f"  Shape: {data.shape}",
            file=sys.stderr,
        )
        print(
            f"  Dtype: {data.dtype}",
            file=sys.stderr,
        )
        print(
            f"  File size: {file_size_mb:.2f} MB",
            file=sys.stderr,
        )
        print(
            f"  Data range: [{data.min():.6f}, {data.max():.6f}]",
            file=sys.stderr,
        )

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
