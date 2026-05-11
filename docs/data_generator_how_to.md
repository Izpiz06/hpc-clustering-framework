# Synthetic Dataset Generator — How-To Guide

A comprehensive guide to using the data generator utility for creating reproducible synthetic datasets for clustering algorithm benchmarking.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Reference](#cli-reference)
3. [Examples](#examples)
4. [Reproducibility](#reproducibility)
5. [Performance](#performance)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)
8. [FAQ](#faq)

---

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Izpiz06/hpc-clustering-framework.git
   cd hpc-clustering-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python src/utils/data_generator.py --help
   ```

### Basic Usage

Generate a simple synthetic dataset (10,000 samples, 32 features):

```bash
python src/utils/data_generator.py --samples 10000 --features 32
```

Output:
```
Generating 10,000 samples × 32 features (normal distribution, seed=42)...

Dataset saved successfully!
  Location: data/20260511-154500_10000_32_normal_seed42.csv
  Shape: (10000, 32)
  Dtype: float64
  File size: 5.34 MB
  Data range: [-4.127356, 4.891234]
```

The dataset is saved as a CSV file in the `data/` directory by default.

---

## CLI Reference

### Command Syntax

```bash
python src/utils/data_generator.py \
  --samples NUM \
  --features NUM \
  [--format FORMAT] \
  [--distribution DIST] \
  [--seed SEED] \
  [--random-state SEED] \
  [--output-dir DIR]
```

### Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--samples` | int | — | **Yes** | Number of samples (rows) to generate. Must be > 0. |
| `--features` | int | — | **Yes** | Number of features (columns) per sample. Must be > 0. |
| `--format` | str | `csv` | No | Output format: `csv` (human-readable) or `npy` (binary, faster). |
| `--distribution` | str | `normal` | No | Data distribution: `normal` (Gaussian) or `uniform` ([-1, 1]). |
| `--seed` | int | `42` | No | Random seed for reproducibility. Use same seed → identical output. |
| `--random-state` | int | None | No | Alias for `--seed` (numpy convention). Overrides `--seed` if provided. |
| `--output-dir` | str | `data/` | No | Output directory. Created if missing. |

### Output File Naming

Generated files follow the pattern:

```
<timestamp>_<samples>_<features>_<distribution>_seed<seed>.<format>
```

Examples:
- `20260511-154500_10000_32_normal_seed42.csv`
- `20260511-154500_5000_64_uniform_seed12345.npy`

---

## Examples

### Example 1: Basic CSV Generation (Default)

Generate 10K × 32 dataset with default parameters:

```bash
python src/utils/data_generator.py --samples 10000 --features 32
```

- Format: CSV (human-readable)
- Distribution: Normal (Gaussian, mean=0, std=1)
- Seed: 42 (default)
- Output: `data/20260511-XXXXXX_10000_32_normal_seed42.csv`

### Example 2: NPY Binary Format (Faster)

Generate 5K × 64 dataset in fast binary format:

```bash
python src/utils/data_generator.py \
  --samples 5000 \
  --features 64 \
  --format npy
```

Binary format is:
- **Faster:** No serialization overhead
- **Smaller:** More compact on disk (~40% smaller than CSV)
- **Precise:** Preserves full float64 precision

### Example 3: Uniform Distribution

Generate uniformly distributed data in range [-1, 1]:

```bash
python src/utils/data_generator.py \
  --samples 1000 \
  --features 32 \
  --distribution uniform
```

Use uniform distribution for:
- Testing algorithms' behavior on different data patterns
- Clustering with varying cluster separation
- Benchmarks that require non-Gaussian input

### Example 4: Custom Seed for Reproducibility

Generate dataset with custom seed:

```bash
python src/utils/data_generator.py \
  --samples 2000 \
  --features 16 \
  --seed 999
```

Always use the same `--seed` to regenerate **bit-identical** datasets. This is critical for:
- Reproducible research
- Comparing algorithms fairly
- Documentation and reference datasets

### Example 5: Custom Output Directory

Save datasets to a specific directory:

```bash
python src/utils/data_generator.py \
  --samples 5000 \
  --features 32 \
  --output-dir /tmp/benchmark_data
```

Directory is created automatically if missing.

### Example 6: Combining Multiple Options

Advanced example with all custom parameters:

```bash
python src/utils/data_generator.py \
  --samples 100000 \
  --features 128 \
  --format npy \
  --distribution uniform \
  --seed 54321 \
  --output-dir ./datasets/benchmark_v2
```

This generates:
- 100K samples, 128 features
- Uniform distribution
- Binary NPY format (fast loading)
- Seed 54321 (reproducible)
- Saved to `./datasets/benchmark_v2/`

---

## Reproducibility

### Why Reproducibility Matters

In machine learning research, reproducibility is essential for:
- **Fair comparison:** Two algorithms must run on identical data
- **Publication:** Reviewers must be able to verify results
- **Debugging:** Researchers must isolate algorithm issues from data variance
- **Collaboration:** Teams need to share exact datasets

### How to Ensure Bit-Identical Output

The data generator is **deterministic** — identical parameters produce **bit-identical** output:

```bash
# Generate dataset A
python src/utils/data_generator.py --samples 1000 --features 32 --seed 42

# Generate dataset B with exact same parameters
python src/utils/data_generator.py --samples 1000 --features 32 --seed 42

# Datasets A and B are bit-identical (byte-for-byte identical)
```

### Critical: Version Pinning

Reproducibility requires **exact version pinning**. See `requirements.txt`:

```
numpy==2.4.4
scikit-learn==1.8.0
pytest==8.3.4
```

Do **NOT** upgrade versions without:
1. Verifying datasets still reproduce identically
2. Generating new golden baselines
3. Documenting the upgrade in ADR-001

See [ADR-001: Deterministic, Reproducible Data Generation](../lifecycle/project/sprints/SPRINT-001/ADR-001-deterministic-reproducible-data-generation.md).

### Seed Specification in Publications

When publishing results using generated datasets, always document:

```markdown
**Dataset Generation:**
- Generator: `src/utils/data_generator.py`
- Samples: 10,000
- Features: 32
- Distribution: Normal (mean=0, std=1)
- Seed: 42
- Version: src/utils/data_generator.py commit abc1234

This dataset is bit-identical across runs and can be regenerated using:
python src/utils/data_generator.py --samples 10000 --features 32 --seed 42
```

---

## Performance

### Generation Times

On a modern CPU (Intel i7/Ryzen 7, 2023+), typical generation times:

| Size | Distribution | Time (CSV) | Time (NPY) | Memory |
|------|--------------|----------|-----------|--------|
| 1K × 16 | normal | <10ms | <1ms | 0.1 MB |
| 10K × 32 | normal | 50ms | 10ms | 2.4 MB |
| 100K × 64 | normal | 400ms | 50ms | 48 MB |
| 1M × 128 | normal | 4.5s | 0.5s | 980 MB |

**Note:** Times vary by system. First run may be slower due to module import.

### Performance Tips

**For large datasets (>1M samples):**

1. **Use NPY format** — Much faster than CSV (~8-10x)
   ```bash
   python src/utils/data_generator.py \
     --samples 1000000 \
     --features 128 \
     --format npy
   ```

2. **Reuse generated datasets** — Generate once, use many times
   ```bash
   # Generate
   python src/utils/data_generator.py --samples 100000 --features 32 --seed 42
   
   # Reuse in benchmarks
   python -c "import numpy as np; data = np.load('data/..._seed42.npy')"
   ```

3. **Load with mmap for memory efficiency** (NPY only)
   ```python
   import numpy as np
   # Load without loading into memory immediately
   data = np.load('data/20260511-154500_1000000_32_normal_seed42.npy', mmap_mode='r')
   ```

4. **Parallel generation** (if benchmarking multiple seeds)
   ```bash
   # Generate multiple datasets in parallel
   for seed in 42 123 999; do
     python src/utils/data_generator.py \
       --samples 100000 \
       --features 32 \
       --seed $seed &
   done
   wait
   ```

### Memory Requirements

Memory usage scales linearly with dataset size:

```
Memory = samples × features × 8 bytes (float64)
```

Examples:
- 10K × 32: 2.4 MB
- 100K × 64: 48 MB
- 1M × 128: 976 MB
- 10M × 256: 19.5 GB

---

## Troubleshooting

### Problem: "Module not found: src.utils"

**Solution:** Ensure you're running from the project root directory:

```bash
# Wrong
cd src/utils
python data_generator.py --samples 100 --features 10

# Correct
cd hpc-clustering-framework
python src/utils/data_generator.py --samples 100 --features 10
```

### Problem: "Permission denied" saving to output directory

**Solution:** Check directory permissions:

```bash
# Check current permissions
ls -ld data/

# Fix permissions (Linux/Mac)
chmod 755 data/

# Create with proper permissions
python src/utils/data_generator.py \
  --samples 1000 \
  --features 32 \
  --output-dir ./data
```

### Problem: "Disk full" error

**Solution:** Free up disk space or use a different output directory:

```bash
# Check available space
df -h

# Use different directory
python src/utils/data_generator.py \
  --samples 100000 \
  --features 32 \
  --output-dir /tmp
```

### Problem: "Out of memory" for large datasets

**Solution:** Reduce dataset size or increase RAM:

```bash
# Reduce size
python src/utils/data_generator.py \
  --samples 100000 \
  --features 64 \
  --format npy  # Use binary for efficiency

# Or split into chunks (generate multiple smaller datasets)
for i in {1..10}; do
  python src/utils/data_generator.py \
    --samples 100000 \
    --features 32 \
    --seed $((42 + i)) \
    --output-dir ./chunks
done
```

### Problem: "AttributeError: 'numpy.random.RandomState' has no attribute..."

**Solution:** Ensure numpy version matches requirements:

```bash
pip install numpy==2.4.4
python -c "import numpy; print(numpy.__version__)"
```

### Problem: Generated datasets don't match expected values

**Solution:** Verify seed and parameters are identical:

```bash
# Incorrect: Different seeds
python src/utils/data_generator.py --samples 1000 --features 32 --seed 42
python src/utils/data_generator.py --samples 1000 --features 32 --seed 123
# These will be different!

# Correct: Same seed
python src/utils/data_generator.py --samples 1000 --features 32 --seed 42
python src/utils/data_generator.py --samples 1000 --features 32 --seed 42
# These are bit-identical
```

---

## Advanced Usage

### Programmatic Generation

Use the generator in Python code:

```python
from src.utils.data_generator import generate_dataset, save_dataset
import os

# Generate dataset
data = generate_dataset(samples=10000, features=32, distribution='normal', seed=42)
print(f"Generated: {data.shape}")

# Save to disk
output_path = os.path.join('data', 'my_dataset.csv')
save_dataset(data, output_path, format='csv')
print(f"Saved to: {output_path}")

# Use in algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42).fit(data)
print(f"Found {kmeans.n_clusters} clusters")
```

### Batch Generation

Generate multiple datasets with different parameters:

```bash
#!/bin/bash
# Generate datasets for benchmarking

SEEDS=(42 123 999 555 777)
SIZES=(1000 5000 10000)
FEATURES=(16 32 64)

for seed in "${SEEDS[@]}"; do
  for size in "${SIZES[@]}"; do
    for feat in "${FEATURES[@]}"; do
      python src/utils/data_generator.py \
        --samples $size \
        --features $feat \
        --seed $seed \
        --format npy \
        --output-dir ./benchmark_data
      echo "Generated: $size × $feat (seed=$seed)"
    done
  done
done
```

### Integration with Benchmarking

```python
import subprocess
import numpy as np
from pathlib import Path
import time

def benchmark_clustering():
    """Benchmark clustering with generated datasets."""
    
    datasets = [
        {'samples': 1000, 'features': 16, 'seed': 42},
        {'samples': 10000, 'features': 32, 'seed': 123},
        {'samples': 100000, 'features': 64, 'seed': 999},
    ]
    
    for config in datasets:
        # Generate dataset
        subprocess.run([
            'python', 'src/utils/data_generator.py',
            '--samples', str(config['samples']),
            '--features', str(config['features']),
            '--seed', str(config['seed']),
            '--format', 'npy',
            '--output-dir', 'benchmark_data',
        ])
        
        # Find generated file
        files = list(Path('benchmark_data').glob('*.npy'))
        latest = max(files, key=lambda p: p.stat().st_mtime)
        
        # Run benchmark
        data = np.load(latest)
        start = time.time()
        # Your clustering algorithm here
        elapsed = time.time() - start
        
        print(f"{config['samples']} × {config['features']}: {elapsed:.2f}s")
```

---

## FAQ

### Q: Can I use this for production machine learning?

**A:** No. This generator creates synthetic data for **benchmarking** clustering algorithms. For production ML:
- Use real data from your domain
- Implement proper data validation pipelines
- Consider privacy and fairness implications

Synthetic data is useful for algorithm development and testing, not for models deployed in production.

### Q: Can I add custom distributions?

**A:** Currently, only normal and uniform distributions are supported (see `--distribution`). To add custom distributions:

1. Edit `src/utils/data_generator.py`
2. Add new distribution in `generate_dataset()` function
3. Add corresponding tests in `tests/test_data_generator.py`
4. Update regression baselines (see ADR-001)
5. Submit a PR to the main repository

### Q: Why does my dataset look different from the expected one?

**A:** Possible causes:
- **Different seed:** Use `--seed 42` (or original seed)
- **Different numpy version:** Run `pip install numpy==2.4.4`
- **Different platform:** Some numerical operations may have minor differences on different CPUs (rare; use regression tests to verify)

### Q: Can I generate datasets larger than available RAM?

**A:** Not directly. The generator loads entire dataset into memory before saving. For very large datasets (>100GB):

1. **Generate multiple smaller chunks:**
   ```bash
   for i in {1..100}; do
     python src/utils/data_generator.py \
       --samples 1000000 \
       --features 32 \
       --seed $((42 + i)) \
       --format npy
   done
   ```

2. **Use memory-mapped loading** (NPY only):
   ```python
   import numpy as np
   data = np.load('data.npy', mmap_mode='r')  # Doesn't load into RAM
   ```

### Q: Should I commit generated datasets to version control?

**A:** **No.** Generated datasets are large and deterministic. Instead:

1. **Commit the generator code** (you have it)
2. **Commit golden regression baselines** (`tests/regression_baselines.json`)
3. **Document dataset generation** in your project README:
   ```bash
   # Regenerate datasets locally
   python scripts/generate_benchmarks.sh
   ```
4. **Use gitignore** to exclude generated files:
   ```
   data/
   *.npy
   *.csv
   ```

### Q: How do I verify my dataset hasn't been corrupted?

**A:** Use regression tests:

```bash
python -m pytest tests/regression_tests.py -v
```

This verifies:
- Generated checksums match golden baselines
- No algorithm drift
- Reproducibility is intact

### Q: Can I parallelize generation?

**A:** Yes, using different seeds:

```python
import subprocess
from concurrent.futures import ThreadPoolExecutor

def generate_dataset(seed):
    subprocess.run([
        'python', 'src/utils/data_generator.py',
        '--samples', '100000',
        '--features', '32',
        '--seed', str(seed),
        '--format', 'npy',
    ])

# Generate 10 datasets in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    seeds = range(42, 52)
    executor.map(generate_dataset, seeds)
```

---

## Support

For issues or feature requests:
- **GitHub Issues:** https://github.com/Izpiz06/hpc-clustering-framework/issues
- **Documentation:** See `docs/` directory
- **ADR-001:** [Deterministic, Reproducible Data Generation](../lifecycle/project/sprints/SPRINT-001/ADR-001-deterministic-reproducible-data-generation.md)

---

*Last updated: May 11, 2026*
*Data Generator v1.0*
