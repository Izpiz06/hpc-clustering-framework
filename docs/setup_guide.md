# Developer Setup Guide

This guide helps new contributors set up the HPC Clustering Framework locally, run the test suite, and execute the CPU benchmark script.

## Prerequisites

- Python 3.8 or newer
- Git
- A terminal such as Bash, Zsh, PowerShell, or Windows Terminal

GPU work is optional. You can run the current CPU backend, dataset generator, and tests without CUDA, CuPy, or Numba.

## 1. Clone the Repository

```bash
git clone https://github.com/Izpiz06/hpc-clustering-framework.git
cd hpc-clustering-framework
```

If you are contributing through a fork, replace the clone URL with your fork URL.

## 2. Create a Virtual Environment

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the virtual environment again.

### Windows Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

## 3. Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

The core dependency set installs the CPU backend, benchmark, visualization, and test dependencies.

Optional GPU contributors should install the CuPy build that matches their local CUDA version, for example:

```bash
python -m pip install cupy-cuda12x
```

Use `cupy-cuda11x` instead if your system uses CUDA 11.x.

## 5. Verify Imports

Run a quick import check from the repository root:

```bash
python - <<'PY'
from src.cpu_backend import KMeansCPU
from src.utils.data_generator import generate_dataset

data = generate_dataset(samples=10, features=2, seed=42)
model = KMeansCPU(n_clusters=2, random_state=42).fit(data)
print("Setup OK:", data.shape, model.n_iters_)
PY
```

## 6. Generate Sample Data

Create a small reproducible dataset:

```bash
python src/utils/data_generator.py --samples 1000 --features 16 --seed 42 --format csv
```

Create an NPY dataset:

```bash
python src/utils/data_generator.py --samples 1000 --features 16 --distribution uniform --seed 42 --format npy
```

Generated files are written to the `data/` directory by default.

## 7. Run Tests

Run the full test suite:

```bash
python -m pytest tests
```

Run a single test file while iterating:

```bash
python -m pytest tests/test_kmeans_cpu.py -q
```

Run the regression tests:

```bash
python -m pytest tests/regression_tests.py -q
```

## 8. Run the CPU Benchmark

Run the current CPU K-Means benchmark:

```bash
python -m src.benchmarks.cpu_benchmarks
```

The benchmark generates a synthetic dataset, fits the NumPy CPU backend, and prints timing details. Use this result as the baseline when comparing future GPU or vectorized implementations.

## 9. Common Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root, not from inside `src/` or `tests/`.

### Dependency installation fails

Upgrade packaging tools and retry:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If you are working on GPU code, confirm that your CUDA version matches the CuPy package you installed.

### Tests pass locally but benchmark output changes

Benchmark timings depend on CPU model, power settings, Python version, and background processes. Include your hardware and OS details when reporting performance numbers in a pull request.

## 10. Before Opening a Pull Request

Run these checks from the repository root:

```bash
python -m pytest tests
python -m src.benchmarks.cpu_benchmarks
```

Keep each pull request focused on one issue and include any relevant test or benchmark output in the PR description.
