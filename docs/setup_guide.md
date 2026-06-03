# Developer Setup Guide

This guide walks new contributors through setting up the HPC Clustering Framework locally, running the current CPU benchmark, and validating changes with tests.

## Prerequisites

- Python 3.8 or newer
- Git
- A terminal or command prompt

GPU work is optional. You only need an NVIDIA GPU, CUDA toolkit, and a matching CuPy package when you are contributing to GPU backend tasks.

## 1. Clone The Repository

```bash
git clone https://github.com/Izpiz06/hpc-clustering-framework.git
cd hpc-clustering-framework
```

If you are contributing through a fork, replace the clone URL with your fork URL.

## 2. Create A Virtual Environment

### Linux And macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

If your system uses `python` instead of `python3`, use:

```bash
python -m venv venv
source venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run PowerShell as your user and allow scripts for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
py -m venv venv
venv\Scripts\activate.bat
```

After activation, your prompt should show the virtual environment name, usually `(venv)`.

## 3. Install Dependencies

Upgrade packaging tools first, then install the pinned project dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For Windows, the same commands work after activating the virtual environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Verify Imports

Run a quick import check from the repository root:

```bash
python3 -c "from src.cpu_backend import KMeansCPU; print(KMeansCPU)"
```

If this command fails, confirm that you are in the repository root and that the virtual environment is active.

## 5. Generate A Sample Dataset

The data generator creates reproducible CSV or NPY files for clustering benchmarks.

```bash
python3 -m src.utils.data_generator --samples 1000 --features 16 --seed 42
```

Generated files are written to the `data/` directory by default. For more examples, see `docs/data_generator_how_to.md`.

## 6. Run The CPU Benchmark

The current benchmark script is:

```bash
python3 -m src.benchmarks.cpu_benchmarks
```

This generates a synthetic dataset, trains the NumPy CPU K-Means backend, and prints the elapsed runtime. Use this command before and after performance-related changes to compare behavior.

## 7. Run Tests

Run the full test suite:

```bash
python3 -m pytest tests/
```

Run a focused test file while working on a specific area:

```bash
python3 -m pytest tests/test_kmeans_cpu.py -v
python3 -m pytest tests/test_data_generator.py -v
python3 -m pytest tests/regression_tests.py -v
```

Regression tests compare generated datasets against saved checksums. If a regression test fails after an intentional algorithm change, explain the reason clearly in your pull request.

If these checksum-based tests fail immediately after a fresh install, recreate the virtual environment and confirm you are using the pinned versions from `requirements.txt`. In this repository, the regression baselines are sensitive to NumPy and scikit-learn version drift.

## 8. Optional GPU Contributor Setup

GPU dependencies are intentionally not installed by default because CuPy packages depend on your CUDA version.

1. Confirm your NVIDIA driver and CUDA version:

   ```bash
   nvidia-smi
   ```

2. Install the CuPy wheel that matches your CUDA runtime. Examples:

   ```bash
   python -m pip install cupy-cuda11x
   python -m pip install cupy-cuda12x
   ```

3. Only install Numba when your assigned issue needs it:

   ```bash
   python -m pip install "numba>=0.56.0"
   ```

Mention your GPU model, CUDA version, dataset size, and benchmark results in pull requests that change GPU or performance code.

## 9. Common Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run commands from the repository root:

```bash
pwd
```

The output should end with `hpc-clustering-framework`.

### `pytest: command not found`

Activate the virtual environment and install dependencies again:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests/
```

### Dependency Installation Fails

Check your Python version:

```bash
python3 --version
```

Use Python 3.8 or newer, then recreate the virtual environment if needed:

```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows Command Prompt, remove the virtual environment with:

```bat
rmdir /s /q venv
py -m venv venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### GPU Package Does Not Import

Confirm that the installed CuPy package matches your CUDA version. If you are not working on a GPU issue, you can skip GPU dependencies and still run CPU benchmarks and tests.

## 10. Before Opening A Pull Request

From the repository root, run:

```bash
python3 -m src.benchmarks.cpu_benchmarks
python3 -m pytest tests/
```

Then include the commands you ran and their results in your pull request description.
