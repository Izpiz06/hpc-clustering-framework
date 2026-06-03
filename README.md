# High-Performance GPU-Accelerated Clustering Framework

[![GSSoC 2026](https://img.shields.io/badge/GSSoC-2026-orange?style=for-the-badge)](https://gssoc.girlscript.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Contributors](https://img.shields.io/github/contributors/Izpiz06/hpc-clustering-framework?style=for-the-badge)](https://github.com/Izpiz06/hpc-clustering-framework/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/Izpiz06/hpc-clustering-framework?style=for-the-badge)](https://github.com/Izpiz06/hpc-clustering-framework/issues)

An open-source high-performance machine learning optimization framework focused on accelerating clustering algorithms using GPU computing and exposing contributors to practical systems-level performance engineering concepts.

---

## Overview

Traditional CPU-based machine learning pipelines struggle when processing large-scale, high-dimensional datasets. While many libraries provide GPU support, the underlying mechanics of acceleration, memory optimization, profiling, and parallel computation often remain hidden from developers.

This project bridges that gap by providing a modular framework where contributors can directly compare standard CPU implementations against GPU-accelerated backends using modern Python HPC tools.

The framework serves as a hands-on introduction to:
- GPU computing
- Parallel processing
- High-performance computing (HPC)
- Performance profiling
- Memory optimization
- ML systems engineering

---

## What This Project Does

This framework benchmarks and accelerates clustering algorithms across CPU and GPU backends using:
- NumPy
- CuPy
- Numba

Contributors can experiment with:
- backend optimization
- vectorized computation
- runtime benchmarking
- memory profiling
- scalability testing

The project is designed to be educational, contributor-friendly, and scalable from beginner-level tasks to advanced GPU optimization challenges.

---

## Why This Project?

Most beginner ML repositories focus primarily on:
- model training
- datasets
- accuracy metrics

This project instead focuses on:
- GPU acceleration
- scalability
- performance engineering
- backend optimization
- systems-level ML concepts

The goal is to help contributors transition from standard scripting into practical machine learning systems and high-performance computing.

---

## Features

- CPU-based K-Means implementation using NumPy
- GPU-accelerated implementation using CuPy
- Optional Numba JIT kernel integration
- Runtime benchmarking tools
- Memory profiling utilities
- Synthetic dataset generation
- Clustering visualization tools
- Modular backend architecture
- Streamlit dashboard for performance analysis
- Beginner-friendly contribution structure

---

## Architecture Overview

```text
Dataset Input
      ↓
Backend Selector
 ┌─────────────┐
 │ CPU Backend │
 │ GPU Backend │
 └─────────────┘
      ↓
Benchmark Engine
      ↓
Visualization Dashboard
```

---

## Tech Stack

### Core Technologies
- Python 3.8+
- NumPy
- Scikit-Learn
- CuPy
- Numba

### Visualization & Dashboard
- Matplotlib
- Plotly
- Streamlit

### Testing & Profiling
- PyTest
- cProfile

---

## Repository Structure

```text
├── src/
│   ├── cpu_backend/        # NumPy implementations
│   ├── gpu_backend/        # CuPy and Numba acceleration
│   ├── benchmarks/         # Benchmark scripts
│   └── utils/              # Dataset generators and helpers
│
├── tests/                  # Automated test suite
├── dashboard/              # Streamlit visualization dashboard
├── docs/                   # Documentation
├── examples/               # Example scripts
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Current Status

| Component | Status |
|---|---|
| CPU K-Means Baseline | Completed |
| GPU Backend | In Progress |
| Benchmark Engine | Planned |
| Dashboard | Planned |
| Multi-GPU Support | Future |

---

## Getting Started

### Prerequisites

#### CPU-Only Setup
- Python 3.8+

#### GPU Acceleration Setup
- NVIDIA GPU
- CUDA Toolkit
- Compatible CUDA Drivers

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Izpiz06/hpc-clustering-framework.git
cd hpc-clustering-framework
```

### Create a Virtual Environment

#### Linux / macOS
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Synthetic Dataset Generation

Generate reproducible synthetic datasets for benchmarking clustering algorithms:

```bash
# Basic: 10K samples, 32 features, CSV format
python src/utils/data_generator.py --samples 10000 --features 32

# Advanced: Custom distribution, seed, and format
python src/utils/data_generator.py \
  --samples 5000 \
  --features 64 \
  --distribution uniform \
  --seed 12345 \
  --format npy
```

Generated datasets are saved to the `data/` directory with deterministic, bit-identical output when using the same seed. This ensures reproducible benchmarking across runs.

For detailed usage, examples, and reproducibility guidance, see [docs/data_generator_how_to.md](docs/data_generator_how_to.md).

### Run CPU Benchmark

```bash
python src/benchmarks/cpu_benchmark.py
```

### Run GPU Benchmark

```bash
python src/benchmarks/gpu_benchmark.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Benchmarking Goals

The framework focuses on comparing:
- CPU vs GPU execution time
- Memory consumption
- Scalability with increasing dataset sizes
- Performance gains from vectorization
- Batch processing efficiency

Example benchmark metrics:
- Runtime
- Throughput
- GPU utilization
- Memory transfer overhead
- Clustering convergence speed

---

## Sample Benchmark Results

| Dataset Size | CPU Runtime | GPU Runtime | Speedup |
|---|---|---|---|
| 10,000 Points | 0.5s | TBD | TBD |
| 100,000 Points | 23.8s | TBD | TBD |

> Note: Benchmark values above are sample results and will vary depending on hardware configuration.

---

## Learning Objectives

Contributors will gain practical experience in:
- GPU acceleration
- ML systems engineering
- HPC fundamentals
- Vectorized computation
- Performance engineering
- Parallel processing
- Profiling and benchmarking

---

## Open Source Goals

This repository aims to:
- provide beginner-friendly HPC exposure
- encourage systems-level ML learning
- introduce GPU acceleration concepts
- create a contributor-friendly optimization framework
- help contributors explore practical performance engineering

---

## Contributing

We welcome contributors of all experience levels.

Please read `CONTRIBUTING.md` before making contributions.

### Beginner-Friendly Contributions
- Documentation improvements
- README enhancements
- Unit tests
- Dataset generation scripts
- Visualization improvements
- CLI utilities

### Intermediate Contributions
- Performance profiling
- Benchmarking tools
- NumPy optimization
- Memory optimization
- Additional clustering algorithms

### Advanced Contributions
- Custom CUDA kernels with Numba
- Multi-GPU experimentation
- Distributed clustering
- Backend optimization
- Advanced profiling utilities

---

## Good First Issues

Issues labeled:
- `good first issue`
- `beginner friendly`
- `documentation`
- `visualization`

are specifically curated for new contributors.

---

## Roadmap

- [x] CPU K-Means baseline
- [ ] GPU acceleration with CuPy
- [ ] Benchmark dashboard
- [ ] Numba kernel optimization
- [ ] Mini-Batch K-Means
- [ ] Multi-GPU experimentation
- [ ] Distributed clustering
- [ ] PyTorch backend integration

---

## Future Scope

Planned future additions include:
- Distributed clustering
- Multi-GPU acceleration
- CUDA kernel optimization
- PyTorch backend integration
- Real-time benchmark visualization
- Additional clustering algorithms

---

## License

This project is licensed under the MIT License.

---

## Maintainers

### Mohammad Izaan

---

## Contributors

We appreciate all contributors who help improve this project.

<a href="https://github.com/Izpiz06/hpc-clustering-framework/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Izpiz06/hpc-clustering-framework" />
</a>

---

## Community & Support

- GitHub Discussions: https://discord.gg/Nc4ttTzqG

Contributors are encouraged to ask questions, suggest improvements, and participate in discussions related to GPU computing, machine learning systems, and high-performance computing.
