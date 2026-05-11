# High-Performance GPU-Accelerated Clustering Framework 🚀

[![GSSoC 2026](https://img.shields.io/badge/GSSoC-2026-orange?style=for-the-badge)](https://gssoc.girlscript.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)

An open-source high-performance ML optimization framework designed to bridge the gap between standard Python scripting and GPU-accelerated systems engineering. This project implements and benchmarks clustering algorithms (starting with K-Means) across CPU and GPU backends.

## 🌟 Project Vision
Most ML developers treat the GPU as a "black box." This project aims to demystify hardware acceleration by providing a modular framework where contributors can compare traditional **NumPy** implementations against highly optimized **CuPy** and **Numba** backends. It is a playground for learning parallel computation, memory management, and performance profiling.

## 🛠️ Tech Stack
- **Language:** Python 3.8+
- **CPU Backend:** NumPy, Scikit-Learn
- **GPU Backend:** CuPy, Numba (JIT Kernels)
- **Visualization:** Matplotlib, Streamlit (Dashboard)
- **Testing & Profiling:** PyTest, cProfile

## 📁 Repository Structure
```text
├── src/                  # Core source code
│   ├── cpu_backend/      # Baseline NumPy implementations
│   ├── gpu_backend/      # Accelerated CuPy/Numba logic
│   └── utils/            # Data generators & helpers
├── tests/                # Automated test suite
├── dashboard/            # Streamlit performance visualizer
└── CONTRIBUTING.md       # Contribution guidelines
