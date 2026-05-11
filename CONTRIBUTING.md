# 🤝 Contributing to HPC Clustering Framework

Welcome, and thank you for your interest in contributing to the **High-Performance GPU-Accelerated Clustering Framework**.

This repository is part of **GSSoC 2026** and is designed to help contributors explore:
- GPU Computing
- Machine Learning Systems
- Performance Engineering
- High-Performance Computing (HPC)
- Parallel Processing Concepts

We welcome contributors of all experience levels and encourage meaningful, educational, and collaborative contributions.

---

# 📜 Code of Conduct

By participating in this project, you agree to:
- maintain respectful communication
- collaborate constructively
- support fellow contributors
- focus on learning and quality contributions

Harassment, spam, plagiarism, or disruptive behavior will not be tolerated.

---

# 🚀 Before You Start

Please:
- read the README carefully
- understand the repository structure
- check existing issues before creating a new one
- ask questions if you are unsure about anything

---

# 🛠️ Contribution Workflow

## 1️⃣ Find an Issue

Browse the Issues tab and look for labels such as:
- `gssoc26`
- `good first issue`
- `beginner friendly`
- `documentation`
- `gpu`
- `optimization`

---

## 2️⃣ Request Assignment

Comment on the issue:

```text
I would like to work on this issue. Please assign it to me.
```

Please wait until a maintainer assigns the issue before starting work.

---

## 3️⃣ Fork and Clone the Repository

```bash
git clone https://github.com/Izpiz06/hpc-clustering-framework.git
cd hpc-clustering-framework
```

---

## 4️⃣ Create a New Branch

```bash
git checkout -b feature/your-feature-name
```

Examples:
```bash
git checkout -b feature/gpu-benchmarking
git checkout -b fix/readme-typo
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Make Your Changes

Please ensure your contributions are:
- clean and readable
- properly documented
- modular and maintainable
- consistent with the existing code style

---

## 7️⃣ Run Tests

Before submitting your PR, ensure all tests pass:

```bash
pytest tests/
```

---

## 8️⃣ Commit Your Changes

Use meaningful commit messages.

Examples:
```bash
git commit -m "Add GPU benchmark utility"
git commit -m "Improve clustering visualization"
```

---

## 9️⃣ Push and Open a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request to the `main` branch.

---

# 📌 Pull Request Guidelines

## ⚠️ Important Rules

### 🚫 No Assignment = No PR
Please do not submit Pull Requests for issues that were not assigned to you.

---

### 🎯 One Issue Per Pull Request
Keep PRs focused and modular.

Do not combine:
- bug fixes
- documentation changes
- feature additions

into a single PR unless explicitly requested.

---

### ✅ Meaningful Contributions Only
The following types of PRs are generally discouraged:
- fixing a single typo
- unnecessary formatting changes
- adding empty folders/files
- low-effort cosmetic edits

We value quality contributions over quantity.

---

# 🤖 AI Usage Policy

AI tools such as ChatGPT, Gemini, or Copilot may be used for:
- learning concepts
- debugging assistance
- understanding GPU optimization
- exploring HPC topics

However:

## 🧠 Human Understanding is Mandatory
You must fully understand the code you submit.

Contributors may be asked to explain:
- optimization logic
- implementation choices
- backend behavior
- performance improvements

---

## 🚫 Avoid Raw AI-Generated PRs
Do not submit large blocks of unverified AI-generated code without modification or understanding.

---

## 💬 Authentic Collaboration
Please write your own:
- PR descriptions
- issue discussions
- contributor communication

Human collaboration is an important part of open source.

---

# 📊 Performance Benchmarking Guidelines

If your contribution involves:
- GPU acceleration
- performance optimization
- vectorization
- backend improvements

please include a short performance report in your Pull Request.

Example:

```text
CPU Runtime: 42.3 seconds
GPU Runtime: 3.1 seconds
Hardware Used: NVIDIA RTX 3050 (8GB VRAM)
Dataset Size: 100,000 samples
```

This helps maintain benchmarking consistency across contributions.

---

# 🌱 Beginner-Friendly Contribution Areas

New contributors can start with:
- documentation improvements
- README enhancements
- unit tests
- dataset generators
- visualization improvements
- benchmark scripts
- CLI support

---

# ⚙️ Intermediate Contribution Areas

Intermediate contributors can work on:
- NumPy optimization
- memory optimization
- profiling utilities
- CuPy acceleration
- benchmarking tools
- clustering extensions

---

# 🔥 Advanced Contribution Areas

Advanced contributors may explore:
- custom CUDA kernels
- Numba JIT optimization
- distributed clustering
- multi-GPU experimentation
- advanced profiling systems
- backend architecture improvements

---

# 📁 Project Structure

```text
src/
├── cpu_backend/
├── gpu_backend/
├── benchmarks/
└── utils/

tests/
dashboard/
docs/
examples/
```

---

# 🆘 Need Help?

If you are stuck or confused:
- open a discussion
- ask in issue comments
- contact the maintainers

We encourage collaborative learning and contributor growth.

---

# 👨‍💻 Maintainers

- Mohammad Izaan

---

Thank you for contributing to the HPC Clustering Framework and helping build an accessible learning environment for GPU computing and ML systems engineering.
