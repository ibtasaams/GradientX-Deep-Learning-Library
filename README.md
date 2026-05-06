<div align="center">

<img src="https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" />
<img src="https://img.shields.io/badge/Deep_Learning-From_Scratch-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Autograd-Reverse_Mode-4CAF50?style=for-the-badge" />
<img src="https://img.shields.io/badge/Accuracy-97%25+-E91E63?style=for-the-badge" />
<img src="https://img.shields.io/badge/Zero-Dependencies-9C27B0?style=for-the-badge" />

# 🧠 GradientX

### A Production-Grade Deep Learning Framework Written Entirely in C++17

**No TensorFlow. No PyTorch. No Eigen. No NumPy. No BLAS. Zero ML Dependencies.**

Every tensor operation, every gradient, every memory allocation — handcrafted from first principles.

[Features](#-core-engine-features) · [Architecture](#-system-architecture) · [Datasets](#-multi-dataset-intelligence-pipeline) · [Math](#-mathematical-engine) · [Build](#%EF%B8%8F-build--installation) · [Web Demo](#-browser-based-inference-engine)

---

</div>

## 🔥 What Makes GradientX Different

Most "from scratch" neural networks are 200-line scripts that multiply matrices. **GradientX is not that.** It is a fully engineered deep learning *framework* — a miniature PyTorch — with its own computation graph compiler, reverse-mode automatic differentiation engine, virtual memory management system, and multi-dataset training pipeline. It compiles neural network architectures into optimized execution programs, propagates gradients through arbitrary DAGs, and manages memory through OS-level virtual memory primitives.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GradientX v1.0  —  Systems-Level Deep Learning Infrastructure         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ✦ Reverse-Mode Automatic Differentiation (Autograd)                   │
│  ✦ Computation Graph Compiler with Topological Sort                    │
│  ✦ Virtual Memory Arena Allocator (VirtualAlloc / mmap)                │
│  ✦ Multi-Dataset Pipeline (MNIST + Fashion-MNIST + Custom)             │
│  ✦ Xavier/Glorot Weight Initialization                                 │
│  ✦ Residual Skip Connections (ResNet-inspired)                         │
│  ✦ Model Serialization & Binary Weight Export                          │
│  ✦ Browser-Based Neural Network Inference Engine                       │
│  ✦ Real-Time TrueColor Terminal Visualization                          │
│  ✦ Interactive Prediction Console                                      │
│  ✦ PCG-32 Cryptographic-Quality PRNG                                   │
│  ✦ Cross-Platform (Windows + Linux)                                    │
│                                                                        │
│  Language: C++17  │  Lines of Code: ~1,500  │  External ML Deps: 0     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 System Architecture

```text
                          ┌─────────────────────────────┐
                          │         main.cpp            │
                          │   Application Controller    │
                          │  ┌───────────────────────┐  │
                          │  │ Dataset Selection     │  │
                          │  │ Network Construction  │  │
                          │  │ Training Loop         │  │
                          │  │ Interactive Console   │  │
                          │  │ Weight Serialization  │  │
                          │  │ TrueColor Renderer    │  │
                          │  └───────────────────────┘  │
                          └─────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
   │    matrix.cpp      │   │     model.cpp      │   │     prng.cpp       │
   │  Tensor Engine     │   │  Autograd Engine   │   │  PCG-32 PRNG       │
   │                    │   │                    │   │                    │
   │ • mat_mul (4 modes)│   │ • Graph Builder    │   │ • State Machine    │
   │ • mat_relu         │   │ • Topo Sort (DFS)  │   │ • XorShift + Rot   │
   │ • mat_softmax      │   │ • Forward Pass     │   │ • Uniform Float    │
   │ • mat_cross_entropy│   │ • Backward Pass    │   │ • Deterministic    │
   │ • Jacobian Compute │   │ • SGD Optimizer    │   │   Reproducibility  │
   │ • Gradient Accum.  │   │ • Batch Scheduler  │   │                    │
   └────────┬───────────┘   └────────┬───────────┘   └────────┬───────────┘
            │                        │                         │
            └────────────────────────┼─────────────────────────┘
                                     │
                                     ▼
                          ┌────────────────────────┐
                          │      arena.cpp         │
                          │  Memory Subsystem      │
                          │                        │
                          │ • VirtualAlloc / mmap   │
                          │ • Reserve / Commit      │
                          │ • Scratch Arenas (TLS)  │
                          │ • O(1) Bump Allocation  │
                          │ • Conflict Resolution   │
                          └────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────────┐
   │                    Python Tooling Layer                            │
   │  getmnist.py │ getfashion.py │ import_custom_dataset.py │         │
   │                              │ export_web_data.py                 │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Core Engine Features

### 1. Reverse-Mode Automatic Differentiation

The autograd engine constructs a **directed acyclic graph (DAG)** at model definition time. Each node (`ModelVar`) stores its value tensor, gradient tensor, operation type, and parent pointers. During compilation, the graph is **topologically sorted via iterative DFS** into a linear execution program.

```cpp
// Computation graph nodes
struct ModelVar {
    Matrix*    val;                          // Forward value
    Matrix*    grad;                         // Accumulated gradient
    ModelVarOp op;                           // Operation enum
    ModelVar*  inputs[MODEL_VAR_MAX_INPUTS]; // Parent edges
};

// Supported differentiable operations
enum class ModelVarOp : u32 {
    Relu, Softmax,                          // Unary
    Add, Sub, MatMul, CrossEntropy          // Binary
};
```

**Backward pass** implements the chain rule in reverse topological order, with analytically derived gradients for every operation — including the full **Jacobian matrix computation** for softmax backpropagation.

### 2. Computation Graph Compiler

```text
Model Definition  →  Graph Construction  →  Iterative DFS  →  Topological Sort  →  ModelProgram
                                                                                        │
                                                     ┌─────────────────────────────────┘
                                                     ▼
                                              Linear execution
                                              of sorted nodes
                                              (forward & backward)
```

The compiler produces two execution programs:
- **Forward Program** — for inference (ends at softmax output)
- **Cost Program** — for training (ends at cross-entropy loss, enables backward pass)

### 3. Virtual Memory Arena Allocator

GradientX bypasses `malloc`/`free` entirely. Memory is managed through a custom **linear arena allocator** built on OS virtual memory primitives:

| Platform | Reserve | Commit | Decommit | Release |
|----------|---------|--------|----------|---------|
| **Windows** | `VirtualAlloc(MEM_RESERVE)` | `VirtualAlloc(MEM_COMMIT)` | `VirtualFree(MEM_DECOMMIT)` | `VirtualFree(MEM_RELEASE)` |
| **Linux** | `mmap(PROT_NONE)` | `mprotect(PROT_READ\|WRITE)` | `madvise(MADV_DONTNEED)` | `munmap()` |

**Key properties:**
- **O(1) allocation** — pointer bump, no free-list traversal
- **Zero fragmentation** — contiguous address space
- **Demand-paged commits** — reserves 1 GiB virtual, commits in 1 MiB pages
- **Thread-local scratch arenas** — two per thread with conflict-aware selection
- **Typed push helpers** — `push_struct<T>()` and `push_array<T>()` with optional zero-initialization

### 4. High-Performance Tensor Engine

The matrix library implements **four specialized GEMM kernels** for all transpose combinations:

```text
mat_mul dispatch:
  ┌─────────────────┬──────────────────────────┐
  │ transpose flags  │ kernel                   │
  ├─────────────────┼──────────────────────────┤
  │ NN (0b00)       │ _mat_mul_nn  (standard)  │
  │ NT (0b01)       │ _mat_mul_nt  (B^T)       │
  │ TN (0b10)       │ _mat_mul_tn  (A^T)       │
  │ TT (0b11)       │ _mat_mul_tt  (both)      │
  └─────────────────┴──────────────────────────┘
```

Loop ordering uses **i-k-j** for cache-friendly row-major access patterns. Backpropagation through `MatMul` uses the transpose kernels:
- `dL/dA = dL/dC · B^T` (NT kernel)
- `dL/dB = A^T · dL/dC` (TN kernel)

### 5. Multi-Dataset Intelligence Pipeline

GradientX supports **three distinct dataset modes** through a unified training interface:

| Dataset | Input Dim | Classes | Resolution | Use Case |
|---------|-----------|---------|------------|----------|
| **MNIST** | 784 | 10 | 28×28 | Handwritten digit recognition |
| **Fashion-MNIST** | 784 | 10 | 28×28 | Clothing item classification |
| **Custom Dataset** | N×N | K | Configurable | Any image classification task |

The custom dataset pipeline (`import_custom_dataset.py`) accepts **any folder of real-world photographs**, automatically:
- Discovers class labels from subdirectory names
- Converts to grayscale and resizes to 64×64
- Generates 80/20 train/test split with Fisher-Yates shuffle
- Writes binary `.mat` files + metadata for the C++ engine

### 6. Model Serialization & Weight Persistence

Trained weights are serialized to compact binary files for deployment:

```text
Binary Format:  [u32 rows][u32 cols][f32[] data]

Saved files:    w0.bin  w1.bin  w2.bin  (weight matrices)
                b0.bin  b1.bin  b2.bin  (bias vectors)
```

This enables **train-once, deploy-anywhere** — including the browser-based inference engine.

### 7. Browser-Based Inference Engine

`export_web_data.py` converts trained weights + test images into a JavaScript data file. The `presentation.html` runs the **full neural network forward pass in the browser** — matrix multiplications, ReLU, softmax — with zero server dependencies.

### 8. Real-Time Terminal Visualization

The CLI renders MNIST/Fashion-MNIST images using **24-bit TrueColor ANSI escape sequences** with per-pixel grayscale mapping:

```text
    +--------------------------------------------------------+
    |                    (TrueColor render)                   |
    |              28×28 grayscale visualization              |
    +--------------------------------------------------------+
    Confidence breakdown:
         0: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.12%
         7: ████████████████████████████████████████░░  98.43%  ← predicted
         ...
```

### 9. Interactive Prediction Console

After training, users enter an interactive REPL to query any test image by index, view the TrueColor render, and see a full confidence breakdown across all classes.

---

## 🧠 Neural Network Architecture

```text
Input (784 / 4096)
     │
     ▼
┌─────────────────────────┐
│  Dense Layer 0          │
│  W₀·x + b₀             │     Xavier/Glorot Initialization
│  ReLU Activation        │     b₀ = √(6 / (fan_in + fan_out))
└─────────┬───────────────┘
          │
          ├─────────────────────────────┐  ← Skip Connection
          │                             │
          ▼                             │
┌─────────────────────────┐             │
│  Dense Layer 1          │             │
│  W₁·a₀ + b₁            │             │
│  ReLU Activation        │             │
└─────────┬───────────────┘             │
          │                             │
          ▼                             │
┌─────────────────────────┐             │
│  Residual Addition      │◄────────────┘
│  a₁ = a₀ + ReLU(z₁)    │  ResNet-style identity mapping
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Dense Layer 2          │
│  W₂·a₁ + b₂            │
│  Softmax Activation     │
└─────────┬───────────────┘
          │
          ▼
    Probability Distribution (10 or K classes)
          │
          ▼
    Cross-Entropy Loss ←── One-Hot Ground Truth
```

The architecture **dynamically adapts**: MNIST/Fashion-MNIST use 16-neuron hidden layers; custom high-resolution datasets automatically scale to 64 neurons.

---

## 📊 Training Pipeline

```text
┌────────────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (per epoch)                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Fisher-Yates Shuffle (PCG-32 randomized)                         │
│     └→ Decorrelate sample ordering to prevent gradient bias          │
│                                                                      │
│  2. Mini-Batch Forward Pass                                          │
│     └→ Execute topologically-sorted forward program                  │
│     └→ Accumulate cross-entropy cost                                 │
│                                                                      │
│  3. Reverse-Mode Backpropagation                                     │
│     └→ Seed: dL/dL = 1.0                                            │
│     └→ Chain rule through every graph node (reverse order)           │
│     └→ Gradient accumulation across batch samples                    │
│                                                                      │
│  4. SGD Parameter Update                                             │
│     └→ W ← W - (η / batch_size) · ∇W                               │
│                                                                      │
│  5. Test-Set Evaluation                                              │
│     └→ Forward-only pass on 10,000 test images                      │
│     └→ Report accuracy + average cost                                │
│                                                                      │
│  Hyperparameters: epochs=10, batch_size=50, η=0.01                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Training Results

| Epoch | MNIST Accuracy | Fashion-MNIST Accuracy | Avg Loss |
|-------|---------------|----------------------|----------|
| 1     | ~92%          | ~82%                 | 0.29     |
| 3     | ~95%          | ~85%                 | 0.18     |
| 5     | ~96%          | ~86%                 | 0.13     |
| 10    | **~97%**      | **~87%**             | 0.09     |

---

## 📂 Project Structure

```text
GradientX/
│
├── base.hpp                    # Fixed-width type system (i32, f32, u64, b8)
│                               # Memory size helpers (KiB, MiB, GiB)
│                               # Alignment & generic utility templates
│
├── arena.hpp / arena.cpp       # Virtual memory arena allocator
│                               # OS-level reserve/commit/decommit/release
│                               # Thread-local scratch arenas with conflict resolution
│                               # Typed push_struct<T> / push_array<T> helpers
│
├── prng.hpp / prng.cpp         # PCG-32 pseudo-random number generator
│                               # XorShift + bit rotation output function
│                               # Uniform float generation for weight init
│
├── matrix.hpp / matrix.cpp     # Row-major tensor engine
│                               # 4-kernel GEMM dispatcher (NN/NT/TN/TT)
│                               # Forward: ReLU, Softmax, CrossEntropy
│                               # Backward: Jacobian, gradient accumulation
│                               # Reductions: sum, argmax
│
├── model.hpp / model.cpp       # Computation graph & autograd engine
│                               # DAG construction with ModelVar nodes
│                               # Iterative DFS topological sort compiler
│                               # Forward/backward program execution
│                               # Mini-batch SGD training loop
│
├── main.cpp                    # Application controller
│                               # Multi-dataset selection (MNIST/Fashion/Custom)
│                               # Network construction & Xavier initialization
│                               # TrueColor terminal image rendering
│                               # Interactive prediction REPL
│                               # Binary weight serialization
│
├── CMakeLists.txt              # Cross-platform build configuration
│
├── getmnist.py                 # MNIST dataset downloader & preprocessor
├── getfashion.py               # Fashion-MNIST downloader & preprocessor
├── import_custom_dataset.py    # Custom image folder → binary pipeline
├── export_web_data.py          # Weight + image → JavaScript exporter
│
└── presentation.html           # Browser-based inference demo
                                # Full forward pass in vanilla JavaScript
```

---

## ⚙️ Build & Installation

### Prerequisites

- C++17 compiler (MSVC / GCC / Clang)
- CMake 3.10+
- Python 3.x with `numpy`

### Windows

```powershell
# 1. Download dataset
pip install numpy
python getmnist.py            # Standard MNIST
python getfashion.py          # Fashion-MNIST (optional)

# 2. Build
cmake -B build
cmake --build build

# 3. Run
cd build/Debug
./deeplearn.exe
```

### Linux / macOS

```bash
pip3 install numpy
python3 getmnist.py

g++ -std=c++17 -O2 -o deeplearn \
    main.cpp arena.cpp prng.cpp matrix.cpp model.cpp -lm

./deeplearn
```

### Custom Dataset

```bash
# Organize images into class folders:
#   my_photos/cat/*.jpg
#   my_photos/dog/*.jpg

pip install Pillow
python import_custom_dataset.py my_photos/

# Then run deeplearn.exe and select option 3
```

---

## 🏗️ Technical Specifications

| Component | Specification |
|-----------|--------------|
| **Language** | C++17 (no STL containers, minimal `<cstdio>` / `<cstring>` / `<cmath>`) |
| **Memory Model** | Virtual memory arena with demand-paged commits |
| **Address Space** | 1 GiB reserved, 1 MiB commit granularity |
| **Scratch Arenas** | 2 × 64 MiB thread-local, conflict-aware |
| **PRNG** | PCG-32 (period: 2⁶⁴, statistically superior to LCG) |
| **Tensor Layout** | Row-major, contiguous f32 arrays |
| **Autograd** | Reverse-mode, DAG-based, topologically compiled |
| **Optimizer** | Mini-batch SGD with configurable η and batch size |
| **Initialization** | Xavier/Glorot uniform: U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))] |
| **Loss** | Cross-entropy with numerically stable log computation |
| **Platforms** | Windows (VirtualAlloc), Linux (mmap/mprotect) |

---

## 🏆 Achievements

- ✅ **Complete Autograd Engine** — reverse-mode differentiation with analytical Jacobians
- ✅ **Computation Graph Compiler** — iterative DFS topological sort into executable programs
- ✅ **Virtual Memory Allocator** — OS-level reserve/commit with demand paging
- ✅ **4-Kernel GEMM Dispatcher** — NN/NT/TN/TT matrix multiply with cache-optimized loops
- ✅ **Multi-Dataset Pipeline** — MNIST, Fashion-MNIST, and arbitrary custom image datasets
- ✅ **Model Serialization** — binary weight export for cross-platform deployment
- ✅ **Browser Inference** — full forward pass in vanilla JavaScript
- ✅ **TrueColor Visualization** — 24-bit ANSI terminal rendering of images
- ✅ **Interactive REPL** — live prediction console with confidence charts
- ✅ **Residual Connections** — ResNet-style skip connections for gradient flow
- ✅ **Xavier Initialization** — proper variance scaling for stable convergence
- ✅ **97%+ Accuracy** — on MNIST with a 3-layer architecture
- ✅ **Zero ML Dependencies** — no TensorFlow, PyTorch, Eigen, BLAS, or any ML library

---

## 👨‍💻 Authors

| Name | Role |
|------|------|
| **Ibtasaam Abbasi** | Lead Developer & Systems Architect |
| **Alfa Gohar** | Developer & Co-Architect |
| **Tabassum Farid** | Developer & Co-Architect |

### 🏫 Institution

**National University of Sciences & Technology (NUST)**
School of Electrical Engineering & Computer Science (SEECS)
**Course:** CS-107 — Computer Programming

---

## 📜 License

Educational and research use. Built to demonstrate that deep learning is engineering, not magic.

---

<div align="center">

### ⭐ If this project impressed you, consider starring the repository ⭐

*Engineered from first principles in C++17*

**GradientX** — Where every gradient is earned, not borrowed.

</div>

