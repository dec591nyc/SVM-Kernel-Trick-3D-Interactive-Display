# ⚛️ SVM Kernel Trick 3D Interactive Demonstration

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3D_Interactive-FF7F0E.svg?style=flat-square)](https://plotly.com/)
[![Manim CE](https://img.shields.io/badge/Manim-Community_Edition-2E8B57.svg?style=flat-square)](https://www.manim.community/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

An educational interactive platform and mathematical visualization tool designed to explain the mechanics of the **Support Vector Machine (SVM) Kernel Trick** using concentric circle datasets.

This project takes students and machine learning developers on a journey from a non-linearly separable 2D space to a linearly separable 3D feature space, showcasing both **explicit coordinate lifting (paraboloid projection)** and **continuous RBF decision surfaces** with smooth interactive transitions.

---

## 📖 Table of Contents
1. [Core Features](#-core-features)
2. [Mathematical Background](#-mathematical-background)
3. [Repository Reorganization](#-repository-reorganization)
4. [Installation and Setup](#-installation-and-setup)
5. [Running the Application](#-running-the-application)
    - [Phase 1: Render Manim Concept Animation](#phase-1-render-manim-concept-animation)
    - [Phase 2: Run Mathematical Verification](#phase-2-run-mathematical-verification)
    - [Phase 3: Run Interactive Streamlit Web App](#phase-3-run-interactive-streamlit-web-app)
6. [Design and Color Palette Guidelines](#-design-and-color-palette-guidelines)
7. [License](#-license)

---

## 🌟 Core Features

- **2D Projection Plane**: Real-time classification boundaries, margin bounds ($f(x, y) = \pm 1$), and highlighted support vectors with locked axis ranges (`[-2.2, 2.2]`) to prevent layout shifts.
- **3D Feature Lifting (Plotly)**: Dynamic coordinate transition animations that lift data points smoothly from $z = 0$ up to the paraboloid surface $z = x^2 + y^2$, showing the separating hyperplane slice in action.
- **RBF Landscape Morphing (Plotly)**: Visualizes the true RBF decision landscape by morphing a flat 2D plane into a 3D decision confidence topography $z = f(x, y)$ client-side.
- **Manim Educational Video**: Embedded pre-rendered mathematical animation illustrating the transition from 2D coordinates to 3D space.
- **Customizable Color Themes**: Switch between **Modern Blue/Red** and **Manim Sakura/Sky** directly from the sidebar.

---

## 🔬 Mathematical Background

### 1. Inseparable 2D Space
Concentric circles (inner Class 0 and outer Class 1) cannot be separated by any 2D straight line of the form:
$$w_1 x_1 + w_2 x_2 + b = 0$$

### 2. Explicit 3D Lifting Function
We can lift the coordinates explicitly into 3D space using the mapping:
$$\Phi(x, y) = \left(x, y, x^2 + y^2\right)$$
Because the inner circle has a smaller radius, the points lie at the bottom of the paraboloid ($z \approx 0.16$), while the outer circle points sit higher up ($z \approx 1.0$). A flat separating hyperplane at $z = 0.5$ cleanly separates the two classes.

### 3. True RBF Kernel Space
The Radial Basis Function (RBF) Kernel is defined as:
$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$
It maps features into an **infinite-dimensional Hilbert space**. We visualize this landscape via the 3D confidence topography:
$$z = f(x_1, x_2) = \sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

---

## 📂 Repository Reorganization

The codebase is structured under the `src/` directory to isolate execution logic from workspace meta-files:

```text
├── README.md                      # Comprehensive developer guide
├── requirements.txt               # Dependencies listing
├── outputs/                       # Rendered verification plots
├── media/                         # Manim CE output folders & video assets
└── src/                           # Source directory package root
    ├── __init__.py
    ├── phase1_manim_kernel_trick.py   # Manim CE concept animation script
    ├── phase2_rbf_decision_surface.py # Matplotlib math verification script
    ├── phase3_streamlit_app.py        # Streamlit interactive web dashboard
    └── utils/                         # Shared utilities module
        ├── __init__.py
        ├── data_generator.py          # Concentric circles dataset generator
        └── svm_utils.py               # Shared ML fitting & stable mesh-grids
```

---

## 🛠️ Installation and Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/dec591nyc/SVM-Kernel-Trick-3D-Interactive-Display.git
   cd SVM-Kernel-Trick-3D-Interactive-Display
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Rendering the Manim video locally requires installing [FFmpeg](https://ffmpeg.org/) on your operating system).*

---

## 🚀 Running the Application

### Phase 1: Render Manim Concept Animation
To render the high-quality concept video explaining the 2D to 3D mapping:
```bash
# High Quality (1080p)
manim -pqh src/phase1_manim_kernel_trick.py SVMKernelTrick3D

# Low Quality (480p - fast render draft)
manim -pql src/phase1_manim_kernel_trick.py SVMKernelTrick3D
```

### Phase 2: Run Mathematical Verification
To verify the RBF SVM fit, plot decision contours, margins, and support vectors statically:
```bash
python src/phase2_rbf_decision_surface.py
```
This saves the high-resolution visualization to `outputs/verification_plot.png`.

### Phase 3: Run Interactive Streamlit Web App
To boot the interactive Streamlit dashboard locally:
```bash
streamlit run src/phase3_streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```
Open **[http://127.0.0.1:8501](http://127.0.0.1:8501)** in your browser to interact with the visualizations.

---

## 🎨 Design and Color Palette Guidelines

For a unified visual narrative, the graphics maintain color consistency:

| Component | Modern Blue/Red |
| :--- | :---: |
| **Class 0 (Inner)** | `#3b82f6` (Vibrant Blue) |
| **Class 1 (Outer)** | `#ef4444` (Vibrant Red) |
| **Decision Boundary** | `#eab308` (Gold Yellow) |
| **Support Vectors** | Outlined in Gold |
| **Margins ($f=\pm 1$)** | Slate Gray (Dashed) |

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
