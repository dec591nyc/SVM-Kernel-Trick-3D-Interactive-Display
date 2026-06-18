# SVM Kernel Trick 3D Interactive Display

An educational platform and mathematical visualization tool designed to explain the **Support Vector Machine (SVM) Kernel Trick** using concentric circles.

This project guides students and machine learning developers through the transition from non-linearly separable 2D space to linearly separable 3D feature spaces, showcasing both explicit coordinate lifting and continuous RBF decision surfaces.

---

## 📂 Project Structure

```text
├── README.md                      # Project documentation and guide
├── requirements.txt               # Required Python packages
├── phase1_manim_kernel_trick.py   # Manim CE concept animation script
├── phase2_rbf_decision_surface.py # Matplotlib math verification script
├── phase3_streamlit_app.py        # Streamlit interactive web dashboard
├── utils/
│   ├── data_generator.py          # Concentric circles dataset generator
│   └── svm_utils.py               # Shared ML fitting and grid utilities
├── assets/                        # Video and image outputs
└── outputs/                       # Verification plots
```

---

## ⚛️ The Mathematical Story

1. **Inseparable 2D Space:**
   Concentric circles (an inner cluster of Blue points and an outer cluster of Red points) cannot be separated by a straight 2D line.
   
2. **Conceptual 3D Lifting Function:**
   We can lift the coordinates explicitly into 3D using the mapping:
   $$\phi(x, y) = (x, y, x^2 + y^2)$$
   This places all points onto a 3D paraboloid surface. Because the inner points have a smaller radius, they sit at the bottom of the paraboloid ($z \approx 0.16$), while the outer points sit higher up ($z \approx 1.0$).
   
3. **Linear Separation in 3D:**
   In 3D, a flat plane (hyperplane) at $z = c$ (e.g. $z = 0.5$) separates the two classes cleanly.
   
4. **Projection Back to 2D:**
   The intersection of the plane $z = c$ and the paraboloid $z = x^2 + y^2$ forms a circle: $x^2 + y^2 = c$. Projecting this circle back to 2D gives the circular decision boundary.

5. **True RBF Kernel Space:**
   While the paraboloid is useful for visualization, the **Radial Basis Function (RBF) Kernel** ($K(x, x') = \exp(-\gamma \|x - x'\|^2)$) actually maps features into an **infinite-dimensional Hilbert space**. We visualize this landscape via the 3D classification confidence landscape:
   $$z = f(x, y)$$

---

## 🛠️ Installation and Setup

1. **Clone the repository** and navigate to the root directory.

2. **Install the dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you plan to render the Manim CE animation (Phase 1), ensure you have [FFmpeg](https://ffmpeg.org/) installed on your operating system.*

---

## 🚀 How to Run

### Phase 1: Render Manim Concept Animation
To generate the high-quality concept video explaining the 2D to 3D mapping:
```bash
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
```
*Use `-pql` instead of `-pqh` for a fast, low-resolution draft render.*

### Phase 2: Run Mathematical Verification
To verify the RBF SVM fit, plot decision contours, margins, and support vectors:
```bash
python phase2_rbf_decision_surface.py
```
This saves a high-resolution plot to `outputs/verification_plot.png`.

### Phase 3: Start Streamlit Interactive App
To run the interactive web app locally:
```bash
streamlit run phase3_streamlit_app.py
```
This opens a browser tab with the dashboard, allowing you to tweak parameters ($C$, $\gamma$, noise, degree) in real-time.

---

## 🎨 Visual Guidelines & Consistency
* **Inner Class (Class 0):** Blue
* **Outer Class (Class 1):** Red
* **Decision Boundary ($f(x, y) = 0$):** Yellow
* **Support Vectors:** Outlined in Gold
* **Margins ($f(x, y) = \pm 1$):** Dashed slate-gray lines
