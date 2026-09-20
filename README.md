# solar-composite-tools
High-performance Python utilities for combining and compositing full-disk solar imagery (e.g., ASO-S/LST/SDI, SDO/AIA, GOES-R/SUVI) with coronagraph observations (e.g., ASO-S/LST/SCI, SOHO/LASCO).

## 🚀 Key Advantages & Design Focus

While official SunPy tutorials demonstrate full coordinate reprojection for overlaying coronal images (see [SunPy LASCO Overlay Example](https://docs.sunpy.org/en/stable/generated/gallery/plotting/lasco_overlay.html)), pixel-by-pixel WCS reprojection can be **computationally expensive and slow**, especially when processing large time series or high-resolution datasets.

This package offers a lightweight, fast alternative optimized for rapid visual compositing, multi-wavelength alignment, and generating publication-ready composite maps without heavy computational overhead.

---

## 📁 Repository Structure

* **`solar_composite_tools.py`**: Core library containing map alignment, scaling, and compositing functions.
* **`examples.ipynb`**: Interactive Jupyter Notebook with three step-by-step examples demonstrating full-disk + coronagraph overlays.
* **`requirements.txt`**: Standard dependencies list for one-step environment setup.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/xuejcak/solar-composite-tools.git](https://github.com/xuejcak/solar-composite-tools.git)
   cd solar-composite-tools
