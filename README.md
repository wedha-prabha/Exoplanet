# 🌌 A World Away — AI-Powered Exoplanet Detection

## Overview
**A World Away** is an AI-powered exoplanet detection pipeline that combines a hybrid **CNN + Transformer model** with interactive **light curve visualization**. It analyzes light curve data from space telescopes like **Kepler** and **TESS**, predicting planetary transits and estimating key parameters such as period, depth, and duration.  

The pipeline supports **multi-task learning**, enabling simultaneous classification (transit detection) and regression (planetary parameters). It also supports **synthetic and user-uploaded datasets**, enhancing model robustness and discovery potential.

---

## Features
- Hybrid **CNN + Transformer architecture** for accurate light curve analysis.  
- **Multi-task learning:** simultaneous transit detection & parameter regression.  
- **Interactive Streamlit dashboard** for visualization of light curves and predictions.  
- Upload custom CSV/TXT flux data for instant analysis.  
- Estimate **planet radius** using transit depth and star radius input.  
- Supports **synthetic dataset generation** for testing.  
- **Checkpoint support:** load a trained model (`best_model.pth`) for real predictions.

---
