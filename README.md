# Neuro-Physical Inverter for Magnetotelluric Data

This repository contains code accompanying a research study on a **Neuro-Physical Inverter (NPI)**
for 1D magnetotelluric (MT) data. The method combines:

- 1D MT forward modeling,
- ensemble-approximated conditional Gaussian Processes (EnsCGP),
- neural residual refinement using a ResNet-based architecture.

The code is organized as a Python package and includes example scripts and notebooks
demonstrating forward modeling, ensemble-based inversion, and neural refinement.

---

## Installation

We recommend using a conda environment.

```bash
conda create -n npi-mt python=3.10
conda activate npi-mt
pip install -e .
