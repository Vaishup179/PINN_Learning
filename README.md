# PINN_Learning

This repository is dedicated to learning **Physics-Informed Neural Networks (PINNs)**, connecting my physics research (experimental results) with computational methods.

---

## Table of Contents

- [What are PINNs?](#what-are-pinns)
- [Why PINNs?](#why-pinns)
- [Repository Structure](#repository-structure)
- [Examples Covered](#examples-covered)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [References](#references)

---

## What are PINNs?

**Physics-Informed Neural Networks (PINNs)** are a class of deep neural networks that incorporate physical laws — expressed as differential equations — directly into their training process. Rather than learning purely from data, PINNs embed the governing equations of a physical system into the loss function, acting as a soft constraint that guides the network toward physically consistent solutions.

A PINN is trained by minimising a composite loss function:

```
L_total = L_data + L_physics + L_boundary + L_initial
```

where:
- **L_data** — mean-squared error against any available measurement/observation data
- **L_physics** — residual of the governing PDE/ODE evaluated at collocation points (no labels needed)
- **L_boundary** — error on boundary conditions
- **L_initial** — error on initial conditions

The physics residual is computed by differentiating the network output with respect to its inputs using **automatic differentiation**, so no finite-difference approximation is required.

---

## Why PINNs?

| Traditional Numerical Solvers | PINNs |
|-------------------------------|-------|
| Require mesh generation | Mesh-free |
| High cost for high-dimensional problems | Scales better to high dimensions |
| Need complete knowledge of the system | Can assimilate sparse experimental data |
| Solve forward problems efficiently | Solve both **forward** and **inverse** problems |
| Fixed geometry/domain | Flexible domain via sampling |

PINNs are especially useful when:
- Experimental data is sparse or noisy
- The PDE has unknown parameters (inverse problem)
- The domain geometry makes meshing difficult
- A differentiable, continuous surrogate model is needed

---

## Repository Structure

```
PINN_Learning/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── 01_harmonic_oscillator/
│   ├── pinn_harmonic_oscillator.py  # PINN for a 1D harmonic oscillator (ODE)
│   └── README.md                    # Example-specific notes
└── 02_heat_equation/
    ├── pinn_heat_equation.py        # PINN for the 1D heat equation (PDE)
    └── README.md                    # Example-specific notes
```

---

## Examples Covered

### 1. 1D Harmonic Oscillator (ODE)
**Equation:** `d²u/dt² + ω²u = 0`,  `u(0) = 1`,  `u'(0) = 0`

The harmonic oscillator is an excellent first PINN example because:
- The analytical solution is known: `u(t) = cos(ωt)`
- It is a second-order ODE — requires computing second-order derivatives via automatic differentiation
- It demonstrates how initial conditions are enforced in the loss

### 2. 1D Heat Equation (PDE)
**Equation:** `∂u/∂t = α ∂²u/∂x²`

With boundary conditions `u(0,t) = u(L,t) = 0` and initial condition `u(x,0) = sin(πx/L)`.

This example shows:
- How spatial and temporal derivatives are handled simultaneously
- How Dirichlet boundary conditions and initial conditions are incorporated
- Comparison with the analytical solution `u(x,t) = exp(-α(π/L)²t) sin(πx/L)`

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with autograd support)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run an example

```bash
# 1D Harmonic Oscillator
python 01_harmonic_oscillator/pinn_harmonic_oscillator.py

# 1D Heat Equation
python 02_heat_equation/pinn_heat_equation.py
```

Each script will train the PINN, print the training progress, and save a comparison plot of the PINN solution versus the analytical solution.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

| Package | Purpose |
|---------|---------|
| `torch` | Neural network and automatic differentiation |
| `numpy` | Numerical utilities |
| `matplotlib` | Plotting results |
| `scipy` | Reference analytical solutions |

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045
2. Karniadakis, G. E., et al. (2021). **Physics-informed machine learning.** *Nature Reviews Physics*, 3(6), 422–440. https://doi.org/10.1038/s42254-021-00314-5
3. Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). **Artificial neural networks for solving ordinary and partial differential equations.** *IEEE Transactions on Neural Networks*, 9(5), 987–1000.
