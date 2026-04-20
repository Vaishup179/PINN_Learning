# Example 1: 1D Harmonic Oscillator (ODE)

## Problem

Solve the initial value problem for the undamped harmonic oscillator:

```
d²u/dt² + ω²u = 0,   t ∈ [0, T]
u(0)  = 1
u'(0) = 0
```

Analytical solution: `u(t) = cos(ωt)`

## PINN Approach

A fully-connected neural network `u_θ(t)` is trained to satisfy:

1. **Physics residual** at N_f collocation points sampled in [0, T]:
   ```
   f(t) = u_tt + ω²u  →  minimise ||f||²
   ```
2. **Initial condition loss**:
   ```
   ||u_θ(0) - 1||²  +  ||u'_θ(0) - 0||²
   ```

Both residuals are computed via PyTorch's `autograd`.

## Running

```bash
python pinn_harmonic_oscillator.py
```

Produces `harmonic_oscillator_result.png` comparing the PINN with the exact solution.
