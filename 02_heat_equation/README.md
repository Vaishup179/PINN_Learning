# Example 2: 1D Heat Equation (PDE)

## Problem

Solve the 1D heat (diffusion) equation:

```
∂u/∂t = α ∂²u/∂x²,   x ∈ [0, L],  t ∈ [0, T]
```

with boundary and initial conditions:

```
u(0, t) = 0            (left Dirichlet BC)
u(L, t) = 0            (right Dirichlet BC)
u(x, 0) = sin(πx/L)   (initial condition)
```

Analytical solution: `u(x, t) = exp(-α(π/L)² t) · sin(πx/L)`

## PINN Approach

A fully-connected neural network `u_θ(x, t)` takes two inputs and is
trained to satisfy:

1. **Physics residual** at N_f interior collocation points:
   ```
   f(x, t) = u_t - α u_xx  →  minimise ||f||²
   ```
2. **Boundary condition loss** at N_b points on x=0 and x=L:
   ```
   ||u_θ(0, t)||² + ||u_θ(L, t)||²
   ```
3. **Initial condition loss** at N_i points on t=0:
   ```
   ||u_θ(x, 0) - sin(πx/L)||²
   ```

All derivatives are obtained via PyTorch autograd.

## Running

```bash
python pinn_heat_equation.py
```

Produces `heat_equation_result.png` with solution snapshots and error plots.
