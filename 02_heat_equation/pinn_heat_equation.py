"""
PINN for the 1D Heat (Diffusion) Equation
==========================================
Equation : ∂u/∂t = α·∂²u/∂x²,   x∈[0,L], t∈[0,T]
BCs      : u(0,t) = 0,  u(L,t) = 0
IC       : u(x,0) = sin(π·x/L)
Exact    : u(x,t) = exp(-α·(π/L)²·t) · sin(π·x/L)

Loss function:
    L = L_ic  +  L_bc  +  λ·L_physics

where the physics residual enforces the PDE at collocation points and
the IC/BC terms enforce the constraints on the boundary of the domain.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
ALPHA = 0.1        # thermal diffusivity
L = 1.0            # domain length
T_MAX = 1.0        # time horizon

N_COLLOC = 5000    # interior collocation points for PDE residual
N_BC = 200         # boundary condition points (per boundary)
N_IC = 500         # initial condition points

HIDDEN_LAYERS = [64, 64, 64]
LEARNING_RATE = 1e-3
N_EPOCHS = 8000
LAMBDA_PHYSICS = 1.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """Fully-connected network mapping (x, t) -> u(x, t)."""

    def __init__(self, hidden_layers):
        super().__init__()
        layers = []
        in_dim = 2  # inputs: x and t
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------
def pde_residual(model, x_f, t_f):
    """Residual of heat equation: u_t - α·u_xx = 0."""
    u = model(x_f, t_f)

    u_t = torch.autograd.grad(u, t_f,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f,
                                grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    residual = u_t - ALPHA * u_xx
    return torch.mean(residual ** 2)


# ---------------------------------------------------------------------------
# Boundary / initial condition losses
# ---------------------------------------------------------------------------
def loss_bc(model, t_bc):
    """Dirichlet BCs: u(0,t)=0 and u(L,t)=0."""
    x_left = torch.zeros_like(t_bc)
    x_right = torch.full_like(t_bc, L)

    u_left = model(x_left, t_bc)
    u_right = model(x_right, t_bc)

    return torch.mean(u_left ** 2) + torch.mean(u_right ** 2)


def loss_ic(model, x_ic):
    """IC: u(x,0) = sin(π·x/L)."""
    t_zero = torch.zeros_like(x_ic)
    u_pred = model(x_ic, t_zero)
    u_exact = torch.sin(np.pi * x_ic / L)
    return torch.mean((u_pred - u_exact) ** 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    model = PINN(HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Interior collocation points
    x_f = torch.FloatTensor(N_COLLOC, 1).uniform_(0.0, L).requires_grad_(True)
    t_f = torch.FloatTensor(N_COLLOC, 1).uniform_(0.0, T_MAX).requires_grad_(True)

    # Boundary condition points
    t_bc = torch.FloatTensor(N_BC, 1).uniform_(0.0, T_MAX)

    # Initial condition points
    x_ic = torch.FloatTensor(N_IC, 1).uniform_(0.0, L)

    print(f"Training PINN for 1D Heat Equation  (α={ALPHA}, L={L}, T=[0,{T_MAX}])")
    print(f"  Architecture      : {HIDDEN_LAYERS}")
    print(f"  Collocation pts   : {N_COLLOC}")
    print(f"  BC points / side  : {N_BC}")
    print(f"  IC points         : {N_IC}")
    print(f"  Epochs            : {N_EPOCHS}")
    print()

    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()

        l_ph = pde_residual(model, x_f, t_f)
        l_bc = loss_bc(model, t_bc)
        l_ic = loss_ic(model, x_ic)

        loss = l_ic + l_bc + LAMBDA_PHYSICS * l_ph
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} | total={loss.item():.4e} "
                  f"| L_ic={l_ic.item():.4e} "
                  f"| L_bc={l_bc.item():.4e} "
                  f"| L_phys={l_ph.item():.4e}")

    return model


# ---------------------------------------------------------------------------
# Evaluation & plotting
# ---------------------------------------------------------------------------
def evaluate(model):
    x_vals = np.linspace(0, L, 100)
    t_snapshots = [0.0, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colours = ["royalblue", "darkorange", "green", "red"]
    l2_errors = []

    for t_val, col in zip(t_snapshots, colours):
        x_tensor = torch.FloatTensor(x_vals.reshape(-1, 1))
        t_tensor = torch.full_like(x_tensor, t_val)

        with torch.no_grad():
            u_pred = model(x_tensor, t_tensor).numpy().flatten()

        u_exact = np.exp(-ALPHA * (np.pi / L) ** 2 * t_val) * np.sin(np.pi * x_vals / L)
        err = np.sqrt(np.mean((u_pred - u_exact) ** 2))
        l2_errors.append(err)

        axes[0].plot(x_vals, u_exact, color=col, linewidth=2,
                     label=f"Exact t={t_val}")
        axes[0].plot(x_vals, u_pred, color=col, linestyle="--", linewidth=1.5,
                     label=f"PINN  t={t_val}")
        axes[1].plot(x_vals, np.abs(u_pred - u_exact), color=col, linewidth=2,
                     label=f"t={t_val}  L2={err:.2e}")

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x, t)")
    axes[0].set_title("Heat Equation — PINN vs Exact")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|u_PINN - u_exact|")
    axes[1].set_title("Absolute Error at Each Snapshot")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    out_path = "heat_equation_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    print(f"Mean L2 error across snapshots: {np.mean(l2_errors):.4e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)
