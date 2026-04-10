"""
PINN for the 1D Harmonic Oscillator
=====================================
Equation : d²u/dt² + ω²·u = 0
ICs      : u(0) = 1,  u'(0) = 0
Exact    : u(t) = cos(ω·t)

The network u_θ(t) is trained by minimising:
    L = L_ic  +  λ · L_physics

where
    L_ic      = ||u_θ(0) - 1||² + ||u'_θ(0) - 0||²
    L_physics = (1/N_f) Σ ||u''_θ(t_i) + ω²·u_θ(t_i)||²
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
OMEGA = 2.0        # angular frequency
T_MAX = 2.0        # time domain [0, T_MAX]
N_COLLOC = 1000    # number of collocation (physics) points
N_IC = 1           # number of IC points (just t=0)
HIDDEN_LAYERS = [32, 32, 32]
LEARNING_RATE = 1e-3
N_EPOCHS = 5000
LAMBDA_PHYSICS = 1.0   # weight for physics loss relative to IC loss
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """Fully-connected network mapping t -> u(t)."""

    def __init__(self, hidden_layers):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)


# ---------------------------------------------------------------------------
# Helper: compute first and second derivatives via autograd
# ---------------------------------------------------------------------------
def derivatives(model, t):
    """Return u, u', u'' for input tensor t (requires_grad=True)."""
    u = model(t)
    u_t = torch.autograd.grad(u, t,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t,
                                grad_outputs=torch.ones_like(u_t),
                                create_graph=True)[0]
    return u, u_t, u_tt


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def loss_physics(model, t_f):
    """Residual of ODE: u_tt + ω²·u = 0."""
    _, _, u_tt = derivatives(model, t_f)
    u = model(t_f)
    residual = u_tt + OMEGA ** 2 * u
    return torch.mean(residual ** 2)


def loss_ic(model, t0):
    """Initial condition: u(0)=1, u'(0)=0."""
    u, u_t, _ = derivatives(model, t0)
    loss = (u - 1.0) ** 2 + (u_t - 0.0) ** 2
    return torch.mean(loss)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    model = PINN(HIDDEN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Collocation points (physics) — random in [0, T_MAX]
    t_f = torch.FloatTensor(N_COLLOC, 1).uniform_(0.0, T_MAX).requires_grad_(True)

    # IC point
    t0 = torch.zeros(N_IC, 1, requires_grad=True)

    print(f"Training PINN for harmonic oscillator  (ω={OMEGA}, T=[0,{T_MAX}])")
    print(f"  Architecture : {HIDDEN_LAYERS}")
    print(f"  Collocation  : {N_COLLOC} points")
    print(f"  Epochs       : {N_EPOCHS}")
    print()

    for epoch in range(1, N_EPOCHS + 1):
        optimizer.zero_grad()

        l_ic = loss_ic(model, t0)
        l_ph = loss_physics(model, t_f)
        loss = l_ic + LAMBDA_PHYSICS * l_ph

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} | total={loss.item():.4e} "
                  f"| L_ic={l_ic.item():.4e} | L_phys={l_ph.item():.4e}")

    return model


# ---------------------------------------------------------------------------
# Evaluation & plotting
# ---------------------------------------------------------------------------
def evaluate(model):
    t_test = np.linspace(0, T_MAX, 400)
    t_tensor = torch.FloatTensor(t_test.reshape(-1, 1))

    with torch.no_grad():
        u_pred = model(t_tensor).numpy().flatten()

    u_exact = np.cos(OMEGA * t_test)

    l2_error = np.sqrt(np.mean((u_pred - u_exact) ** 2))
    print(f"\nL2 error vs exact solution: {l2_error:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_test, u_exact, "b-", linewidth=2, label="Exact: cos(ωt)")
    axes[0].plot(t_test, u_pred, "r--", linewidth=2, label="PINN")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u(t)")
    axes[0].set_title("Harmonic Oscillator — PINN vs Exact")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_test, np.abs(u_pred - u_exact), "g-", linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|u_PINN - u_exact|")
    axes[1].set_title(f"Absolute Error  (L2={l2_error:.2e})")
    axes[1].grid(True)

    plt.tight_layout()
    out_path = "harmonic_oscillator_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trained_model = train()
    evaluate(trained_model)
