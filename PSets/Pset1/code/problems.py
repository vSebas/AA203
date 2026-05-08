#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np


def f(x: np.ndarray, q: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * x.T @ q @ x - b.T @ x

def grad_f(x: np.ndarray, q: np.ndarray, b: np.ndarray) -> np.ndarray:
    return q @ x - b

def gd_constant(x0: np.ndarray, q: np.ndarray,
                 b: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    
    x = x0.astype(float).copy()
    traj = [x.copy()]
    vals = [f(x, q, b)]
    tol = 1e-4

    while np.linalg.norm(grad_f(x, q, b)) > tol:
        g = grad_f(x, q, b)
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("non-finite gradient in gd_constant")

        x_next = x - eta * g
        if not np.all(np.isfinite(x_next)):
            raise FloatingPointError("non-finite iterate in gd_constant")
        if np.linalg.norm(x_next) > 1e12:
            raise FloatingPointError(
                "gd_constant diverged before convergence; iterate norm exceeded 1e12"
            )

        x = x_next
        traj.append(x.copy())
        vals.append(f(x, q, b))
    return np.array(traj), np.array(vals)

def gd_exact_line_search(x0: np.ndarray,
                         q: np.ndarray,
                         b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    x = x0.astype(float).copy()
    traj = [x.copy()]
    vals = [f(x, q, b)]
    tol = 1e-4

    while np.linalg.norm(grad_f(x, q, b)) > tol:
        d = -grad_f(x, q, b)
        if not np.all(np.isfinite(d)):
            raise FloatingPointError("non-finite search direction in gd_exact_line_search")
        denom = d.T @ q @ d
        if np.isclose(denom, 0.0):
            break
        eta = (d.T @ d) / denom
        x_next = x + eta * d
        if not np.all(np.isfinite(x_next)):
            raise FloatingPointError("non-finite iterate in gd_exact_line_search")

        x = x_next

        traj.append(x.copy())
        vals.append(f(x, q, b))

    return np.array(traj), np.array(vals)

def plot_iteration_arrows(ax, traj: np.ndarray, color: str) -> None:
    for x_start, x_end in zip(traj[:-1], traj[1:]):
        ax.annotate(
            "",
            xy=x_end,
            xytext=x_start,
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linewidth=0.7,
                shrinkA=3,
                shrinkB=3,
                mutation_scale=9,
                alpha=0.75,
            ),
        )

def run(gamma: float, x0: np.ndarray) -> None:
    q = np.diag([1.0, gamma])
    b = np.zeros(2)
    eta_const = 0.15

    traj_const, vals_const = gd_constant(x0, q, b, eta_const)
    traj_ls, vals_ls = gd_exact_line_search(x0, q, b)

    print(f"\n=== gamma={gamma}, x0={x0.tolist()} ===")
    print("x* = [0, 0]")
    print(f"constant eta={eta_const:.6f}, final x={traj_const[-1]}")
    print(f"exact line search final x={traj_ls[-1]}")
    print("first 5 exact-line-search iterates:")
    for i, x in enumerate(traj_ls[:5]):
        print(f"  k={i:2d}: {x}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(traj_const[:, 0], traj_const[:, 1], "o-", color="C0",
                 linewidth=0.8, markersize=3, label="constant")
    axes[0].plot(traj_ls[:, 0], traj_ls[:, 1], "s-", color="C1",
                 linewidth=0.8, markersize=3, label="exact")
    plot_iteration_arrows(axes[0], traj_const, "C0")
    plot_iteration_arrows(axes[0], traj_ls, "C1")
    axes[0].plot(0, 0, "k*", ms=10, label="x*")
    axes[0].set_title(f"trajectory, gamma={gamma}")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(vals_const, "o-", linewidth=0.8, markersize=3, label="constant")
    axes[1].semilogy(vals_ls, "s-", linewidth=0.8, markersize=3, label="exact")
    axes[1].set_title(f"objective, x0={x0.tolist()}")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("f(x)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = f"p1_gamma{int(gamma)}_x0_{int(x0[0])}_{int(x0[1])}.png"
    out_path = os.path.join(os.path.dirname(__file__), out)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved plot: {out_path}")

def p1() -> None:
    initial_cond = [np.array([5.0, 1.0]), np.array([1.0, 5.0])]
    for gamma in [10.0, 1.0]:
        for x0 in initial_cond:
            run(gamma, x0)

def p2() -> None:
    T = 20
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [1.0]])

    Q = np.eye(2)
    R = np.eye(1)
    Q_T = 10.0 * np.eye(2)

    x0 = np.array([1.0, 0.0])

    n = A.shape[0]
    m = B.shape[1]
    u0 = np.zeros(m * T)

    ######## Construct LQR as a QP

    # nu maps x0 to the stacked state vector X = [x1; ...; xT].
    nu = np.zeros((n * T, n))
    # V maps U = [u0; ...; u_{T-1}] to X = [x1; ...; xT].
    V = np.zeros((n * T, m * T))

    a_powers = [np.eye(n)]
    for _ in range(T):
        a_powers.append(a_powers[-1] @ A)

    for t in range(T):
        nu[t * n:(t + 1) * n, :] = a_powers[t + 1]
        for k in range(t + 1):
            sub_matrix = a_powers[t - k] @ B
            V[t * n:(t + 1) * n, k * m:(k + 1) * m] = sub_matrix

    H = np.zeros((n * T, n * T))
    for t in range(T - 1):
        H[t * n:(t + 1) * n, t * n:(t + 1) * n] = Q
    H[(T - 1) * n:T * n, (T - 1) * n:T * n] = Q_T

    N = np.zeros((m * T, m * T))
    for t in range(T):
        N[t * m:(t + 1) * m, t * m:(t + 1) * m] = R

    Q_tilde = 2.0 * (V.T @ H @ V + N)
    b_tilde = -2.0 * V.T @ H @ nu @ x0

    eigvals = np.linalg.eigvalsh(Q_tilde)
    eta_const = 1e-5
    traj_const, vals_const = gd_constant(u0, Q_tilde, b_tilde, eta_const)

    u_star = traj_const[-1]
    x_stack = nu @ x0 + V @ u_star

    J_star = vals_const[-1]
    J_lqr = x0.T @ Q @ x0
    for t in range(T - 1):
        x_t = x_stack[t * n:(t + 1) * n]
        u_t = u_star[t * m:(t + 1) * m]
        J_lqr += x_t.T @ Q @ x_t + u_t.T @ R @ u_t
    x_T = x_stack[(T - 1) * n:T * n]
    J_lqr += x_T.T @ Q_T @ x_T

    print("\n=== Problem 2: Gradient Descent on Condensed QP ===")
    print(f"condition number = {eigvals[-1] / eigvals[0]:.6e}")
    print(f"eta_const = {eta_const:.6e}")
    print(f"iterations = {len(traj_const) - 1}")
    print(f"final objective = {J_star:.6f}")
    print(f"LQR cost J(u*) = {J_lqr:.6f}")
    print(f"||grad J(u*)|| = {np.linalg.norm(grad_f(u_star, Q_tilde, b_tilde)):.6e}")
    print(f"first 5 controls = {u_star[:5]}")
    print(f"terminal state x_T = {x_T}")


def main() -> None:
    p1()
    p2()

if __name__ == "__main__":
    main()
