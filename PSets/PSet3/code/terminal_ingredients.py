import numpy as np
import matplotlib.pyplot as plt

import cvxpy as cp

def generate_ellipsoid_points(M, num_points=100):
    """Generate points on a 2-D ellipsoid.
    The ellipsoid is described by the equation
    `{ x | x.T @ inv(M) @ x <= 1 }`,
    where `inv(M)` denotes the inverse of the matrix argument `M`.
    The returned array has shape (num_points, 2).
    """
    L = np.linalg.cholesky(M)
    θ = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack([np.cos(θ), np.sin(θ)])
    x = u @ L.T
    return x

def semi_def_program():
    A = np.array([
        [0.9, 0.6],
        [0.0, 0.8]
    ])
    B = np.array([
        [0.0],
        [1.0]
    ])

    rx = 5.0
    n = A.shape[0]
    M = cp.Variable((n, n), symmetric=True)

    block = cp.bmat([
        [M, A @ M],
        [M @ A.T, M]
    ])

    constraints = [
        M >> 0,
        block >> 0,
        rx**2 * np.eye(n) - M >> 0
    ]

    objective = cp.Maximize(cp.log_det(M))
    prob = cp.Problem(objective,constraints)

    prob.solve()

    # print(M.value)
    W = np.linalg.inv(M.value)
    print("W=")
    print(np.round(W, 3))

    X_T = generate_ellipsoid_points(M.value)

    X_T_next = X_T @ A.T

    # X = {x | ||x||_2 <= rx}
    theta = np.linspace(0, 2*np.pi, 100)
    X_pts = np.column_stack([
        rx * np.cos(theta),
        rx * np.sin(theta)
    ])

    plt.figure(figsize=(7, 7))

    plt.plot(X_pts[:, 0], X_pts[:, 1], label=r"$X=\{x:\|x\|_2 \leq r_x\}$")
    plt.plot(X_T[:, 0], X_T[:, 1], label=r"$X_T$")
    plt.plot(X_T_next[:, 0], X_T_next[:, 1], label=r"$A X_T$")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    plt.title(r"Ellipsoids: $A X_T \subseteq X_T \subseteq X$")
    plt.savefig("ellipsoids.png")

    return A, B, rx, M.value, W

def setup_mpc(A, B, W, N=4, rx=5.0, ru=1.0):
    n, m = B.shape
    Q = np.eye(n)
    R = np.eye(m)
    P = np.eye(n)

    x0 = cp.Parameter(n)
    x = cp.Variable((N + 1, n))
    u = cp.Variable((N, m))

    cost = 0.0
    constraints = [x[0, :] == x0]

    for t in range(N):
        cost += cp.quad_form(x[t, :], Q) + cp.quad_form(u[t, :], R)
        constraints += [
            x[t + 1, :] == A @ x[t, :] + B @ u[t, :],
            cp.norm(x[t, :], 2) <= rx,
            cp.norm(u[t, :], 2) <= ru,
        ]

    cost += cp.quad_form(x[N, :], P)
    constraints += cp.quad_form(x[N, :], W) <= 1

    prob = cp.Problem(cp.Minimize(cost), constraints)
    return prob, x0, x, u

def simulate_mpc(A, B, M, W, N=4, T=15, rx=5.0):
    prob, x0_param, x_var, u_var = setup_mpc(A, B, W, N=N, rx=rx)

    x = np.array([0.0, -4.5])
    actual_states = [x.copy()]
    planned_states = []
    applied_controls = []

    for _ in range(T):
        x0_param.value = x
        prob.solve()

        x_plan = x_var.value
        u_plan = u_var.value
        planned_states.append(x_plan.copy())
        applied_controls.append(u_plan[0, 0])

        x = A @ x + B @ u_plan[0, :]
        actual_states.append(x.copy())

    actual_states = np.array(actual_states)
    applied_controls = np.array(applied_controls)

    plot_mpc_trajectories(A, M, actual_states, planned_states, rx)
    plot_control_trajectory(applied_controls)

def plot_mpc_trajectories(A, M, actual_states, planned_states, rx):
    X_T = generate_ellipsoid_points(M)
    X_T_next = X_T @ A.T

    theta = np.linspace(0, 2*np.pi, 100)
    X_pts = np.column_stack([
        rx * np.cos(theta),
        rx * np.sin(theta)
    ])

    plt.figure(figsize=(7, 7))
    plt.plot(X_pts[:, 0], X_pts[:, 1], label=r"$X=\{x:\|x\|_2 \leq r_x\}$")
    plt.plot(X_T[:, 0], X_T[:, 1], label=r"$X_T$")
    plt.plot(X_T_next[:, 0], X_T_next[:, 1], label=r"$A X_T$")

    for x_plan in planned_states:
        plt.plot(x_plan[:, 0], x_plan[:, 1], "--*", color="k", alpha=0.35)

    plt.plot(actual_states[:, 0], actual_states[:, 1], "-o", label="actual trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    plt.title("Closed-loop MPC trajectory with planned trajectories")
    plt.savefig("terminal_mpc_trajectory.png")

def plot_control_trajectory(applied_controls):
    plt.figure()
    plt.plot(np.arange(len(applied_controls)), applied_controls, "-o")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
    plt.axhline(-1.0, color="k", linestyle="--", linewidth=1)
    plt.grid(True)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u_t$")
    plt.title("Applied MPC control trajectory")
    plt.savefig("terminal_mpc_control.png")

def main():
    A, B, M, W = semi_def_program()
    simulate_mpc(A, B, M, W)

if __name__ == "__main__":
    main()
