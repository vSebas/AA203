import numpy as np
import matplotlib.pyplot as plt

ACTIONS = np.array([  [0,1],  # up
                      [0,-1], # down
                      [-1,0], # left
                      [1,0],  # right
                   ])

def idx_to_state(s,n=20):
    x1 = s % n
    x2 = s // n
    return np.array([x1,x2])

def move(x,a,n=20):
    x_next = x+a
    x_next = np.clip(x_next,0,n-1)

    return x_next[1]*n + x_next[0]

def value_iteration(n,eps=1e-6):
    sigma = 10
    gamma = 0.95
    x_eye = np.array([15,15])
    x_goal = np.array([19,9])
    num_states = n*n
    V = np.zeros(n*n)
    opt_policy = np.zeros(n*n, dtype=int)

    T = np.zeros((n*n,4,n*n))   # current, action, next
    R = np.zeros(num_states)
    R[x_goal[1] * n + x_goal[0]] = 1

    for s in range(num_states):
        # storm_influence
        x = idx_to_state(s)
        omega = np.exp(-np.linalg.norm(x - x_eye)**2 / (2 * sigma**2))

        # Build transition
        for a in range(4):
            # specified action
            next_s = move(x,ACTIONS[a])
            T[s,a, next_s] += 1 - omega

            # random action
            for a_rand in range(4):
                next_s_rand = move(x,ACTIONS[a_rand])
                T[s,a,next_s_rand] += omega/4

    diff = np.inf
    while diff > eps:
        V_new = np.zeros_like(V)

        for s in range(num_states):
            q = np.zeros(4)

            for a in range(4):
                q[a] =  np.sum(T[s,a,:] * (R + gamma*V))

            opt_policy[s] = np.argmax(q)
            V_new[s] = q[opt_policy[s]]#np.max(q)

        diff = np.max(np.abs(V_new - V))
        V = V_new

    return V, opt_policy

def simulate_MDP(policy,n):
    sigma = 10
    x_eye = np.array([15,15])
    x_init = np.array([0,19])
    time_steps = 100
    path = [x_init.copy()]

    x = x_init.copy()
    for _ in range(time_steps):
        s = x[1] * n + x[0]
        action = policy[s]
        omega = np.exp(-np.linalg.norm(x - x_eye)**2 / (2 * sigma**2))

        if np.random.rand() < omega:
            action = np.random.randint(4)

        next_s = move(x,ACTIONS[action],n)
        x = idx_to_state(next_s,n)
        path.append(x.copy())

    return np.array(path)

def plot_heatmaps(V,policy,path,n):
    V_grid = V.reshape((n,n))
    policy_grid = policy.reshape((n,n))

    plt.figure()
    plt.imshow(V_grid, origin='lower', extent=[0,n-1,0,n-1])
    plt.colorbar(label='Value')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Optimal Value Function')
    plt.tight_layout()
    # plt.show()
    plt.savefig("optimal_value_heatmap.png")

    plt.figure()
    plt.imshow(policy_grid, origin='lower', extent=[0,n-1,0,n-1], vmin=0, vmax=3)
    plt.colorbar(label='Action: up=0, down=1, left=2, right=3')
    plt.plot(path[:,0], path[:,1], color='white', linewidth=2, marker='o', markersize=2)
    plt.scatter(path[0,0], path[0,1], color='lime', edgecolor='black', label='start')
    plt.scatter(19,9, color='yellow', edgecolor='black', label='goal')
    plt.scatter(path[-1,0], path[-1,1], color='red', edgecolor='black', label='end')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Optimal Policy and Simulated Path')
    plt.legend()
    plt.tight_layout()
    plt.savefig("policy_heatmap_path.png")

def main():
    n = 20
    
    # Part A
    V,policy = value_iteration(n)

    # Part B
    path = simulate_MDP(policy,n)

    plot_heatmaps(V,policy,path,n)


if __name__ == '__main__':
    main()
