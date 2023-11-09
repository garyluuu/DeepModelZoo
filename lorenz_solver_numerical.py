import torch

def lorenz_eq(sigma, beta, rho, dt):
    def lorenz_solver(u_n):
        x, y, z = u_n.split(1, dim=-1)

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        u_n_plus_1 = u_n + torch.cat([dx_dt, dy_dt, dz_dt], dim=-1) * dt

        return u_n_plus_1

    return lorenz_solver

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Define initial condition, parameters, and solver
    batch_size = 10
    u_0 = torch.ones(batch_size, 3)
    T = 50.0
    dt = 0.001
    sigma = 10
    beta = 8/3
    rho = 28
    solver1 = lorenz_eq(sigma, beta, rho, dt)

    # Calculate the total number of steps
    num_steps = int(T / dt)

    # Initialize the result tensor
    trajectory = []
    u_t = u_0

    # Iteration
    for i in range(num_steps):
        if i % 20 == 0:
            trajectory.append(u_t)
        u_t = solver1(u_t)

    # Convert trajectory to a tensor
    trajectory = torch.stack(trajectory)

    # Visualize the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for b in range(1):
        ax.scatter(trajectory[:, b, 0].numpy(), trajectory[:, b, 1].numpy(), trajectory[:, b, 2].numpy())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig("lorenz.png")
    