import torch
import matplotlib.pyplot as plt

params = {
    "sigma": 10.0,
    "beta": 8.0 / 3.0,
    "rho": 28.0,
    # "initial_state": torch.rand(dataset_num, 3),
    "dt": 0.01,
    "T": 50.0,
}

dataset_num = 512
sigma = torch.linspace(6,14,8)
beta = torch.linspace(1,4,8)
rho = torch.linspace(20,36,8)
sigma, beta, rho = torch.meshgrid(sigma, beta, rho)
sigma = sigma.reshape(-1,1)
beta = beta.reshape(-1,1)
rho = rho.reshape(-1,1)
# beta = 8/3
# rho = 28
initial_data = torch.rand(dataset_num, 3)
dt = 0.01
T = 50.0
Nt = int(T/dt)

# Define the solver
def lorenz_eq(sigma, beta, rho, dt):
    def lorenz_solver(u_n):
        x, y, z = u_n.split(1, dim=-1)

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        u_n_plus_1 = u_n + torch.cat([dx_dt, dy_dt, dz_dt], dim=-1) * dt

        return u_n_plus_1

    return lorenz_solver

# Iterate to get the trajectory
solver = lorenz_eq(sigma, beta, rho, dt)
u = initial_data
results = []
for i in range(Nt):
    u = solver(u)
    results.append(u)

# Simplify the results
results = torch.stack(results, dim=0).numpy()
gap = 5
# results = results[::gap]

# Visualize the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

idx=0 #specify the number of point in dataset you want to visualize
x = results[:,idx,0]
y = results[:,idx,1]
z = results[:,idx,2]
ax.plot(x, y, z)
fig.savefig('data_full.png')

import numpy as np
allparams = torch.cat([sigma, beta, rho], dim=-1)

# Save the dataset
np.savez('lorenz_dataset3_1_full',u=results,params = allparams.numpy())