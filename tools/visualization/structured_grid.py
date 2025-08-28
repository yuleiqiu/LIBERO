import numpy as np
import matplotlib.pyplot as plt

# Structured grid sampling
Nx = Ny = 10
x = np.linspace(-0.4, 0.1, Nx)
y = np.linspace(-0.4, 0.1, Ny)
print(f"{x=}")
X, Y = np.meshgrid(x, y)
print(f"{X=}")
points = np.column_stack([X.ravel(), Y.ravel()])

# Plotting
plt.figure()
plt.scatter(points[:, 0], points[:, 1], marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Structured Grid Sampling in [-0.4, 0.1] x [-0.4, 0.1]')
plt.grid(True)
# plt.show()

# Save the figure
plt.savefig('structured_grid_sampling.png')
