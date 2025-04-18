#!/usr/bin/env python3

"""
Danette_test.py

 Test package

Author: Danette Farnsworth
Date: 2025-04-08
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import Danette

print("####################################################################")
print("Simple Derivative")
print("f(x) = x^2 - 5x")
print("####################################################################")


def f(x):
    return x * (x - 5)


errors = pd.Series(index=(10**-2, 10**-4, 10**-6, 10**-8, 10**-10, 10**-12, 10**-14))
for exponent in errors.index:
    errors[exponent] = Danette.numerical_derivative(f, 1, exponent)

# Plot Error in Numerical Dericative Approx
plt.figure(figsize=(8, 6))
plt.loglog(errors, marker="o", linestyle="-", color="pink")
plt.xlabel("Delta")
plt.ylabel("Absolute Error")
plt.title("Error in Numerical Derivative Approximation")
plt.grid(True, which="both", linestyle="--")
plt.show()


print("####################################################################")
print("Calculated electric potenial of two charges +/-C 10cm apart")
print("Using the Permittivity of Water (F/m)")
print("####################################################################")
print("\n")

# Constants
epsO = 80.0 * 8.85e-12  # Permittivity of Water (F/m)
q = 1e-9  # Charge magnitude (Coulombs)
d = 0.1  # Distance between charges (cm)
grid_size = 0.5  # Grid size (cm)
spacing = 0.005  # Grid spacing (cm)
softening = 1e-1  # Softening parameter to avoid singularities

# Create grid points
x = np.arange(-grid_size / 2, grid_size / 2 + spacing, spacing)
y = np.arange(-grid_size / 2, grid_size / 2 + spacing, spacing)
X, Y = np.meshgrid(x, y)

# Position of the charges
pos_q = (-d / 2, 0)  # Positive charge at (-d/2, 0)
neg_q = (d / 2, 0)  # Negative charge at (d / 2, 0)

# Compute potential due to both charges
phi_plus = Danette.potential(q, *pos_q, X, Y)
phi_minus = Danette.potential(-q, *neg_q, X, Y)

# Total potential
phi_total = phi_plus + phi_minus

# Set symmetric color limits for better visualation
vmax = np.max(np.abs(phi_total))  # Define vmax based on the potential range

# Plot the potential
plt.figure(figsize=(8, 6))
plt.imshow(phi_total, extent=[-grid_size / 2, grid_size / 2, -grid_size / 2, grid_size / 2], origin="lower", cmap="coolwarm")
plt.scatter([pos_q[0], neg_q[0]], [pos_q[1], neg_q[1]], color="black", marker="o", label="Charges")
plt.colorbar(label="Electric Potential (V)")
plt.title("Electric Potential of a Dipole")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.legend()
plt.show()

# Compute electric field components (negative gradient of potential)
Ey, Ex = np.gradient(-phi_total, spacing)  # Note: np.gradient returns [d/dy, d/dx]

# Compute magnitude of the electric field
E_magnitude = np.sqrt(Ex**2 + Ey**2)

# Limit the magnitude to avoid excessive values near charges
E_max = np.percentile(E_magnitude, 75)  # Set max threshold to 95th percentile
scaling_factor = np.clip(E_max / (E_magnitude + softening), 0, 1)
Ex *= scaling_factor
Ey *= scaling_factor

# Subsample for quiver plot (to avoid too many arrows)
step = 4  # Adjust to control arrow density
X_quiver = X[::step, ::step]
Y_quiver = Y[::step, ::step]
Ex_quiver = Ex[::step, ::step]
Ey_quiver = Ey[::step, ::step]

# Plot the electric field vectors using quiver
plt.figure(figsize=(8, 6))
plt.imshow(phi_total, extent=[-grid_size / 2, grid_size / 2, -grid_size / 2, grid_size / 2],
           origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
plt.colorbar(label="Electric Potential (V)")
plt.quiver(X_quiver, Y_quiver, Ex_quiver, Ey_quiver, color="black")
plt.title("Electric Field of a Dipole")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")

# Mark charge locations
plt.scatter([pos_q[0], neg_q[0]], [pos_q[1], neg_q[1]], c=["red", "blue"], marker="o", s=100, label="Charges")

# Plot charges manually with labels
neg_charge = plt.scatter(neg_q[0], neg_q[1], c="blue", marker="o", s=100, label="Negative Charge")
pos_charge = plt.scatter(pos_q[0], pos_q[1], c="red", marker="o", s=100, label="Positive Charge")
plt.legend(handles=[neg_charge, pos_charge])
plt.show()

print("####################################################################")
print("Lorenz Equations / Runge-Kutta 4th order")
print("sigma = 14.0, r = 45.0 to create more chaos (could use 22.0 for less chaos and it might converge")
print("t_end = 100.0")
print("####################################################################")

# Parameters
sigma = 14.0  # changed from 10.0 to 14.0
r = 45.0   # changed from 28.0 to 45.0 for more chaotic, you can also use 22.0 for less chaotic and it might converage to a fixed point
b = 2.0  # changed from 8.0 / 3.0 to 2.0
r0 = np.array([0.0, 1.0, 0.0])  # Initial conditions for x, y, z
t0 = 0.0  # Start time
t_end = 100.0  # changed 50.0 to 100.0  # End time
h = 0.01  # Time step

# Solve the Lorenz system using Runge-Kutta 4th order method
t_values, r_values = Danette.runge_kutta_4(Danette.lorenz, r0, t0, t_end, h, r, sigma, b)

# Extract x, y, z
x_values = r_values[:, 0]
y_values = r_values[:, 1]
z_values = r_values[:, 2]

# Plot y vs. t (Chaotic behavior)
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label="y(t)", color="purple")
plt.xlabel("Time (t)", fontsize=14)
plt.ylabel("y(t)", fontsize=14)
plt.title("Lorenz System: y(t) vs. Time", fontsize=16)
plt.grid(True)
plt.legend()
plt.show()

# Multi-colored: Plot z vs. x (Strange attractor)
plt.figure(figsize=(10, 6))
plt.scatter(x_values, z_values, c=t_values, cmap="viridis", s=1)  # Scatter plot with color by
plt.colorbar(label="Time (t)")  # Color bar to show time scale
plt.xlabel("x", fontsize=14)
plt.ylabel("z", fontsize=14)
plt.title("Lorenz Attractor: z vs. x", fontsize=16)
plt.grid(True)
plt.show()

# Create 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot the Lorenz attractor in 3D
sc = ax.scatter(x_values, y_values, z_values, c=t_values, cmap="viridis", s=1)
ax.set_xlabel("X", fontsize=14)
ax.set_ylabel("Y", fontsize=14)
ax.set_zlabel("Z", fontsize=14)
ax.set_title("3D Lorenz Attractor", fontsize=16)

# Add color bar to indicate time progression
plt.colorbar(sc, label="Time (t)")

# Show plot
plt.show()
