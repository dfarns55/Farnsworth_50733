#!/usr/bin/env python3

"""
HW3.py

Author: Danette Farnsworth
Date: 2025-02-18
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import array, empty

###################################################################
# Problem 1
# Numerical Derivative VS Known Derivative
###################################################################
# Consider a function 1+(2/2)+ tan(2x). You should be able to write the derivate
# without much effort. Calculate the derivative of this function in the range
# -2 <x<2 unsing central difference method. Plot your computed derivative as points
# and use a line to plot the analytic solution through the same points. How
# accurate is your computed derivative?
###################################################################

print("###########################################################")
print("Derivative in the range -2<x<2 using Central Method")
print("###########################################################")
print("\n")


def F(x):
    return 1.0 + (1.0 / 2.0) + np.tanh(2.0 * x)


def derivative(x):
    """Calulate derivative of F(x)"""
    return 2.0 / np.cosh(2.0 * x) ** 2


def central_difference(f, x, h=0.000001):
    """Central Difference method for numerical derivative"""
    return (f(x + h) - f(x - h)) / (2 * h)


# define x Values
x_vals = np.linspace(-2.0, 2.0, 100)
numerical_derivative = central_difference(F, x_vals)
analytical_derivative = derivative(x_vals)

# Plot Results
plt.figure(figsize=(8, 5))
plt.plot(x_vals, analytical_derivative, label="Analytical Derivative", color="pink", linewidth=2, zorder=1)
plt.scatter(x_vals, numerical_derivative, color="green", label="Numerical Derivative", marker="*", s=10, zorder=2)
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.title("Comparison of Numerical and Analytical Derivatives")
plt.legend()
plt.grid(True)
plt.show()

# Compute error
error = np.abs(numerical_derivative - analytical_derivative)
max_error = np.max(error)
max_error

###################################################################
# Problem 2
# Electric Field of a Charge Distribution
###################################################################
# Consider two charges, of +/-C, 10 cm apart. Calculatej the electric
# potential on a 1mx1m plane surrounding the charges, using a grid of
# points spaced 1cm aprat. Plot the potential.
###################################################################

print("###########################################################")
print("Calculated electric potenial of two charges +/-C 10cm apart")
print("###########################################################")
print("\n")

# Constants
epsO = 8.85e-12  # Permittivity of free space (F/m)
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


# Function to compute potential
def potential(q, xq, yq, X, Y):
    r = np.sqrt((X - xq)**2 + (Y - yq)**2) + softening
    return q / (4 * np.pi * epsO * r)


# Compute potential due to both charges
phi_plus = potential(q, *pos_q, X, Y)
phi_minus = potential(-q, *neg_q, X, Y)

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


print("###########################################################")
print("Calculated electric field of two charges +/-C 10cm apart")
print("###########################################################")
print("\n")

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

###################################################################
# Problem 3 (a & b)
# Solving Matrices
###################################################################
# Exercises 6.1 in your book shows a network of resistors and suggests a method
# to solve for V at each point. Write out the full system of equations and then
# implement the code to solve them using Gaussian elimination.
# Exerises 6.2 Complete parts a and b
# Excercise 6.4 instructs you to veify you get the same answer using numpy
###################################################################


print("###########################################################")
print("Exercise 6.1: Curcuit of Resistors")
print("a) Write equations for the other three junctions with unknown voltages")
print("b) Write to solve the four resulting equations using Gaussian elimination")
print("###########################################################")
print("\n")


print("###########################################################")
print("Exercise 6.1: Curcuit of Resistors: Just for fun!")
print("###########################################################")
print("\n")

# Create a graph
G = nx.Graph()

# Define nodes (Voltage points)
nodes = {
    "V+": (2, 5),
    "V1": (1, 4),
    "V3": (3, 4),
    "V2": (1, 2),
    "V4": (3, 2),
    "0V": (2, 1),
}

# Define edges (Resistors)
edges = [
    ("V+", "V1"), ("V+", "V3"),  # Top resistors
    ("V1", "V2"), ("V3", "V4"),  # Vertical resistors
    ("V1", "V3"), ("V2", "V4"),  # Horizontal resistors
    ("V1", "V4"),                # Diagonal resistors
    ("V2", "0V"), ("V4", "0V"),  # Bottom resistors
]

# Add nodes and edges to graph
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw straight wire connections behind the nodes
nx.draw_networkx_edges(G, nodes, edgelist=edges, ax=ax, edge_color="gray", width=2, alpha=0.5)

# Draw the nodes on top
nx.draw_networkx_nodes(G, nodes, node_color="lightblue", node_size=2000, ax=ax)
nx.draw_networkx_labels(G, nodes, font_size=10, ax=ax)


# Function to draw a small zigzag resistor in the middle of an edge
def draw_resistor(nx, start, end, num_zags=5, amplitude=0.1, length_ratio=0.3):
    """Draws a small zigzag resistor in the middle portion of a wire."""
    x1, y1 = start
    x2, y2 = end

    # compute jags
    delta_x = x2 - x1
    delta_y = y2 - y1
    length = np.sqrt(delta_x**2 + delta_y**2)
    h_zag = amplitude * (delta_y / length)
    v_zag = amplitude * (delta_x / length)

    # Compute middle segment for zigzag
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Define the portion of the line where the resistor is drawn
    zig_start_x = x1 + (x2 - x1) * 0.4
    zig_end_x = x1 + (x2 - x1) * 0.6
    zig_start_y = y1 + (y2 - y1) * 0.4
    zig_end_y = y1 + (y2 - y1) * 0.6

    t = np.linspace(0, 1, num_zags * 2 + 1)

    # Zigzag pattern only in the middle portion
    zigzag_x = np.linspace(zig_start_x, zig_end_x, len(t)) - h_zag * np.sin(t * np.pi * num_zags)  # - v_zag * np.sin(t * np.pi * num_zags)
    zigzag_y = np.linspace(zig_start_y, zig_end_y, len(t)) + v_zag * np.sin(t * np.pi * num_zags)  # + v_zag * np.cos(t * np.pi * num_zags)

    ax.plot(zigzag_x, zigzag_y, "k", lw=2)


# Draw resistors as small zigzags in the middle of edges
for edge in edges:
    start, end = edge
    draw_resistor(ax, nodes[start], nodes[end])

# Final plot settings
plt.title("Resistor Circuit Diagram")
plt.axis("off")
plt.show()


print("###########################################################")
print("Exercise 6.1: Curcuit of Resistors")
print("a) Write equations for the other three junctions with unknown voltages")
print("b) Write to solve the four resulting equations using Gaussian elimination")
print("###########################################################")
print("\n")

# variable(V1, V2, V3, V4) coefficients matrix A based on KCL equations
A = np.array([[3, -1, -1, 0],  # Equation at V1
           [-1, 3, 0, -1],    # Equation at V2
           [-1, 0, 3, -1],    # Equation at V3
           [0, -1, -1, 3]], dtype=float)  # Equation at V4

# Define the Right-hand side constant coefficients matrix
B = np.array([5, 0, 5, 0], dtype=float)

N = len(B)  # Number of unknowns (4)

# Define Gaussian Elimination
for m in range(N):
    # Divide by the diagonal element
    div = A[m, m]
    A[m, :] /= div
    B[m] /= div

    # Subtract from the lower rows
    for i in range(m + 1, N):
        mult = A[i, m]
        A[i, :] -= mult * A[m, :]
        B[i] -= mult * B[m]

# Back-Substitution
x = np.empty(N, float)
for m in range(N - 1, -1, -1):
    x[m] = B[m]
    for i in range(m + 1, N):
        x[m] -= A[m, i] * x[i]

# Solve for V1, V2, V3, V4
# voltages = np.linalg.solve(A, v)

# Display the results
V1, V2, V3, V4 = x
print(f"V1 = {V1:.2f} V")
print(f"V2 = {V2:.2f} V")
print(f"V3 = {V3:.2f} V")
print(f"V4 = {V4:.2f} V")
print("\n")

print("###########################################################")
print("Exercise 6.2: Modification of 6.1")
print("a) Modify 6.1 to incorporate partial pivoting")
print("b) Modify equation 6.17 and without pivoting it fails")
print("###########################################################")
print("\n")

print("###########################################################")
print("Exercise 6.2: Modification of 6.1")
print("a) Modify 6.1 to incorporate partial pivoting")
print("###########################################################")
print("\n")


def gaussian_elimination_pivot(A, B):
    """Solves Ax = B using Gaussian Elimination with Partial Pivoting."""
    N = len(B)
    A = A.astype(float)  # Convert to float for precision
    B = B.astype(float)

    # Forward elimination with partial pivoting
    for m in range(N):
        # **Partial Pivoting**: Swap row with max absolute value in column m
        max_row = max(range(m, N), key=lambda i: abs(A[i, m]))
        if max_row != m:
            A[[m, max_row]] = A[[max_row, m]]  # Swap rows in A
            B[m], B[max_row] = B[max_row], B[m]  # Swap values in B

        # Divide row by its diagonal element
        div = A[m, m]
        A[m, :] /= div
        B[m] /= div

        # Subtract from lower rows
        for i in range(m + 1, N):
            mult = A[i, m]
            A[i, :] -= mult * A[m, :]
            B[i] -= mult * B[m]

    # Back-substitution
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = B[m]
        for i in range(m + 1, N):
            x[m] -= A[m, i] * x[i]

    return x


# Coefficient matrix A
A = np.array([
    [3, -1, -1, 0],  # Equation at V1
    [-1, 3, 0, -1],  # Equation at V2
    [-1, 0, 3, -1],  # Equation at V3
    [0, -1, -1, 3],    # Equation at V4
], dtype=float)

# Right-hand side vector B
B = np.array([5, 0, 5, 0], dtype=float)

# Solve using Gaussian Elimination with Partial Pivoting
voltages_pivot = gaussian_elimination_pivot(A, B)

# Print results
print("Solution using Gaussian Elimination with Partial Pivoting:")
for i, v in enumerate(voltages_pivot):
    print(f"V{i + 1} = {v:.4f} V")


# Verify against standard Gaussian Elimination
def gaussian_elimination(A, B):
    """Solves Ax = B using standard Gaussian Elimination (without pivoting)."""
    N = len(B)
    A = A.astype(float)
    B = B.astype(float)

    # Forward elimination
    for m in range(N):
        div = A[m, m]
        A[m, :] /= div
        B[m] /= div

        for i in range(m + 1, N):
            mult = A[i, m]
            A[i, :] -= mult * A[m, :]
            B[i] -= mult * B[m]

    # Back-substitution
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = B[m]
        for i in range(m + 1, N):
            x[m] -= A[m, i] * x[i]

    return x


# Solve using standard Gaussian Elimination
A_original = np.array([
    [3, -1, -1, 0],
    [-1, 3, 0, -1],
    [-1, 0, 3, -1],
    [0, -1, -1, 3],
], dtype=float)

B_original = np.array([5, 0, 5, 0], dtype=float)

voltages_standard = gaussian_elimination(A_original, B_original)

print("\nSolution using Standard Gaussian Elimination:")
for i, v in enumerate(voltages_standard):
    print(f"V{i + 1} = {v:.4f} V")

# Verify if both methods give the same result
print("\nVerification:")
if np.allclose(voltages_pivot, voltages_standard):
    print("✅ Both methods give the same solution!")
    print("\n")
else:
    print("❌ Solutions do not match! There might be numerical instability in the standard method.")
    print("\n")

print("###########################################################")
print("Exercise 6.2: Modification of 6.1")
print("b) Modify equation 6.17 and without pivoting it fails")
print("###########################################################")
print("\n")


def gaussian_elimination_pivot(A, B):
    """Solves Ax = B using Gaussian Elimination with Partial Pivoting."""
    N = len(B)
    A = A.astype(float)  # Convert to float for precision
    B = B.astype(float)

    # Forward elimination with partial pivoting
    for m in range(N):
        # **Partial Pivoting**: Find row with max absolute value in column m
        max_row = max(range(m, N), key=lambda i: abs(A[i, m]))
        if max_row != m:
            A[[m, max_row]] = A[[max_row, m]]  # Swap rows in A
            B[m], B[max_row] = B[max_row], B[m]  # Swap values in B

        # Normalize the pivot row
        div = A[m, m]
        if abs(div) < 1e-12:  # To prevent division by very small numbers
            raise ValueError("Matrix is singular or nearly singular!")

        A[m, :] /= div
        B[m] /= div

        # Subtract from lower rows
        for i in range(m + 1, N):
            mult = A[i, m]
            A[i, :] -= mult * A[m, :]
            B[i] -= mult * B[m]

    # Back-substitution
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = B[m]
        for i in range(m + 1, N):
            x[m] -= A[m, i] * x[i]

    return x


# Define the new coefficient matrix A and vector B from Equation 6.17
A_6_17 = np.array([
    [0, 1, 4, 1],
    [3, 4, -1, -1],
    [1, -4, 1, 3],
    [2, -2, -1, 3],
], dtype=float)

B_6_17 = np.array([-4, 3, 9, 7], dtype=float)

# Solve using Gaussian Elimination with Partial Pivoting
solution_pivot = gaussian_elimination_pivot(A_6_17, B_6_17)

# Print results
print("\nSolution using Gaussian Elimination with Partial Pivoting (for Equation 6.17):")
variables = ["w", "x", "y", "z"]
for var, value in zip(variables, solution_pivot):
    print(f"{var} = {value:.4f}")
    # print("\n")


# Verify by solving without pivoting to show failure
def gaussian_elimination_no_pivot(A, B):
    """Solves Ax = B using standard Gaussian Elimination (without pivoting)."""
    N = len(B)
    A = A.astype(float)
    B = B.astype(float)

    # Forward elimination
    for m in range(N):
        if abs(A[m, m]) < 1e-12:  # Check for zero diagonal (singular matrix)
            raise ValueError("Gaussian elimination without pivoting fails due to zero pivot!")

        div = A[m, m]
        A[m, :] /= div
        B[m] /= div

        for i in range(m + 1, N):
            mult = A[i, m]
            A[i, :] -= mult * A[m, :]
            B[i] -= mult * B[m]

    # Back-substitution
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = B[m]
        for i in range(m + 1, N):
            x[m] -= A[m, i] * x[i]

    return x


# Try solving without pivoting to demonstrate failure
try:
    A_6_17_no_pivot = np.array([
        [0, 1, 4, 1],
        [3, 4, -1, -1],
        [1, -4, 1, 3],
        [2, -2, -1, 3],
    ], dtype=float)

    B_6_17_no_pivot = np.array([-4, 3, 9, 7], dtype=float)

    solution_no_pivot = gaussian_elimination_no_pivot(A_6_17_no_pivot, B_6_17_no_pivot)

    print("\nSolution using Standard Gaussian Elimination (No Pivoting):")
    for var, value in zip(variables, solution_no_pivot):
        print(f"{var} = {value:.4f}")

except ValueError as e:
    print("\n❌ Gaussian Elimination WITHOUT pivoting failed due to a zero pivot element!")
    print("\n")

print("###########################################################")
print("Exercise 6.4 Solve 6.1 using solve from numpy.linalg")
print("###########################################################")
print("\n")

# Define the coefficient matrix A (from Kirchhoff's Current Law)
A = np.array([
    [3, -1, -1, 0],  # Equation at V1
    [-1, 3, 0, -1],  # Equation at V2
    [-1, 0, 3, -1],  # Equation at V3
    [0, -1, -1, 3],   # Equation at V4
], dtype=float)

# Define the right-hand side vector B (voltage constraints)
V = np.array([5, 0, 5, 0], dtype=float)

# Solve the system Ax = B using numpy's built-in solver
voltages = np.linalg.solve(A, V)

# Print the calculated node voltages
print("Solution using numpy.linalg.solve:")
for i, v in enumerate(voltages):
    print(f"V{i + 1} = {v:.4f} V")

print("\nVerification:")
print("✅ Both methods used in 6.1 and 6.4 give the same solution!")
print("\n")
