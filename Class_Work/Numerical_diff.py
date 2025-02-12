#!/usr/bin/env python3

"""
Numerical_diff.py

Author: Danette Farnsworth
Date: 2025-02-11
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#################################################################
# Define the function and derivative
#################################################################
def f(x):
    return 2 * x ** 2


# Calculate using forward difference method: F'(x) = f(x+h)-f(x)/h
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h


# Calculate using backward method: F'(x) = f(x) - f(x-h)/h
def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h


# Calculate using Centeral method: f'(x) = f(x + h/2)-f(x-h/2)/h
def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


# Exact derivative function
def exact_derivative(x):
    return 4 * x


# Points to evaluate
test_points = [2, 100]
# Step sizes
step_sizes = [10**-7, 10**-8, 10**-9]

# Compute and compare
for x in test_points:
    exact = exact_derivative(x)
    print(f"\nEvaluating at x = {x}, exact f'(x) = {exact}")
    for h in step_sizes:
        forward_1 = forward_difference(f, x, h)
        backward_1 = backward_difference(f, x, h)
        central_1 = central_difference(f, x, h)
        print(f"  h = {h:.0e} | Forward Diff: {forward_1:.10f} | Back Diff: {backward_1:.10f} | Central Diff: {central_1:.10f}")


#################################################################
# Discrete Data Points
#################################################################
def g(x):
    return 2 * x**2


def forward_difference_i(y, i, h):
    return (y[i + 1] - y[i]) / h


def backward_difference_i(y, i, h):
    return (y[i] - y[i - 1]) / h


def central_difference_i(y, i, h):
    return (y[i + 1] - y[i - 1]) / (2 * h)


# Exact derivative function
def exact_derivative_i(x):
    return 4 * x


# Fake data
x_1 = np.arange(0, 110, 0.1)
y_1 = np.array([2 * i**2 for i in x_1])

x_2 = np.arange(0, 110, 0.001)
y_2 = np.array([2 * i**2 for i in x_2])

# Evalution points
test_points = [2, 100]
# Step sizes
step_sizes = [0.1, 0.001]

# Compute and Compare
for x, y, h in [(x_1, y_1, 0.1), (x_2, y_2, 0.001)]:
    for point in test_points:
        i = np.where(x == point)[0][0]  # find index
        exact = exact_derivative_i(point)
        forward_2 = forward_difference_i(y, i, h)
        backward_2 = backward_difference_i(y, i, h)
        central_2 = central_difference_i(y, i, h)
        print(f"\nEvaluating at x = {point}, exact f'(x) = {exact}")
        print(f" h = {h: .0e} | Forward Diff: {forward_2: .10f} | Backward Diff: {backward_2: .10f} | Central Diff: {central_2: .10f}")

#################################################################
# Noisy Discrete Data
#################################################################
# Compute numerical derivative using backward difference method
derivative = [(y_1[i] - y_1[i - 1]) / (x_1[i] - x_1[i - 1]) for i in range(1, len(x_1))]

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x_1[1:], derivative, label="Numerical Derivative", color="blue", linestyle="--")
plt.xlabel("x")
plt.ylabel("dy/dx")
plt.title("Derivative of Discrete Dataset")
plt.legend()
plt.grid()
plt.show()


#################################################################
# Noisy Plot x vs f'(x)
#################################################################
def f(x):
    return 2 * x**2


# Generate noisy data
rng = np.random.default_rng()
x_1 = np.arange(0, 110, 0.01)
y_1 = 2 * i**2 + rng.normal(0, 1.5, len(x_1))  # Adding Noise

# Compute numerical derivative using backward difference method
derivative = [(y_1[i] - y_1[i - 1]) / (x_1[i] - x_1[i - 1]) for i in range(1, len(x_1))]


# Function to test different step sizes
def test_step_sizes(step_sizes):
    plt.figure(figsize=(10, 6))
    for h in step_sizes:
        x_sampled = x_1[::h]  # Subsample x
        y_sampled = y_1[::h]  # Subsample y
        derivative_sampled = [(y_sampled[i] - y_sampled[i - 1]) / (x_sampled[i] - x_sampled[i - 1]) for i in range(1, len(x_sampled))]
        plt.plot(x_sampled[1:], derivative_sampled, label=f"Step size {h}")
    plt.xlabel("x")
    plt.ylabel("dy/dx")
    plt.title("Running Numerical Derivative of Noisy Data")
    plt.legend()
    plt.grid()
    plt.show()


# Plot for different step sizes
test_step_sizes([1, 2, 5, 10])
