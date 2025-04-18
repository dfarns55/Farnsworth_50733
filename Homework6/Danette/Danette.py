#!/usr/bin/env python3

"""
Danette.py

 Example of a pip installable package

Author: Danette Farnsworth
Date: 2025-04-08
"""

import math as m

import networkx as nx
import numpy as np
from numpy import array, empty

# first function: Homework 2: Simple Derivative


def numerical_derivative(F, x, delta):
    """Calculate a Simple Derivative"""
    print(f"{delta}: derivative is {(F(x + delta) - F(x)) / delta}", end="")
    print(f"errors = {(F(x + delta) - F(x)) / delta - 1}")
    return np.abs((F(x + delta) - F(x)) / delta - 1)


# second function: Homework 3: Electric Field of a Charge Distribution

# Function to compute potential
def potential(q, xq, yq, X, Y):
    """Electric Potential Function"""
    softening = 1e-1  # Softening parameter to avoid singularities
    epsO = 80.0 * 8.85e-12  # Permittivity of water (F/m)
    r = np.sqrt((X - xq)**2 + (Y - yq)**2) + softening
    return q / (4 * np.pi * epsO * r)


# third function: Homework 4: Lorenz Functions / Runge-Kutta 4th Order

# Lorenz system of equations
def lorenz(r, sigma, b, r_values):
    """Lorenz system of Equations"""
    x, y, z = r_values
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])


# Runge-Kutta 4th order
def runge_kutta_4(f, r0, t0, t_end, h, *params):
    """ Runge-Kutta 4th order method for solving a system"""
    t_values = np.arange(t0, t_end + h, h)
    r_values = np.zeros((len(t_values), len(r0)))
    r_values[0] = r0
    for i in range(len(t_values) - 1):
        t = t_values[i]
        r = r_values[i]
        k1 = h * f(*params, r)
        k2 = h * f(*params, r + 0.5 * k1)
        k3 = h * f(*params, r + 0.5 * k2)
        k4 = h * f(*params, r + k3)
        r_values[i + 1] = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, r_values
