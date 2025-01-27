#!/usr/bin/env python3

"""
HW1.py

Author: Danette Farnsworth
Date: 2025-01-20
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Homework Problem 1
# In nuclear physics, the semi-empirical mass formula is a formula for calculating
# the approximate nuclear binding energy of an atomic nucleus Zwith atomic number
# Z and mass number A. The formula looks like this:
# B= a_1*A - a_2*A^(2/3) - a_3*(z^2/A^(1/3) - a_4*((A-2*Z)^2/A) - a_5/A^(1/2)
# where, in units of millions of electron volts (MeV), the constants are
# a_1=15.67, a_2=17.23 , a_3=0.75 , a_4=93.2 and
# a_5 = (0 A is odd, 12.0 A and Z are both even, -12.0 Ais even and Z is odd
# Write a function that takes as its input the values of A and Z, and prints
# out the binding energy for the corresponding atom.
# Use your program to find the binding energy of an atom with A = 58
# and Z=28.


def calculate_binding_energy(Z, A):
    # Constants in Mev
    a1 = 15.67
    a2 = 17.23
    a3 = 0.75
    a4 = 93.2

    # Determine a5 based on A and Z
    if A % 2 != 0:
        a5 = 0  # A is odd
    elif Z % 2 == 0:
        a5 = 12.0  # A and Z are both odd
    else:
        a5 = -12.0  # A is even and Z is odd

    # Calculating binding energy using the semi-empirial mass formula
    term1 = a1 * A
    term2 = - a2 * A**(2 / 3)
    term3 = - a3 * (Z**2 / A**(1 / 3))
    term4 = - a4 * ((A - 2 * Z)**2 / A)
    term5 = - a5 / A**(1 / 2)

    binding_energy = term1 + term2 + term3 + term4 + term5
    return binding_energy


# Example Usage: Calculate binding energy for Z = 28 and A = 58
Z = 28
A = 58
binding_energy = calculate_binding_energy(Z, A)
print(f"The Binding Energy for Z= {Z}, A={A} is approximately {binding_energy:.2f} MeV")


# Homework Problem 2
# Now create a new function that calculates the binding energy per nucleon, B/A.
# You shoudl be able to write a very short function that call your previous
# function to do the heavy lifting. Test you function with some famous cases:
# Iron 56 and Carbon 12.


def calculate_binding_energy_per_nucleon(Z, A):
    """For a given atomic number and mass, compute the binding energy per nucleon"""
    binding_energy = calculate_binding_energy(Z, A)
    return binding_energy / A


# Samples: Iron 56 and Carbon 12
samples = ((26, 56), (6, 12))
for Z, A in samples:
    binding_energy_per_nucleon = calculate_binding_energy_per_nucleon(Z, A)
    print(f"The Binding Energy per nucleon for Z = {Z},", end="")
    print(f" A = {A} is approximately {binding_energy_per_nucleon: .2f} MeV.")

# Homework Problem 3
# Create yet another function that takes a single argument, Z, and finds the
# value of A at which the binding energy per nucleon is largest. You are aided
# in this problem in the fact that A is discrete; you can simply check all
# reasonable values without worrying about mimizing a function or any such
# thing. You should be able to easily determine the minimum value for A. If
# you consult a table of nuclides, you'll see some elements have nuclides,
# with measured half lives, with A > 2Z, so in this exercise use A = 3Z as an
# upper limit.


def compute_best_A(Z, verbose=False):
    """Compute the A for every Z, then the A that has the highest B"""
    binding_energy_per_nucleon = pd.Series(np.nan, index=list(range(Z, 3 * Z + 1)),
                                           name="Binding Energy Per Nucleon")
    for A in binding_energy_per_nucleon.index:
        binding_energy_per_nucleon[A] = calculate_binding_energy_per_nucleon(Z, A)
    best_a = binding_energy_per_nucleon.idxmax()
    if verbose:
        print(f"For the element with atomic number {Z}, ", end="")
        print(f"max binding energy is at mass {best_a} ", end="")
        print(f"with binding energy {binding_energy_per_nucleon[best_a]:.3f} MeV")
    return best_a, binding_energy_per_nucleon


# find all binding energies (and the best one) for iron
best_a, all_binding_energies = compute_best_A(28, verbose=True)


# Homework Problem 4
# Finally, create a plot of the highest binding energy per nucleon as a function
# of Z. Remember: science quality plots have labeled axes and a title.


def calc_max_binding(Z_min, Z_max):
    """Plot of the highest binding energy per nucleon as a function of Z"""
    output = pd.Series(index=range(Z_min, Z_max + 1))
    for z in output.index:
        best_A, binding_energies = compute_best_A(z)  # Find the best A and max B/A for each Z
        output[z] = binding_energies[best_A]
    return output


def plot_max_binding_energy_per_nucleon(in_data):
    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(in_data, color="pink", label="Max Binding Energy per Nucleon", zorder=1)
    plt.scatter(in_data.index, in_data, marker="*", color="lightblue", zorder=2)
    plt.title("Maximum Binding Energy per Nucleon as a Function of Z")
    plt.xlabel("Atomic Number (Z)")
    plt.ylabel("Binding Energy per Nucleon (MeV)")
    plt.grid(True)
    plt.legend()
    # Add shifted text
    plt.text(26 + 2, in_data[26], "Iron")
    plt.scatter(26, in_data[26], marker="$😊$", s=400, color="red", zorder=3)
    plt.text(6 + 2, in_data[6], "Carbon")
    plt.scatter(6, in_data[6], marker="$😊$", s=400, color="red", zorder=3)
    plt.show()


bindings = calc_max_binding(1, 100)
plot_max_binding_energy_per_nucleon(bindings)

# Homework Problem 5
# It's a common situation in physics that an experiment produces data that lies roughly on a straight line.
# The straight line can be represented in the familiar form y=mx+b and a frequent question is what the
# appropriate values of the slope and intercepti c are that correspond to the measured data. Since the
# data don't fall perfectly on a straight line, there is no perfect answer to such a question, but we can
# find the straight line that gives the best compromise fit to the data. The standard technique for doing
# this is the method of least squares.
# In this repo is a file called millikan.txt. The file contains two columns of numbers, giving the
# x and y coordinates of a set of data points. Write a program to read these data points and make a graph

milikan = pd.read_csv("millikan.txt", names=["Hz", "V"], sep=" ")
Ex = milikan.Hz.mean()
Ey = milikan.V.mean()
Exx = (milikan.Hz**2).mean()
Exy = (milikan.Hz * milikan.V).mean()

m = (Exy - Ex * Ey) / (Exx - Ex**2)
c = (Exx * Ey - Ex * Exy) / (Exx - Ex**2)
fitted_values = m * milikan.Hz + c
print(f"h={m * 1.602e-19}")

# Assuming data is representive of the Milikan Drop Experiment
fig, ax = plt.subplots()
ax.scatter(milikan.Hz, milikan.V, marker="o", color="purple", zorder=2)
ax.plot(milikan.Hz, fitted_values, color="lightblue", zorder=1)
ax.set_title("Milikan Drop Experiment")
ax.set_xlabel("Frequency (nu, in Hz)")
ax.set_ylabel("Volts (V)")
ax.text(0.2, 0.8, f"slope = {m:.2e}\nintercept = {c:.2f}", zorder=3, transform=ax.transAxes)
plt.grid(True)
plt.show()
