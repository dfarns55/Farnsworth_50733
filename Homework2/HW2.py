#!/usr/bin/env python3

"""
HW2.py

Author: Danette Farnsworth
Date: 2025-02-03
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


################################################################
# Problem 1
# Compute Factorials using Integers
################################################################
# Write a program to calculate and print the factorial of a number.
# If you wish you can base your program on the user-defined function
# for factorial given in Section 2.6, but write your program so that
# it calculates the factorial using integer variables, not floating-point
# ones. Use your program to calculate the factorial of 200.
# Now modify your program to use floating-point variables instead
# and again calculate the factorial of 200. What do you find? Explain.
################################################################
# Method 1:
# From Section 2.6 using for loop
################################################################
def factorial(n):
    """Compute the factorial of n, forcing it to be an integer"""
    f = 1
    for k in range(1, int(n) + 1):
        f *= k
    return f


def factorial_float(n):
    """Compute the factorial of n, as a float"""
    f = 1.0
    for k in range(1, int(n) + 1):
        f *= float(k)
    return f


# calculate and print the factorial of 200
a = 200
b = 200.0

print("####################################################################")
print("Problem 1: Loop method")
print("####################################################################")
print(f"{a} factorial is: {factorial(a)}")
print("\n")
print(f"{b} factorial is: {factorial_float(b)}")
print("\n")


###############################################################
# Method 2:
# Using Recursive method
###############################################################
def factorial(input_number):
    """Compute the factorial of n, forcing it to be an integer"""
    if int(input_number) == 1:
        return 1
    return input_number * factorial(int(input_number) - 1)


def factorial_float(input_number):
    """Compute the factorial of n, as a float"""
    if int(input_number) == 1:
        return 1.0
    return float(input_number) * factorial_float(float(input_number) - 1.0)


print("####################################################################")
print("Problem 1: Recursive method")
print("####################################################################")
print(f"200 factorial is {factorial(200)}")
print("\n")
print(f"200.0 factorial is {factorial_float(200.0)}")
print("\n")
print("####################################################################")
print("The factorial for 200.0 as a float is to large so it will outout inf")
print("####################################################################")
print("\n")


################################################################
# Problem 2
# A Simple Derivative
################################################################
# Write a program that defines a function f(x) returning the values x(x-1)
# then calculate the derivative of the point x=1 using the formula:
# df/dx = lim(delta-->0) [(f(x+delta)-f(x))/delta] with  delta = 10^-2.
# Calculate the true value of the same derivatives analytically and compare
# with the answer your program gives. The two will not agree perfectly, why?

# Repeat the calculation for delta = 10^-4, 10^-6, 10^-8, 10^-10, 10^-14.
# You should see that the accuracy of the calculation initially gets better
# as delta gets smaller, but then gets worse again, why?
################################################################
def f(x):
    return x * (x - 1)


def numerical_derivative(F, x, delta):
    print(f"{delta}: derivative is {(F(x + delta) - F(x)) / delta}", end="")
    print(f"errors = {(F(x + delta) - F(x)) / delta - 1}")
    return np.abs((F(x + delta) - F(x)) / delta - 1)


print("####################################################################")
print("Problem 2:Simple Derivative")
print("####################################################################")
print("Analytical derivative is f'(x) = 2x-1, so f'(1)=1.0")
errors = pd.Series(index=(10**-2, 10**-4, 10**-6, 10**-8, 10**-10, 10**-12, 10**-14))
for exponent in errors.index:
    errors[exponent] = numerical_derivative(f, 1, exponent)

print("####################################################################")
print("Accuracy falls off because floating point cannot accurately represent values so close to 0.0")
print("so both the subtraction and division become numerically ill-conditioned")
print("####################################################################")
print("\n")

# Plot Error in Numerical Dericative Approx (just for fun)
plt.figure(figsize=(8, 6))
plt.loglog(errors, marker="o", linestyle="-", color="pink")
plt.xlabel("Delta")
plt.ylabel("Absolute Error")
plt.title("Error in Numerical Derivative Approximation")
plt.grid(True, which="both", linestyle="--")
plt.show()


################################################################
# Problem 3
# Simpson's Rule
################################################################
# Write a program to calculate an approximate value for the integral:
# [0,2] (x^4-2x+1)dx from Example 5.1 in the book, but using Simpson's rule with 10 slices
# instead of the trapezoid rule. Compare your result to the known correct value of 4.4
# What is the fractional error on you calculation

# Copy your code and modify it to use one hundred slices, then one thousand.
################################################################
def g(x):
    return x**4 - 2 * x + 1


##########################################
def simpsons_rule(a, b, n, w):
    if n % 2 == 1:
        raise ValueError("Number of slices must be even for Simpson's rule.")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    f = w(x)

    integral = f[0] + f[-1] + 4 * sum(f[1:n:2]) + 2 * sum(f[2:n - 1:2])
    integral *= h / 3

    return integral


def fractional_error(approx, true_value):
    return abs((approx - true_value) / true_value)


# Given integral bounds and correct value
a, b = 0, 2
true_value = 4.4

print("####################################################################")
print("Problem 3:Simpson's Rule")
print("####################################################################")

# Calculate with different slices
for n in [10, 100, 1000]:
    approx_value = simpsons_rule(a, b, n, g)
    error = fractional_error(approx_value, true_value)

    print(f"n = {n}: Approximation = {approx_value:.6f}, Fractional Error = {error:.6e}")
print("\n")


################################################################
# Problem 4
# An Integral with no Analytic Solution
################################################################
# Consider the integral: E(x) = [0, x] exp^ -t^2 dt
# Write a program to calculte E(x) for values of x from 0 to 3 in steps of 0.1.
# When you are convinced your program is working, extend it further to make
# a graph of E(x) as a function of x.
################################################################
# Function to compute the integral E(x)
def E(x):
    result = simpsons_rule(0, x, int(x * 100), lambda t: np.exp(-t**2))
    return result


# Generate x values from 0 to 3 in steps of 0.1
x_values = np.arange(0 + .1, 4, 0.1)

# Compute corresponding E(x) values
y_values = [E(x) for x in x_values]

# Plot E(x) as a function of x
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r"$E(x) = \int_0^x e^{-t^2} dt$", color="purple")
plt.xlabel("x")
plt.ylabel("E(x)")
plt.title("Integral of $e^{-t^2}$ from 0 to x")
plt.legend()
plt.grid()
plt.show()
