#!/usr/bin/env python3

"""
HW5.py

 

Author: Danette Farnsworth
Date: 2025-03-29
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, minimize

###################################################################
# Problem 1: Fitting a parabola
#
# Create some fake data of a projectile traveling under the influence of gravity.
# We will however remember that objects in motion under the influence of a
# constant acceleration, in a vacuum, move according to the equation
# x = x_0 + v\cdot t $+$ a\cdot t^2 \over 2$.
# We should be able to recover the acceleration due to gravity by finding the
# three coefficients of a second degree polynomial fit to our fake data
# (use `noisy`, not `y`).
# a) Write a 2nd degree polynomial function then use $\chi^2$ minimization to
# find the three best fit coefficients (25 pts)
# b) Plot your best fit results to confirm the goodness of fit (5 pts)
# c) Solve for acceleration due to gravity! (Pretend it's a real problem)(5 pts)
###################################################################

print("###########################################################")
print("Problem 1: Fitting a parabola")
print("###########################################################")
print("\n")

# Step 1: Generate synthetic data (projectile motion with noise)
np.random.seed(0)
t = np.arange(0, 10, 0.1)
y = 10 + 50 * t - 9.8 * (t**2) / 2
noisy = y + np.random.randn(len(y)) * 10

# Step 2: Manually fit a 2nd-degree polynomial (parabola)
# Set up the design matrix
A = np.vstack([t**2, t, np.ones(len(t))]).T

# Calculate coefficients using least squares (linear algebra)
coefficients = np.linalg.inv(A.T @ A) @ A.T @ noisy
a_fit, b_fit, c_fit = coefficients

# Compute the fitted values
fitted_y = A @ coefficients

# Step 3: Plot the data and fitted curve to check goodness of fit
plt.scatter(t, noisy, label="Noisy data", color="skyblue", marker="x", s=20)
plt.plot(t, fitted_y, color="green", label="Polynomial Fit", linewidth=2)
plt.xlabel("Time (t)")
plt.ylabel("Position (y)")
plt.title("Manual Polynomial Fit to Projectile Data")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Solve for acceleration due to gravity (g)
# Recall: a = (1/2)*acceleration, hence g = -2*a
g_recovered = -2 * a_fit

# Print results
print("Fitted coefficients:")
print(f"a = {a_fit:.3f}, b = {b_fit:.3f}, c = {c_fit:.3f}")
print(f"\nRecovered acceleration due to gravity: g = {g_recovered:.3f} m/s^2")
print("The slight deviation from the actual gravitational acc. (9.8m/s) is due to noise.")
print("\n")

###################################################################
# Problem 2: Below I have defined a simple function, $2 \sin x + 1$.
# Your 2-D/N-D gradient descent algorithm should be able to find a
# reasonable answer with little trouble.
# a) Use your gradient descent algorithm to find the best fit line using chi-square
# minimization. Feel free to use the existing `line` function in your fitting;
# you can assume the form and just find coefficients. Make sure the code you
# wrote for gradient descent is in your turned in assignment.
# b) Plot your best fit results to confirm the goodness of fit.
# c) Plot the chi-square surface that along which you descended; feel free to
# copy-paste the 3-D plotting code from class. (5 pts, +5 bonus** points if
# you plot the path your algorithm took down the slope)
###################################################################

print("###########################################################")
print("Problem 2: Gradient Descent Algorithm")
print("###########################################################")
print("\n")


# Define the provided function
def line(x, m, b):
    return m * np.sin(x) + b


# Generate synthetic data
x = np.arange(0, 10, 0.1)
y_true = line(x, 2, 1)
y_noisy = y_true + np.random.randn(len(x)) * 0.5


# Define Chi-square function
def chi_square(params, x, y):
    m, b = params
    return np.sum((y - line(x, m, b)) ** 2)


# Gradient calculation function
def gradient(params, x, y):
    m, b = params
    # Compute partial derivatives (gradients)
    residual = y - line(x, m, b)  # This line defines residual
    d_chi_dm = -2 * np.sum(residual * np.sin(x))  # derivative wrt m
    d_chi_db = -2 * np.sum(residual)               # derivative wrt b
    return np.array([d_chi_dm, d_chi_db])


# Gradient descent algorithm implementation
def gradient_descent(x, y, init_params, lr=0.0005, epochs=100000, tol=1e-8):
    params = np.array(init_params, dtype=float)
    params_history = [params.copy()]
    chi_history = [chi_square(params, x, y)]

    for _ in range(epochs):
        grad = gradient(params, x, y)         # calculate gradient
        if np.linalg.norm(grad) < tol:  # convergence criterion
            break
        params -= lr * grad
        params_history.append(params.copy())
        chi_history.append(chi_square(params, x, y))


#        params -= lr * grad                   # update paramenters by moving against the grad
 #       params_history.append(params.copy())  # record history for plotting
  #      chi_history.append(chi_square(params, x, y))

    return params, params_history, chi_history


# Run gradient descent
initial_guess = [0, 0]  # Initial guess for m and b
best_params, params_history, chi_history = gradient_descent(x, y_noisy, initial_guess)
m_fit, b_fit = best_params

# Plot data and best-fit line
plt.scatter(x, y_noisy, label="Noisy Data", marker="x", s=20, color="pink")
plt.plot(x, line(x, m_fit, b_fit), label=f"Gradient Descent Fit\n(m={m_fit:.2f}, b={b_fit:.2f})", color="purple")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient Descent Fit")
plt.legend()
plt.grid(True)
plt.show()

# Chi-square surface plot
m_vals = np.linspace(m_fit - 2, m_fit + 2, 50)
b_vals = np.linspace(b_fit - 2, b_fit + 2, 50)
M, B = np.meshgrid(m_vals, b_vals)
Chi_surface = np.array([[chi_square([m, b], x, y_noisy) for m in m_vals] for b in b_vals])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(M, B, Chi_surface, cmap="viridis", alpha=0.8)

# Bonus: Plot gradient descent path
params_history = np.array(params_history)
achi = [chi_square(p, x, y_noisy) for p in params_history]
ax.plot(params_history[:, 0], params_history[:, 1], achi, color="red", marker="o", label="Descent Path")

ax.set_xlabel("m")
ax.set_ylabel("b")
ax.set_zlabel("Chi-square")
ax.set_title("Chi-square Surface and Gradient Descent Path")
plt.legend()
plt.show()

# Print results
print(f"Optimized parameters:\nm = {m_fit:.4f}, b = {b_fit:.4f}")
print("\n")

###################################################################
# Problem 3: Scipy Optimize
###################################################################
# simply use `scipy.optimize.minimize`
#  a) Exercise that right now by using `minimize` in place of gradient
# descent or Nelder-Meade to find the best fit of the `line` function above.
# b) In a previouse course you should have seen `scipy.optimize.curve_fit`.
# Fundamentally, curve_fit is also solving a minimization problem under-the-hood.
# Can it find the best fit parameters of our `line` function?
###################################################################

print("############################################################")
print("Problem 3: Scipy Optimize")
print("############################################################")
print("\n")


# Provided function to fit
def line(x, m, b):
    return m * np.sin(x) + b


# Synthetic data
x = np.arange(0, 10, 0.1)
y_true = line(x, 2, 1)
y_noisy = y_true + np.random.randn(len(x)) * 0.5


# Define chi-square function for minimize
def chi_square(params, x, y):
    m, b = params
    return np.sum((y - line(x, m, b)) ** 2)


# Part (a): Use scipy.optimize.minimize
initial_guess = [0, 0]
result_minimize = minimize(chi_square, initial_guess, args=(x, y_noisy))
m_minimize, b_minimize = result_minimize.x

# Part (b): Using scipy.optimize.curve_fit
try:
    popt, pcov = curve_fit(line, x, y_noisy, p0=initial_guess)
    m_curvefit, b_curvefit = popt
except RuntimeError as e:
    print(f"curve_fit failed: {e}")
    m_curvefit, b_curvefit = None, None

# Plots for part a) and b) to overlay
plt.scatter(x, y_noisy, label="Noisy Data", marker="x", s=20, color="lightblue")

# Minimize result (purple line)
plt.plot(x, line(x, m_minimize, b_minimize), color="purple", linewidth=3.5,
         label=f"minimize Fit\n(m={m_minimize:.2f}, b={b_minimize:.2f})")

# Curve_fit result (pink line), if it converged
if m_curvefit is not None:
    plt.plot(x, line(x, m_curvefit, b_curvefit), color="pink", linewidth=1.5,
             label=f"curve_fit Fit\n(m={m_curvefit:.2f}, b={b_curvefit:.2f})")

plt.title("Comparison: minimize vs curve_fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Summarize results
print(f"Results from minimize:\nm = {m_minimize:.4f}, b = {b_minimize:.4f}")
if m_curvefit is not None:
    print(f"\nResults from curve_fit:\nm = {m_curvefit:.4f}, b = {b_curvefit:.4f}")
else:
    print("\ncurve_fit did not find a solution.")

result = minimize(chi_square, initial_guess, args=(x, y_noisy))
m_min, b_min = result.x
