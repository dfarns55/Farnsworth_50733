#!/usr/bin/env python3

"""
HW4.py

 

Author: Danette Farnsworth
Date: 2025-03-05
"""

import math as m

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

###################################################################
# Problem 1: Excercise 8.1
# A low-pass filter
###################################################################
# a) Write a program (or modify a previous one) to solve this equation
# for V_out(t) using the fourth-order Runge-Kutta method when in the
# input signal is a square-wave with frequency ~1 and amp ~1:
#  V_in(t) = {1 if [2t] is even
#           {-1 if [2t] is odd
# where [x] means x rounded down to the next lowest integer. Use the
# program to make plots of the output of the filter circuit from
# t=0 to t=10 when RC=0.01, 0.1, and 1 (so that's three separate plots)
# with initial condition V_out(0)=0. You will have to make a decision
# about what value of to use in your calculation. Small values give
# more accurate results, but the program will take longer to run.
# Try a variety of different values and choose one for your final
# calculations that seems sensible to you.
#
# b) Based on the graphs produced by your program, describe what you
# see and explain what the circuit is doing.
###################################################################

print("###########################################################")
print("Problem 8.1.a Plot")
print("Use RK4 to solve V_out(t)")
print("###########################################################")
print("\n")

print("###########################################################")
print("Problem 8.1.b Explaination")
print("###########################################################")
print("The results of these 3 plots of V_out(t) for different RC values")
print("(0.01, 0.1,and 1, illustate the low-pass filter's effect on a square")
print("wave input. For RC=0.01 the circuit reacts quickly, closely following")
print("the square wave with slight smoothing. At RC=0.1, the response slows,")
print("and the output becomes more rounded, reducing sharp transitions. With")
print("RC=1, the filter response is much slower, significantly smoothing the")
print("wave and removing most high-frequency components. In conclusion, smaller")
print("RC values allow faster responces and retain more high-frequency details,")
print("while larger RC values result in greater smoothing and reduced")
print("high-frequency contect.")
print("\n")


def Vin(t):
    V = -1.0
    if (m.ceil(2.0 * t) % 2 == 0): V = -V
    return V


# Parameters
RC = [0.01, 0.1, 1.0]
Vout0 = 0.0
h = 0.01
t = np.arange(0.0, 10.0 + h, h)  # Time array
Vout = np.zeros((len(t), len(RC)))  # Initialize output array
Vout[0, :] = Vout0  # Initial condition

# Runge-Kutta Method to compute Vout
for j in range(len(RC)):  # Loop over RC Values
    for i in range(len(t) - 1):  # Loop over time steps
        k0 = h * (Vin(t[i]) - Vout[i, j]) / RC[j]
        k1 = h * (Vin(t[i] + 0.5 * h) - (Vout[i, j] + 0.5 * k0)) / RC[j]
        k2 = h * (Vin(t[i] + 0.5 * h) - (Vout[i, j] + 0.5 * k1)) / RC[j]
        k3 = h * (Vin(t[i] + h) - (Vout[i, j] + k2)) / RC[j]
        Vout[i + 1, j] = Vout[i, j] + (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
# Plot the results for different RC values
plt.figure(figsize=(12, 8))

for idx, rc_value in enumerate(RC):
    plt.plot(t, Vout[:, idx], label=f"RC = {rc_value}")

plt.xlabel("Time (t)", fontsize=14)
plt.ylabel("Vout(t)", fontsize=14)
plt.title("Low-pass Filter Output for Different RC Values", fontsize=16)
plt.grid(True)
plt.legend(title="RC Values", fontsize=12)
plt.show()

###################################################################
# Problem 2: Excercise 8.2
# The Lotka -- Volterra Equations
###################################################################
# The Lotka--Volterra equations are a mathematical model of
# predator--prey interactions between biological
# species.  Let two variables x and y be proportional to the size of the
# populations of two species, traditionally called "rabbits" (the
# prey) and "foxes" (the predators).  You could think of x and y as
# being the population in thousands, say, so that x=2 means there are 2000
# rabbits.  Strictly the only allowed values of x and y would then be
# multiples of 0.001, since you can only have whole numbers of rabbits or
# foxes.  But 0.001 is a pretty close spacing of values, so it's a decent
# approximation to treat $x$ and y as continuous real numbers so long as
# neither gets very close to zero.
# In the Lotka-Volterra model the rabbits reproduce at a rate proportional
# to their population, but are eaten by the foxes at a rate proportional to
# both their own population and the population of foxes:
# dx/dt = alpha*x - beta*x*y
# where alpha and beta are constants.  At the same time the foxes
# reproduce at a rate proportional the rate at which they eat
# rabbits-because they need food to grow and reproduce-but also die of
# old age at a rate proportional to their own population:
# dy/dt = gamma*x*y - delta*y
# where gamma and delta are constants
# Write a program to solve these equations using the fourth-order
# Runge--Kutta method for the case $\alpha=1$, $\beta=\gamma=0.5$, and
# $\delta=2$, starting from the initial condition $x=y=2$.  Have the
# program make a graph showing both $x$ and $y$ as a function of time on
# the same axes from $t=0$ to $t=30$.  (Hint: Notice that the differential
# equations in this case do not depend explicitly on time $t$; in vector
# notation, the right-hand side of each equation is a function $f(\vec{r})$
# with no $t$ dependence.  You may nonetheless find it convenient to define
# a Python function $\verb|f(r,t)|$ including the time variable, so that your
# program takes the same form as programs given earlier in this chapter.
# You don't have to do it that way, but it can avoid some confusion.) (15 points)
# Describe in words what is going on in the system, in terms of rabbits
# and foxes. (5 points)

print("###########################################################")
print("Problem 8.2.a Plot")
print("Lotka-Volterra-The Rabbit and the Fox")
print("###########################################################")
print("\n")

print("###########################################################")
print("Problem 8.2.b Data interpetation")
print("###########################################################")
print("In this senerio the Lotka-Volterra describes a classic predator-prey interaction:")
print("\033[4mRabbits (prey):\033[0m reproduce at a rate proportional to their population.")
print("The population grows when there are fewer predators (foxes).")
print("\033[4mFoxes (predators):\033[0m reproduce at a rate proportional to the")
print("number of rabbits they eat. However, they also dies at a rate proportional")
print("to thier own population.")
print("\n")
print("\033[4mThe Simulation\033[0m")
print("\033[4mRabbits (prey):\033[0m increase in population as long as there")
print("are not too many predators. When their population grows too large, they")
print("provide more food for the foxes.")
print("\033[4mFoxes (predators):\033[0m increase as they feed on rabbits, but")
print("their population also decreases when there are fewer rabbits available")
print("to sustain them.")
print("\n")
print("\033[4mThe cycle repeats:\033[0m As the rabbit population decreases due to fox predation,")
print("the fox population begins to decline as well, allowing the rabbit population")
print("to recover. This cyclical nature of the populations is characteristic of")
print("the predator-prey model.Thus, the system shows periodic oscillations")
print("where the populations of rabbits and foxes rise and fall in tandem.")
print("\n")


# Lotka-Volterra system of equations
"""Lotka-Volterra system of equation"""


def f(r, t):
    x, y = r
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 2.0
    dxdt = alpha * x - beta * x * y
    dydt = gamma * x * y - delta * y
    return np.array([dxdt, dydt])


# Runge-Kutta 4th order method
def runge_kutta_4(f, r0, t0, t_end, h):
    t_values = np.arange(t0, t_end + h, h)
    r_values = np.zeros((len(t_values), len(r0)))
    r_values[0] = r0
    for i in range(len(t_values) - 1):
        t = t_values[i]
        r = r_values[i]
        k1 = h * f(r, t)
plt.figure(figsize=(10, 6))
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(r + k3, t + h)
        r_values[i + 1] = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, r_values


# Initial conditions and parameters
r0 = np.array([2.0, 2.0])  # Initial populations of rabbits and foxes
t0 = 0.0  # Start time
t_end = 30.0  # End time
h = 0.01  # Time step size

# Solve the system using Runge-Kutta 4th order method
t_values, r_values = runge_kutta_4(f, r0, t0, t_end, h)

# Extract populations of rabbits and foxes
x_values = r_values[:, 0]
y_values = r_values[:, 1]

# Plot the results
plt.plot(t_values, x_values, label="Rabbits (x)", color="green", linestyle="--")
plt.plot(t_values, y_values, label="Foxes (y)", color="pink")
plt.xlabel("Time (t)", fontsize=14)
plt.ylabel("Population", fontsize=14)
plt.title("Lotka-Volterra Predator-Prey Model", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()


###################################################################
# Problem 3: Excercise 8.3
# The Lorenz Equations
###################################################################
# One of the most celebrated sets of differential equations in
# physics is the Lorenz equations:
# dx/dt = sigma(y-x), dy/dt = r*x - y - x*z, dz/dt = x*y - b*z
# where sigma, r, and b are constants.
# a) Write a program to solve the Lorenz equations for the case sigma=10
# r = 28, and b = (8/3) in the range from t=0 to t=50 with initial
# (x,y,z)=(0,1,0). Have your program make a plot of y as a function of time.
# b) Modify your program to produce a plot of z against x.  You should
# see a picture of the famous "strange attractor'' of the Lorenz
# equations, a lop-sided butterfly-shaped plot that never repeats itself.
####################################################################

# Lorenz system of equations
def lorenz(r, sigma, b, r_values):
    x, y, z = r_values
    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])


# Runge-Kutta 4th order method for solving the system
def runge_kutta_4(f, r0, t0, t_end, h, *params):
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


# Parameters
sigma = 10.0
r = 28.0
b = 8.0 / 3.0
r0 = np.array([0.0, 1.0, 0.0])  # Initial conditions for x, y, z
t0 = 0.0  # Start time
t_end = 50.0  # End time
h = 0.01  # Time step

# Solve the Lorenz system using Runge-Kutta 4th order method
t_values, r_values = runge_kutta_4(lorenz, r0, t0, t_end, h, r, sigma, b)

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
plt.scatter(x_values, z_values, c=t_values, cmap="viridis", s=1)  # Scatter plot with color by time
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

###################################################################
# Problem 4 : Excercise 8.5
# The Driven Pendulum
###################################################################
# A pendulum like the one in Exercise 8.4 can be driven by, for
# example, exerting a small oscillating force horizontally on the mass.
# Then the equation of motion for the pendulum becomes:
# d^2theta / dt^2 = -g/l sin(theta) + C*cos(theta)*sin(omega*t)
# where C and Omega are constants.
# Write a program to solve this equation for theta as a function of
# time with l=10cm, C=2(s^-2) and Omega=5(s^-1) and make a plot of theta
# as function of time from t=0 to t=100s. Start the pendulum at rest with
# theta=0 and dtheta/dt=0
# Now change the value of Omega while keeping C the same, to find a value
# for which the pendulum resonates with the driving force and swings widely
# from side to side. Make a plot for this case.
# Create an annimation of the motion of the pendulum
###################################################################

# Constants
g = 9.81  # acceleration due to gravity in m/s^2
ell = 0.1  # length of pendulum in meters
C = 2.0  # driving force coefficient in s^-2
Omega_normal = 5.0  # driving frequency in s^-1 (for the first case)
Omega_resonance = 9.8  # Adjusted to be near natural frequency for resonance
Omegas = [5.0, 7.0, 9.8, 11.0]  # Different driving frequencies for comparison


# Define the system of first-order ODEs
def pendulum_system(t, state, g, ell, C, Omega):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = - (g / ell) * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return np.array([dtheta_dt, domega_dt])


# Solve the system using Runge-Kutta 4th order method
def runge_kutta_4(f, t0, t_end, h, state0, *params):
    t_values = np.arange(t0, t_end + h, h)
    state_values = np.zeros((len(t_values), len(state0)))
    state_values[0] = state0

    for i in range(len(t_values) - 1):
        t = t_values[i]
        state = state_values[i]
        k1 = h * f(t, state, *params)
        k2 = h * f(t + 0.5 * h, state + 0.5 * k1, *params)
        k3 = h * f(t + 0.5 * h, state + 0.5 * k2, *params)
        k4 = h * f(t + h, state + k3, *params)
        state_values[i + 1] = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, state_values


# Initial conditions
theta0 = 0.0  # initial angle (radians)
omega0 = 0.0  # initial angular velocity (rad/s)
state0 = np.array([theta0, omega0])  # initial state

# Time parameters
t0 = 0.0  # start time
t_end = 100.0  # end time
h = 0.01  # time step

# Solve the system for both Omega values
t_values_normal, state_values_normal = runge_kutta_4(pendulum_system, t0, t_end, h, state0, g, ell, C, Omega_normal)
t_values_resonance, state_values_resonance = runge_kutta_4(pendulum_system, t0, t_end, h, state0, g, ell, C, Omega_resonance)

# Extract theta values
theta_values_normal = state_values_normal[:, 0]
theta_values_resonance = state_values_resonance[:, 0]

# Use seaborn for styling
sns.set(style="darkgrid")

# Plot both cases on the same graph
plt.figure(figsize=(10, 6))
sns.lineplot(x=t_values_resonance, y=theta_values_resonance, label=rf"$\Omega = {Omega_resonance}$ s$^{-1}$ (Resonant)", color="pink")
sns.lineplot(x=t_values_normal, y=theta_values_normal, label=rf"$\Omega = {Omega_normal}$ s$^{-1}$ (Non-Resonant)", color="purple")

# Labels and title
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel(r"$\theta$ (radians)", fontsize=14)
plt.title("Comparison of Non-Resonant and Resonant Pendulum Motion", fontsize=16)
plt.legend()
plt.show()

###################################################################
# Addtional animation for a pendulum
###################################################################
"""Animation of the motion of the pendulum for different Omega's"""
# Time parameters
t0 = 0.0  # Start time
t_end = 20.0  # Shortened for animation
h = 0.01  # Time step

# Solve the system for each Omega
solutions = {}
for Omega in Omegas:
    t_values, state_values = runge_kutta_4(pendulum_system, t0, t_end, h, state0, g, ell, C, Omega)
    theta_values = state_values[:, 0]
    x_values = ell * np.sin(theta_values)
    y_values = -ell * np.cos(theta_values)
    solutions[Omega] = (t_values, x_values, y_values)

# Set up the figure with four subplots
sns.set(style="darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axes array to easily iterate
axes = axes.flatten()

# Create animation objects for each subplot
lines = []
for i, Omega in enumerate(Omegas):
    ax = axes[i]
    ax.set_xlim(-ell * 1.2, ell * 1.2)
    ax.set_ylim(-ell * 1.2, ell * 0.2)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(f"Omega = {Omega} s⁻¹")
    line, = ax.plot([], [], "o-", lw=3, markersize=10, markerfacecolor="pink")
    lines.append(line)


# Function to initialize the animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines


# Function to update each animation frame
def update(frame):
    for i, Omega in enumerate(Omegas):
        x_values, y_values = solutions[Omega][1], solutions[Omega][2]
        x, y = x_values[frame], y_values[frame]
        lines[i].set_data([0, x], [0, y])  # Update the pendulum rod and mass position
    return lines


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t_values), init_func=init, interval=5, blit=True)

# Display the animation
plt.show()
