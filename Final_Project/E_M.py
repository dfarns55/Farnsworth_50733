#!/usr/bin/env python3

"""
E_M.py

 

Author: Danette Farnsworth
Date: 2025-05-06
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants
G = 1.327e11       # Gravitational constant * mass of Sun [km^3/s^2]
AU = 1.496e8       # Astronomical Unit [km]
day = 86400        # Seconds in one day

# Orbital radii
r_earth = 1.0 * AU
r_mars = 1.524 * AU

# Orbital speeds
v_earth = np.sqrt(G / r_earth)
v_mars = np.sqrt(G / r_mars)

# Hohmann transfer orbit
a_transfer = 0.5 * (r_earth + r_mars)
v_transfer_earth = np.sqrt(G * (2 / r_earth - 1 / a_transfer))
transfer_time = np.pi * np.sqrt(a_transfer**3 / G)

# Time step and total steps
dt = 0.5 * day
steps = int(1.2 * transfer_time / dt)


# RK4 integrator
def rk4_step(pos, vel, dt):
    def accel(p):
        r = np.linalg.norm(p)
        return -G * p / r**3

    a1 = accel(pos)
    v1 = vel
    a2 = accel(pos + 0.5 * dt * v1)
    v2 = vel + 0.5 * dt * a1
    a3 = accel(pos + 0.5 * dt * v2)
    v3 = vel + 0.5 * dt * a2
    a4 = accel(pos + dt * v3)
    v4 = vel + dt * a3

    new_pos = pos + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    new_vel = vel + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6
    return new_pos, new_vel


# Initial conditions
def init_conditions():
    # Earth at (r, 0) moving +y
    earth_pos = np.array([r_earth, 0])
    earth_vel = np.array([0, v_earth])

    # Mars 44 degrees ahead
    theta = np.radians(44)
    mars_pos = r_mars * np.array([np.cos(theta), np.sin(theta)])
    mars_vel = v_mars * np.array([-np.sin(theta), np.cos(theta)])

    # Satellite at Earth, boosted
    sat_pos = np.copy(earth_pos)
    sat_vel = np.array([0, v_transfer_earth])

    return earth_pos, earth_vel, mars_pos, mars_vel, sat_pos, sat_vel


# Simulate motion
earth_pos, earth_vel, mars_pos, mars_vel, sat_pos, sat_vel = init_conditions()
earth_traj, mars_traj, sat_traj = [], [], []

for _ in range(steps):
    earth_traj.append(earth_pos.copy())
    mars_traj.append(mars_pos.copy())
    sat_traj.append(sat_pos.copy())
    earth_pos, earth_vel = rk4_step(earth_pos, earth_vel, dt)
    mars_pos, mars_vel = rk4_step(mars_pos, mars_vel, dt)
    sat_pos, sat_vel = rk4_step(sat_pos, sat_vel, dt)

earth_traj = np.array(earth_traj)
mars_traj = np.array(mars_traj)
sat_traj = np.array(sat_traj)

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-2 * AU, 2 * AU)
ax.set_ylim(-2 * AU, 2 * AU)

line_earth, = ax.plot([], [], "b-", lw=1, label="Earth")
line_mars, = ax.plot([], [], "r-", lw=1, label="Mars")
line_sat, = ax.plot([], [], "k-", lw=1, label="Satellite")
point_sun = ax.plot(0, 0, "yo", markersize=8, label="Sun")
point_earth, = ax.plot([], [], "bo")
point_mars, = ax.plot([], [], "ro")
point_sat, = ax.plot([], [], "ko")
ax.legend(loc="upper right")


# Update function for animation
def update(frame):
    line_earth.set_data(earth_traj[:frame, 0], earth_traj[:frame, 1])
    line_mars.set_data(mars_traj[:frame, 0], mars_traj[:frame, 1])
    line_sat.set_data(sat_traj[:frame, 0], sat_traj[:frame, 1])
    point_earth.set_data([earth_traj[frame, 0]], [earth_traj[frame, 1]])
    point_mars.set_data([mars_traj[frame, 0]], [mars_traj[frame, 1]])
    point_sat.set_data([sat_traj[frame, 0]], [sat_traj[frame, 1]])
    return line_earth, line_mars, line_sat, point_earth, point_mars, point_sat


# Create animation
anim = FuncAnimation(fig, update, frames=steps, interval=20)

# Save animation as GIF (in current directory)
anim.save("earth_to_mars_hohmann.gif", writer=PillowWriter(fps=30))
