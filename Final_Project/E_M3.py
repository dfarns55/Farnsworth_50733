#!/usr/bin/env python3

"""
E_M3.py

 

Author: Danette Farnsworth
Date: 2025-05-07
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Constants
G = 1.327e11       # Gravitational constant * mass of Sun [km^3/s^2]
AU = 1.496e8       # Astronomical Unit [km]
day = 86400        # Seconds in one day

# Earth and Mars orbital radii
r_earth = 1.0 * AU
r_mars = 1.524 * AU

# Orbital speeds
v_earth = np.sqrt(G / r_earth)
v_mars = np.sqrt(G / r_mars)

# Hohmann transfer details
a_transfer = 0.5 * (r_earth + r_mars)
v_transfer_earth = np.sqrt(G * (2 / r_earth - 1 / a_transfer))
transfer_time = np.pi * np.sqrt(a_transfer**3 / G)

# Time step and total steps
dt = 0.5 * day
stay_duration = 30  # days at Mars before return
steps = int((2 * transfer_time + stay_duration * day) / dt)


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
    earth_pos = np.array([r_earth, 0])
    earth_vel = np.array([0, v_earth])
    theta = np.radians(44)
    mars_pos = r_mars * np.array([np.cos(theta), np.sin(theta)])
    mars_vel = v_mars * np.array([-np.sin(theta), np.cos(theta)])
    sat_pos = np.copy(earth_pos)
    sat_vel = np.array([0, v_transfer_earth])
    return earth_pos, earth_vel, mars_pos, mars_vel, sat_pos, sat_vel


# Load planet and satellite icons
earth_img = mpimg.imread("earth.png")
mars_img = mpimg.imread("mars.png")
sat_img = mpimg.imread("satellite.png")

# Initialize simulation
earth_pos, earth_vel, mars_pos, mars_vel, sat_pos, sat_vel = init_conditions()
earth_traj, mars_traj, sat_traj, times = [], [], [], []

return_initiated = False
stay_counter = 0

for i in range(steps):
    earth_traj.append(earth_pos.copy())
    mars_traj.append(mars_pos.copy())
    sat_traj.append(sat_pos.copy())
    times.append(i * dt / day)

    earth_pos, earth_vel = rk4_step(earth_pos, earth_vel, dt)
    mars_pos, mars_vel = rk4_step(mars_pos, mars_vel, dt)
    sat_pos, sat_vel = rk4_step(sat_pos, sat_vel, dt)

earth_traj = np.array(earth_traj)
mars_traj = np.array(mars_traj)
sat_traj = np.array(sat_traj)
times = np.array(times)

# Create background circular orbits
theta_vals = np.linspace(0, 2 * np.pi, 300)
earth_orbit = r_earth * np.column_stack((np.cos(theta_vals), np.sin(theta_vals)))
mars_orbit = r_mars * np.column_stack((np.cos(theta_vals), np.sin(theta_vals)))

# Setup plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-2 * AU, 2 * AU)
ax.set_ylim(-2 * AU, 2 * AU)
ax.axis("off")

ax.plot(earth_orbit[:, 0], earth_orbit[:, 1], "b--", lw=0.8)
ax.plot(mars_orbit[:, 0], mars_orbit[:, 1], "r--", lw=0.8)
ax.plot(0, 0, "yo", markersize=10)  # Sun

# Animated objects
line_sat, = ax.plot([], [], "k-", lw=1)

# Icon objects
earth_icon = OffsetImage(earth_img, zoom=0.03)
mars_icon = OffsetImage(mars_img, zoom=0.03)
sat_icon = OffsetImage(sat_img, zoom=0.025)

earth_ab = AnnotationBbox(earth_icon, (0, 0), frameon=False)
mars_ab = AnnotationBbox(mars_icon, (0, 0), frameon=False)
sat_ab = AnnotationBbox(sat_icon, (0, 0), frameon=False)

ax.add_artist(earth_ab)
ax.add_artist(mars_ab)
ax.add_artist(sat_ab)

# Title and timestamp
ax.text(0, 2.1 * AU, "Earth to Mars Transfer", fontsize=12, ha="center")
timestamp = ax.text(0, -2.1 * AU, "", fontsize=10, ha="center")


# Animation update
def update(frame):
    line_sat.set_data(sat_traj[:frame, 0], sat_traj[:frame, 1])
    earth_ab.xybox = (earth_traj[frame, 0], earth_traj[frame, 1])
    mars_ab.xybox = (mars_traj[frame, 0], mars_traj[frame, 1])
    sat_ab.xybox = (sat_traj[frame, 0], sat_traj[frame, 1])
    timestamp.set_text(f"t = {times[frame]:.1f} days")
    return line_sat, earth_ab, mars_ab, sat_ab, timestamp


# Animate and save
anim = FuncAnimation(fig, update, frames=steps, interval=20)
anim.save("earth_to_mars.gif", writer=PillowWriter(fps=30))
