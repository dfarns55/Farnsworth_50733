#!/usr/bin/env python3

"""
Planetary_Systems.py

Author: Danette Farnsworth
Date: 2025-05-04
"""

# Runge-Kutta Planetary Simulation with Energy Plotting and GIF Animation (Optimized)

import os
import pickle as pkl

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use a non-interactive backend for image generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

# Constants
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2

# PARAMETERS
TIMESTEP = 24  # Timestep in hours
DAYS_TO_RUN = 10_759  # Total simulation time in days (~29.5 years, one Saturn orbit)


# Define a class for celestial bodies
class Body:
    """
    Represents a celestial body (e.g., a planet or the Sun) with physical and tracking properties:
    - name: Identifier for the body
    - mass: Gravitational mass
    - position: Current 3D position vector
    - velocity: Current 3D velocity vector
    - positions: List of recorded positions over time for plotting
    - kinetic_energy: Time-series list of kinetic energy values
    - potential_energy: Time-series list of potential energy values
    - total_energy: Time-series of KE + PE
    - revolutions: Number of full orbits completed (incremented heuristically)
    - prev_angle: Previous angle from the origin, used for revolution counting
    """

    def __init__(self, name, mass, position, velocity):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.positions = [self.position.copy()]
        self.kinetic_energy = []
        self.potential_energy = []
        self.total_energy = []
        self.revolutions = 0
        self.prev_angle = np.arctan2(position[1], position[0])


# Compute gravitational accelerations between all body pairs
def compute_accelerations(bodies):
    """
    Compute the net gravitational acceleration on each body due to all other bodies.
    """
    n = len(bodies)
    acc = [np.zeros(3) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                r = bodies[j].position - bodies[i].position
                dist = np.linalg.norm(r)
                if dist > 0:
                    acc[i] += G * bodies[j].mass * r / dist**3
    return acc


# Runge-Kutta 4th order step for all bodies
def rk4_step(bodies, dt):
    """
    Perform a single Runge-Kutta 4th-order (RK4) update for positions and velocities.
    """
    n = len(bodies)
    positions = np.array([b.position for b in bodies])
    velocities = np.array([b.velocity for b in bodies])
    masses = np.array([b.mass for b in bodies])

    def a(pos):
        temp_bodies = [Body(bodies[i].name, masses[i], pos[i], velocities[i]) for i in range(n)]
        return compute_accelerations(temp_bodies)

    # RK4 integration for velocity and position
    k1v = np.array(a(positions)) * dt
    k1x = velocities * dt

    k2v = np.array(a(positions + 0.5 * k1x)) * dt
    k2x = (velocities + 0.5 * k1v) * dt

    k3v = np.array(a(positions + 0.5 * k2x)) * dt
    k3x = (velocities + 0.5 * k2v) * dt

    k4v = np.array(a(positions + k3x)) * dt
    k4x = (velocities + k3v) * dt

    # Update positions and velocities using RK4 formula
    for i in range(n):
        bodies[i].position += (k1x[i] + 2 * k2x[i] + 2 * k3x[i] + k4x[i]) / 6
        bodies[i].velocity += (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6
        bodies[i].positions.append(bodies[i].position.copy())


# Compute energy and revolutions per body at given step
def update_energy_and_revolutions(body, position, velocity, central_body):
    """
    Compute KE, PE, Total energy and track orbital revolutions.
    """
    v_mag = np.linalg.norm(velocity)
    ke = 0.5 * body.mass * v_mag**2  # Kinetic energy
    r_vec = position - central_body.position
    r = np.linalg.norm(r_vec)
    pe = -G * body.mass * central_body.mass / r if r != 0 else 0  # Potential energy
    body.kinetic_energy.append(ke)
    body.potential_energy.append(pe)
    body.total_energy.append(ke + pe)  # Total energy

    # Count revolutions based on angle crossing
    angle = np.arctan2(position[1], position[0])
    dtheta = angle - body.prev_angle
    if dtheta < -np.pi:
        dtheta += 2 * np.pi
    elif dtheta > np.pi:
        dtheta -= 2 * np.pi
    if dtheta > 0 and body.prev_angle > 0 and angle < 0:
        body.revolutions += 1
    body.prev_angle = angle


# Compute and plot total system energy for conservation verification
def plot_system_total_energy(bodies, dt):
    """
    Compute and plot Total System energy for Conservation
    """
    print("Plotting total system energy for conservation check...", end="", flush=True)
    total_energies = []
    times = []
    steps = len(bodies[0].positions)

    for step in range(steps):
        ke_total = 0
        pe_total = 0
        for i in range(len(bodies)):
            v_mag = np.linalg.norm(bodies[i].velocity)
            ke_total += 0.5 * bodies[i].mass * v_mag**2
            for j in range(i + 1, len(bodies)):
                r = np.linalg.norm(bodies[i].position - bodies[j].position)
                if r != 0:
                    pe_total -= G * bodies[i].mass * bodies[j].mass / r
        total_energies.append(ke_total + pe_total)
        times.append(step * dt / 86400)  # Convert time to days

    plt.figure(figsize=(10, 6))
    plt.plot(times, total_energies, label="Total System Energy")
    plt.xlabel("Time (days)")
    plt.ylabel("Energy (Joules)")
    plt.title("Total System Energy Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("total_system_energy.png")
    plt.close()
    print("Done.")


# Perform the full simulation
def simulate(bodies, num_steps, dt, store_every=10):
    """
    Run the main simulation loop for all celestial bodies using RK4 integration.
    Intermediate results stored every 'store_every' steps for efficiency.
    """
    print("Starting simulation...")
    for step in range(num_steps):
        rk4_step(bodies, dt)
        if step % store_every == 0:
            for b in bodies:
                update_energy_and_revolutions(b, b.position, b.velocity, bodies[0])
        if step % (num_steps // 10) == 0:
            print(f"  Progress: {100 * step // num_steps}% complete")
    print("Simulation complete.")


# Plot KE, PE, and Total Energy individually per planet
def plot_energy_individual(bodies, dt):
    """
    Plot KE, PE, and Total Energy for each planet in subplots.
    """
    print("Plotting energies...", end="", flush=True)

    # Plot kinetic energy
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, body in enumerate(bodies[1:]):
        ax = axes[i]
        time = np.arange(len(body.kinetic_energy)) * 86400 / dt
        ax.plot(time, body.kinetic_energy)
        ax.set_title(body.name)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Kinetic Energy (Joules)")
        ax.grid(True)
    fig.suptitle("Kinetic Energy")
    plt.tight_layout()
    plt.savefig("kinetic_energy_plot_individual.png")
    plt.close()

    # Plot potential energy
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, body in enumerate(bodies[1:]):
        ax = axes[i]
        time = np.arange(len(body.kinetic_energy)) * 86400 / dt
        ax.plot(time, body.potential_energy, color="purple")
        ax.set_title(body.name)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Potential Energy (Joules")
        ax.grid(True)
    fig.suptitle("Potential Energy")
    plt.tight_layout()
    plt.savefig("potential_energy_plot_individual.png")
    plt.close()

    # Plot total energy
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    for i, body in enumerate(bodies[1:]):
        ax = axes[i]
        time = np.arange(len(body.kinetic_energy)) * 86400 / dt
        ax.plot(time, body.total_energy, color="green")
        ax.set_title(body.name)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Total Energy (Joules")
        ax.grid(True)
    fig.suptitle("Total Energy")
    plt.tight_layout()
    plt.savefig("total_energy_plot_individual.png")
    plt.close()
    print("Done.")


# Create an animated GIF of the planetary orbits
def save_orbit_gif(bodies, dt, filename="planetary_orbits.gif", scale=1.6e12, filter_names=None, max_frames=1000):
    """
    Create an animated GIF of planetary orbits using saved positions.
    Limited to max_frames for speed and memory usage.
    """
    print("Saving orbits to gif...", end="", flush=True)
    colors = ["gray", "orange", "blue", "red", "brown", "gold", "green"]
    if filter_names:
        bodies = [b for b in bodies if b.name in filter_names or b.name == "Sun"]
    fig = plt.figure(figsize=(10, 10), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    frames = []
    total_frames = min(len(bodies[0].positions), max_frames)
    for frame in range(total_frames):
        ax.clear()
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        ax.set_title(f"Day {frame * dt / 86400:.1f}")
        for i, body in enumerate(bodies):
            if frame < len(body.positions):
                pos = np.array(body.positions[frame])
                trail = np.array(body.positions[:frame + 1])
                ax.plot(trail[:, 0], trail[:, 1], color=colors[i % len(colors)], alpha=0.5)
                ax.plot(pos[0], pos[1], "o", color=colors[i % len(colors)], label=body.name)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize="small")
        canvas.draw()
        renderer = canvas.get_renderer()
        image = np.frombuffer(renderer.buffer_rgba(), dtype="uint8")
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(Image.fromarray(image))
    frames[0].save(filename, format="GIF", save_all=True, append_images=frames[1:], duration=20, loop=0)
    plt.close(fig)
    print("Done.")


# ----------------------------------------------
# Setup celestial bodies and simulation params
# ----------------------------------------------
sun = Body("Sun", 1.989e30, [0, 0, 0], [0, 0, 0])
mercury = Body("Mercury", 3.285e23, [5.791e10, 0, 0], [0, 47400, 0])
earth = Body("Earth", 5.972e24, [1.496e11, 0, 0], [0, 29780, 0])
mars = Body("Mars", 6.39e23, [2.279e11, 0, 0], [0, 24077, 0])
venus = Body("Venus", 4.867e24, [1.082e11, 0, 0], [0, 35020, 0])
jupiter = Body("Jupiter", 1.898e27, [7.785e11, 0, 0], [0, 13070, 0])
saturn = Body("Saturn", 5.683e26, [1.429e12, 0, 0], [0, 9680, 0])

# Add all planets to simulation list
bodies = [sun, mercury, venus, earth, mars, jupiter, saturn]

# Set simulation parameters
dt = 60 * 60 * TIMESTEP  # Convert timestep to seconds
num_steps = int(DAYS_TO_RUN * 86_400 / dt)  # Total number of steps

# Load previous simulation or run new one
if os.path.exists("simulation.pkl"):
    print("Loading completed simulation from file.")
    with open("simulation.pkl", "rb") as fl:
        bodies = pkl.load(fl)
else:
    simulate(bodies, num_steps, dt, store_every=10)
    print("Saving completed simulation to file...", end="")
    with open("simulation.pkl", "wb") as fl:
        pkl.dump(bodies, fl)
    print("done.")
# --------------------------------
# Plot results and generate GIF
# --------------------------------
plot_energy_individual(bodies, dt)
plot_system_total_energy(bodies, dt)  # Plot true system total energy
save_orbit_gif(bodies, dt, "planetary_orbits.gif", scale=2.0e12, max_frames=1500)
