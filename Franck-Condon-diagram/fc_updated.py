import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Define the parameters
m = 1.0        # Mass
omega_g = 1.0  # Frequency of ground state
omega_e = 1.0  # Frequency of excited state
x_shift = 0.7  # Displacement in excited state potential
x = np.linspace(-3, 4.5, 1000)  # Position array

# Harmonic potential energy functions
V_g = 0.5 * m * omega_g**2 * x**2
V_e = 0.5 * m * omega_e**2 * (x - x_shift)**2 + 3  # Shift excited state up for clarity
# Ensure V_g does not exceed 3
V_g[V_g > 3] = None
V_e[V_e > 6] = None

# Function to compute the wavefunction of a harmonic oscillator
def harmonic_wavefunction(n, x, m, omega, hbar=0.5):
    Hn = np.polynomial.hermite.Hermite([0]*n + [1])(np.sqrt(m*omega/hbar) * x)
    psi = np.exp(-m*omega*x**2/(2*hbar)) * Hn
    normalization = np.sqrt(np.sqrt(m*omega/(np.pi/hbar)) / (2**n * np.math.factorial(n)))
    return normalization * psi

# Number of vibrational states to consider
n_max = 6

# Compute wavefunctions and energies
psi_g = [0.2 * harmonic_wavefunction(n, x, m, omega_g) for n in range(n_max + 1)]
E_g = [0.2 * omega_g * (2 * n + 1) for n in range(n_max + 1)]

psi_e = [0.2 * harmonic_wavefunction(n, x - x_shift, m, omega_e) for n in range(n_max + 1)]
E_e = [0.2 * omega_e * (2 * n + 1) + 3 for n in range(n_max + 1)]

# Plot the potentials
plt.figure(figsize=(3, 8))
plt.plot(x, V_g, color='gray', linewidth=1)
plt.plot(x, V_e, color='gray', linewidth=1)

# Plot the vibrational levels and wavefunctions
for n in range(n_max + 1):
    plt.plot(x, psi_g[n] + E_g[n], color='gray', linewidth=1)
    plt.plot(x, psi_e[n] + E_e[n], color='gray', linewidth=1)
    plt.hlines(E_g[n], -3, 4, colors='gray', linestyles='dotted', linewidth=1)
    plt.hlines(E_e[n], -3, 4, colors='gray', linestyles='dotted', linewidth=1)
    plt.text(3.6, E_g[n], f'n={n}', fontsize=10, verticalalignment='bottom', horizontalalignment='right', color='gray')
    plt.text(3.6, E_e[n], f'm={n}', fontsize=10, verticalalignment='bottom', horizontalalignment='right', color='gray')

# Function to add dashed arrows
def add_dashed_arrow(ax, start, end, color, linewidth):
    arrow = FancyArrowPatch(start, end, color=color, linewidth=linewidth, linestyle='dashed',
                            arrowstyle='-|>', mutation_scale=10)
    ax.add_patch(arrow)

# Add the arrows
ax = plt.gca()
add_dashed_arrow(ax, (-x_shift, E_g[0]), (-x_shift, E_e[n_max]), 'green', 1)
add_dashed_arrow(ax, (-x_shift, E_e[n_max]), (x_shift, E_e[0]), 'black', 1)
for i in range(6):
    add_dashed_arrow(ax, (0.3 * x_shift * (i + 1), E_e[0]), (0.3 * x_shift * (i + 1), E_g[i]), 'red', 1)

# Add labels and legend with increased font size
plt.xlabel('Position', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.ylim(-0.1, 6.1)
plt.xlim(-2.5, 3.6)

# Remove ticks and tick labels
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

# Save the figure
import os
output_dir = r'C:\Users\Saroj Chand\Documents\GitHub\nv-center-network-analysis\data\FC_diagrams'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "FC.png"), transparent=True, dpi=300)

# Show the plot
plt.grid(False)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# Set the Seaborn style
# sns.set(style="whitegrid")

# Define the parameters
m = 1.0        # Mass
omega_g = 1.0  # Frequency of ground state
omega_e = 1.0  # Frequency of excited state
x_shift = 0.7  # Displacement in excited state potential
x = np.linspace(-3, 4.5, 1000)  # Position array

# Harmonic potential energy functions
V_g = 0.5 * m * omega_g**2 * x**2
V_e = 0.5 * m * omega_e**2 * (x - x_shift)**2 + 3  # Shift excited state up for clarity
# Ensure V_g does not exceed 3
V_g[V_g > 3] = None
V_e[V_e > 6] = None

# Function to compute the wavefunction of a harmonic oscillator
def harmonic_wavefunction(n, x, m, omega, hbar=0.5):
    Hn = np.polynomial.hermite.Hermite([0]*n + [1])(np.sqrt(m*omega/hbar) * x)
    psi = np.exp(-m*omega*x**2/(2*hbar)) * Hn
    normalization = np.sqrt(np.sqrt(m*omega/(np.pi/hbar)) / (2**n * np.math.factorial(n)))
    return normalization * psi

# Number of vibrational states to consider
n_max = 3

# Compute wavefunctions and energies
psi_g = [0.2 * harmonic_wavefunction(n, x, m, omega_g) for n in range(n_max + 1)]
E_g = [0.39 * omega_g * (2 * n + 1) for n in range(n_max + 1)]

psi_e = [0.2 * harmonic_wavefunction(n, x - x_shift, m, omega_e) for n in range(n_max + 1)]
E_e = [0.39 * omega_e * (2 * n + 1) + 3 for n in range(n_max + 1)]

# Plot the potentials
plt.figure(figsize=(3, 8))
plt.plot(x, V_g, color='blue', linewidth=1)
plt.plot(x, V_e, color='red', linewidth=1)

# Plot the vibrational levels and wavefunctions
for n in range(n_max + 1):
    plt.plot(x, psi_g[n] + E_g[n], color='blue', linewidth=1)
    plt.plot(x, psi_e[n] + E_e[n], color='red', linewidth=1)
    plt.hlines(E_g[n], -3, 4, colors='blue', linestyles='dotted', linewidth=1)
    plt.hlines(E_e[n], -3, 4, colors='red', linestyles='dotted', linewidth=1)
    plt.text(3.6, E_g[n], f'n={n}', fontsize=10, verticalalignment='bottom', horizontalalignment='right', color='blue')
    plt.text(3.6, E_e[n], f'm={n}', fontsize=10, verticalalignment='bottom', horizontalalignment='right', color='red')

# Function to add dashed arrows
def add_dashed_arrow(ax, start, end, color, linewidth):
    arrow = FancyArrowPatch(start, end, color=color, linewidth=linewidth, linestyle='dashed',
                            arrowstyle='-|>', mutation_scale=10)
    ax.add_patch(arrow)

# Add the arrows
ax = plt.gca()
add_dashed_arrow(ax, (-x_shift, E_g[0]), (-x_shift, E_e[n_max]), 'green', 1)
add_dashed_arrow(ax, (-x_shift, E_e[n_max]), (x_shift, E_e[0]), 'gray', 1)
for i in range(4):
    add_dashed_arrow(ax, (0.5 * x_shift * (i + 1), E_e[0]), (0.5 * x_shift * (i + 1), E_g[i]), 'red', 1)

# Add labels and legend with increased font size
plt.xlabel('Position', fontsize=14)
plt.ylabel('Energy', fontsize=14)
plt.ylim(-0.3, 6.3)
plt.xlim(-2.5, 3.6)

# Remove ticks and tick labels
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

# Add legend
plt.legend()

# Save the figure
import os
output_dir = r'C:\Users\Saroj Chand\Documents\GitHub\nv-center-network-analysis\data\FC_diagrams'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "FC_seaborn.png"), transparent=True, dpi=300)

# Show the plot
plt.grid(False)
plt.show()
