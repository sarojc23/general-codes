import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
m = 1.0        # Mass
omega_g = 1.0  # Frequency of ground state
omega_e = 1.0  # Frequency of excited state
x_shift = 1.0  # Displacement in excited state potential
x = np.linspace(-3, 4, 1000)  # Position array

# Harmonic potential energy functions normalized to a maximum height of 3
V_g = 3 * (0.5 * m * omega_g**2 * x**2) / np.max(0.5 * m * omega_g**2 * x**2)
V_e = 3 * (0.5 * m * omega_e**2 * (x - x_shift)**2) / np.max(0.5 * m * omega_e**2 * (x - x_shift)**2)

# Shift excited state potential up for clarity
V_e += 3

# Function to compute the wavefunction of a harmonic oscillator
def harmonic_wavefunction(n, x, m, omega, hbar=0.5):
    Hn = np.polynomial.hermite.Hermite([0]*n + [1])(np.sqrt(m*omega/hbar) * x)
    psi = np.exp(-m*omega*x**2/(2*hbar)) * Hn
    normalization = np.sqrt(np.sqrt(m*omega/(np.pi/hbar)) / (2**n * np.math.factorial(n)))
    return normalization * psi

# Number of vibrational states to consider
n_max = 2

# Compute wavefunctions and energies
psi_g = [0.2 * harmonic_wavefunction(n, x, m, omega_g) for n in range(n_max + 1)]
E_g = [0.5 * omega_g * (2 * n + 1) for n in range(n_max + 1)]

psi_e = [0.2 * harmonic_wavefunction(n, x - x_shift, m, omega_e) for n in range(n_max + 1)]
E_e = [0.5 * omega_e * (2 * n + 1) + 3 for n in range(n_max + 1)]

# Plot the potentials
plt.figure(figsize=(6, 8))
plt.plot(x, V_g, label='Ground State Potential', color='blue')
plt.plot(x, V_e, label='Excited State Potential', color='red')

# Plot the vibrational levels and wavefunctions
for n in range(n_max + 1):
    plt.plot(x, psi_g[n] + E_g[n], color='blue')
    plt.plot(x, psi_e[n] + E_e[n], color='red')
    plt.hlines(E_g[n], -3, 4, colors='blue', linestyles='dotted')
    plt.hlines(E_e[n], -3, 4, colors='red', linestyles='dotted')

# Plot vertical transitions
for n in range(n_max + 1):
    for m in range(n_max + 1):
        if n == 0 or m == 0:  # Only plot transitions to/from the ground state vibrational level for clarity
            plt.vlines(x_shift, E_g[n], E_e[m], colors='gray', linestyles='dashed')

# Add labels and legend
plt.xlabel('Position')
plt.ylabel('Energy')
plt.title('Franck-Condon Diagram')
plt.legend()
plt.ylim(0, 6)
plt.xlim(-3, 4)
plt.grid(False)
plt.show()
