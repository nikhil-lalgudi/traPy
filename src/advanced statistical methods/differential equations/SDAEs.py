import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Decorator for a single step
def step(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Stochastic Differential-Algebraic Equations (SDAEs) Solver
class SDAE:
    def __init__(self, f, g, h, y0, z0, t0, t1, dt, noise_intensity):
        self.f = f  # Differential part
        self.g = g  # Stochastic part
        self.h = h  # Algebraic constraints
        self.y = y0
        self.z = z0
        self.t = t0
        self.t1 = t1
        self.dt = dt
        self.noise_intensity = noise_intensity
        self.steps = int((t1 - t0) / dt)
        self.trajectory_y = [y0]
        self.trajectory_z = [z0]
        self.time = [t0]
    
    @step
    def euler_maruyama_step(self, y, z, dt):
        noise = self.noise_intensity * np.sqrt(dt) * np.random.normal(size=y.shape)
        y_new = y + self.f(y, z) * dt + self.g(y, z) * noise
        z_new = solve(self.h(y_new, z), z)
        return y_new, z_new
    
    def solve(self):
        for _ in range(self.steps):
            self.y, self.z = self.euler_maruyama_step(self.y, self.z, self.dt)
            self.t += self.dt
            self.trajectory_y.append(self.y)
            self.trajectory_z.append(self.z)
            self.time.append(self.t)
        return np.array(self.time), np.array(self.trajectory_y), np.array(self.trajectory_z)

# Example usage within a library
def example_usage():
    # Define differential, stochastic, and algebraic constraint functions
    def f(y, z):
        return np.array([y[1], -y[0] + z[0]])
    
    def g(y, z):
        return np.array([0.0, 0.1])
    
    def h(y, z):
        return np.array([[1, -1], [1, 1]])
    
    # Initial conditions
    y0 = np.array([1.0, 0.0])
    z0 = np.array([0.0, 1.0])
    t0 = 2.3
    t1 = 4.5
    dt = 0.01
    noise_intensity = 0.1
    
    # Create SDAE solver
    sdae_solver = SDAE(f, g, h, y0, z0, t0, t1, dt, noise_intensity)
    
    # Solve SDAE
    time, trajectory_y, trajectory_z = sdae_solver.solve()
    
    return time, trajectory_y, trajectory_z

# Functions to process and visualize the results
def process_sdae_solution(time, trajectory_y, trajectory_z):
    # Basic statistics or analysis can be added here
    return time, trajectory_y, trajectory_z

import numpy as np
import matplotlib.pyplot as plt

def stratonovich_integral(f, t, W):
    n = len(t)
    if len(W) != n:
        raise ValueError("Time points and Wiener process must have the same length.")
    
    # Initialize the integral array
    integral = np.zeros(n)
    
    # Midpoint approximation for the Stratonovich integral
    for i in range(1, n):
        delta_t = t[i] - t[i-1]
        delta_W = W[i] - W[i-1]
        midpoint_W = 0.5 * (W[i] + W[i-1])
        integral[i] = integral[i-1] + f(midpoint_W) * delta_W
    
    return integral

# Example usage
np.random.seed(42)  # For reproducibility
n_steps = 1000
T = 1.0
t = np.linspace(0, T, n_steps + 1)
delta_t = T / n_steps
W = np.cumsum(np.sqrt(delta_t) * np.random.randn(n_steps + 1))

# Define the function to integrate
def f(W):
    return W**2

# Compute the Stratonovich integral
integral = stratonovich_integral(f, t, W)

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot the Wiener process
plt.subplot(3, 1, 1)
plt.plot(t, W, label='Wiener Process $W(t)$')
plt.title('Wiener Process')
plt.xlabel('Time $t$')
plt.ylabel('$W(t)$')
plt.legend()

# Plot the function f(W(t))
plt.subplot(3, 1, 2)
plt.plot(t, f(W), label='$f(W(t)) = W(t)^2$', color='orange')
plt.title('Function $f(W(t))$')
plt.xlabel('Time $t$')
plt.ylabel('$f(W(t))$')
plt.legend()

# Plot the Stratonovich integral
plt.subplot(3, 1, 3)
plt.plot(t, integral, label='Stratonovich Integral', color='green')
plt.title('Stratonovich Integral')
plt.xlabel('Time $t$')
plt.ylabel('Integral')
plt.legend()

plt.tight_layout()
plt.show()

def plot_sdae_solution(time, trajectory_y, trajectory_z):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, trajectory_y)
    plt.title('SDAE Solution - Differential Variables')
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, trajectory_z)
    plt.title('SDAE Solution - Algebraic Variables')
    plt.xlabel('Time')
    plt.ylabel('z')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    time, trajectory_y, trajectory_z = example_usage()
    
    # Process results
    processed_time, processed_trajectory_y, processed_trajectory_z = process_sdae_solution(time, trajectory_y, trajectory_z)
    
    # Plot results
    plot_sdae_solution(processed_time, processed_trajectory_y, processed_trajectory_z)
