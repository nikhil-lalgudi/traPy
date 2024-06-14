import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class DAE:
    def __init__(self, f, g, y0, z0, t0, t1, dt):
        self.f = f  # Differential part
        self.g = g  # Algebraic constraints
        self.y0 = y0
        self.z0 = z0
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.time = np.arange(t0, t1 + dt, dt)
    
    def residual(self, t, yz):
        n = len(self.y0)
        y = yz[:n]
        z = yz[n:]
        y_dot = self.f(t, y, z)
        g_val = self.g(t, y, z)
        return np.concatenate([y_dot, g_val])

    def solve(self):
        sol = solve_ivp(self.residual, [self.t0, self.t1], np.concatenate([self.y0, self.z0]), t_eval=self.time, method='BDF')
        n = len(self.y0)
        y_sol = sol.y[:n, :]
        z_sol = sol.y[n:, :]
        return sol.t, y_sol, z_sol

# Example usage within a library
def example_usage():
    # Define differential and algebraic parts
    def f(t, y, z):
        return np.array([y[1], -y[0] + z[0]])
    
    def g(t, y, z):
        return np.array([y[0] + z[0] - 1])
    
    # Initial conditions
    y0 = np.array([0.0, 1.0])
    z0 = np.array([1.0])
    t0 = 0.0
    t1 = 10.0
    dt = 0.01
    
    # Create DAE solver
    dae_solver = DAE(f, g, y0, z0, t0, t1, dt)
    
    # Solve DAE
    time, y_sol, z_sol = dae_solver.solve()
    
    return time, y_sol, z_sol

# Functions to process and visualize the results
def process_dae_solution(time, y_sol, z_sol):
    # Basic statistics or analysis can be added here
    return time, y_sol, z_sol

def plot_dae_solution(time, y_sol, z_sol):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, y_sol.T)
    plt.title('DAE Solution - Differential Variables')
    plt.xlabel('Time')
    plt.ylabel('y')
    plt.legend(['y1', 'y2'])
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, z_sol.T)
    plt.title('DAE Solution - Algebraic Variables')
    plt.xlabel('Time')
    plt.ylabel('z')
    plt.legend(['z1'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    time, y_sol, z_sol = example_usage()
    
    # Process results
    processed_time, processed_y_sol, processed_z_sol = process_dae_solution(time, y_sol, z_sol)
    
    # Plot results
    plot_dae_solution(processed_time, processed_y_sol, processed_z_sol)
