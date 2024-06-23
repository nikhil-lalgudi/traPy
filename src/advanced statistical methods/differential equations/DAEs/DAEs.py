import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class DAESolver:
    def __init__(self, dae_func, alg_func, y0, z0, t):
        """
        Initialize the DAE solver.
        
        Parameters:
        dae_func (callable): Function defining the differential equations.
        alg_func (callable): Function defining the algebraic equations.
        y0 (ndarray): Initial conditions for differential variables.
        z0 (ndarray): Initial conditions for algebraic variables.
        t (ndarray): Time points at which to solve.
        """
        self.dae_func = dae_func
        self.alg_func = alg_func
        self.y0 = y0
        self.z0 = z0
        self.t = t

    def residual(self, y, t):
        """
        Compute the residual of the DAE system.
        
        Parameters:
        y (ndarray): Combined state vector (differential + algebraic variables).
        t (float): Current time point.
        
        Returns:
        res (ndarray): Residual vector.
        """
        y_diff = y[:len(self.y0)]
        y_alg = y[len(self.y0):]
        dy_dt = self.dae_func(y_diff, y_alg, t)
        alg_eqs = self.alg_func(y_diff, y_alg, t)
        return np.concatenate((dy_dt, alg_eqs))

    def solve(self):
        """
        Solve the DAE system.
        
        Returns:
        sol (ndarray): Solution array.
        """
        y0_combined = np.concatenate((self.y0, self.z0))
        sol = odeint(self.residual, y0_combined, self.t)
        self.y_solution = sol[:, :len(self.y0)]
        self.z_solution = sol[:, len(self.y0):]
        return self.t, self.y_solution, self.z_solution

    def plot_solution(self):
        """
        Plot the solution of the DAE system.
        """
        plt.figure(figsize=(12, 6))

        for i in range(self.y_solution.shape[1]):
            plt.plot(self.t, self.y_solution[:, i], label=f'Differential variable y{i+1}')
        
        for i in range(self.z_solution.shape[1]):
            plt.plot(self.t, self.z_solution[:, i], label=f'Algebraic variable z{i+1}', linestyle='--')
        
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.legend()
        plt.title('DAE Solution')
        plt.show()

# Example usage of the DAESolver library
if __name__ == "__main__":
    # Define the DAE system (simple example)
    def dae_func(y, z, t):
        # dy1/dt = z1
        dy1_dt = z[0]
        return [dy1_dt]

    def alg_func(y, z, t):
        # 0 = y1^2 + z1^2 - 1 (circle constraint)
        return [y[0]**2 + z[0]**2 - 1]

    # Initial conditions
    y0 = [0.5]  # Initial condition for y1
    z0 = [0.2]  # Initial condition for z1 (satisfies y1^2 + z1^2 = 1)
    t = np.linspace(0, 10, 100)  # Time points

    # Create DAESolver instance
    dae_solver = DAESolver(dae_func, alg_func, y0, z0, t)

    # Solve the DAE
    t, y_solution, z_solution = dae_solver.solve()

    # Plot the solution
    dae_solver.plot_solution()
