import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Finite Difference Method (FDM) for PDEs
class FDM_PDE:
    def __init__(self, L, T, Nx, Nt, alpha):
        self.L = L      # Length of the spatial domain
        self.T = T      # Total time
        self.Nx = Nx    # Number of spatial steps
        self.Nt = Nt    # Number of time steps
        self.alpha = alpha  # Diffusion coefficient
        self.dx = L / Nx
        self.dt = T / Nt
        self.u = np.zeros((Nx + 1, Nt + 1))
    
    def initial_conditions(self, func):
        self.u[:, 0] = func(np.linspace(0, self.L, self.Nx + 1))
    
    def boundary_conditions(self, left, right):
        self.u[0, :] = left
        self.u[-1, :] = right
    
    def solve(self):
        for n in range(0, self.Nt):
            for i in range(1, self.Nx):
                self.u[i, n+1] = self.u[i, n] + self.alpha * self.dt / self.dx**2 * (
                    self.u[i+1, n] - 2*self.u[i, n] + self.u[i-1, n])
        return self.u

# Finite Element Method (FEM) for PDEs
class FEM_PDE:
    def __init__(self, L, T, Nx, Nt, alpha):
        self.L = L      # Length of the spatial domain
        self.T = T      # Total time
        self.Nx = Nx    # Number of spatial elements
        self.Nt = Nt    # Number of time steps
        self.alpha = alpha  # Diffusion coefficient
        self.dx = L / Nx
        self.dt = T / Nt
        self.u = np.zeros((Nx + 1, Nt + 1))
        self.K = None
        self.M = None
    
    def initial_conditions(self, func):
        self.u[:, 0] = func(np.linspace(0, self.L, self.Nx + 1))
    
    def boundary_conditions(self, left, right):
        self.u[0, :] = left
        self.u[-1, :] = right
    
    def assemble_matrices(self):
        main_diag = 2 / self.dx * np.ones(self.Nx - 1)
        off_diag = -1 / self.dx * np.ones(self.Nx - 2)
        self.K = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()
        
        main_diag = 4 / 6 * self.dx * np.ones(self.Nx - 1)
        off_diag = 1 / 6 * self.dx * np.ones(self.Nx - 2)
        self.M = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()
    
    def solve(self):
        self.assemble_matrices()
        for n in range(0, self.Nt):
            F = self.u[1:-1, n]
            self.u[1:-1, n+1] = spsolve(self.M + self.alpha * self.dt * self.K, self.M @ F)
        return self.u

# Stochastic Partial Differential Equations (SPDEs)
class SPDE:
    def __init__(self, L, T, Nx, Nt, alpha, sigma):
        self.L = L      # Length of the spatial domain
        self.T = T      # Total time
        self.Nx = Nx    # Number of spatial steps
        self.Nt = Nt    # Number of time steps
        self.alpha = alpha  # Diffusion coefficient
        self.sigma = sigma  # Noise intensity
        self.dx = L / Nx
        self.dt = T / Nt
        self.u = np.zeros((Nx + 1, Nt + 1))
    
    def initial_conditions(self, func):
        self.u[:, 0] = func(np.linspace(0, self.L, self.Nx + 1))
    
    def boundary_conditions(self, left, right):
        self.u[0, :] = left
        self.u[-1, :] = right
    
    def solve(self):
        for n in range(0, self.Nt):
            for i in range(1, self.Nx):
                noise = self.sigma * np.sqrt(self.dt) * np.random.normal()
                self.u[i, n+1] = self.u[i, n] + self.alpha * self.dt / self.dx**2 * (
                    self.u[i+1, n] - 2*self.u[i, n] + self.u[i-1, n]) + noise
        return self.u

# Functions to process and visualize the results
def process_pde_solution(solution):
    return solution

def plot_pde_solution(solution, L, T):
    Nx, Nt = solution.shape
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, T, Nt)
    X, T = np.meshgrid(t, x)
    
    plt.figure()
    plt.contourf(T, X, solution, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('PDE Solution')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters for the PDE
    L = 1.0
    T = 0.2
    Nx = 50
    Nt = 500
    alpha = 0.01
    sigma = 0.005
    
    # Initial condition function
    def initial_condition(x):
        return np.sin(np.pi * x)
    
    # Create FDM solver
    fdm_solver = FDM_PDE(L, T, Nx, Nt, alpha)
    fdm_solver.initial_conditions(initial_condition)
    fdm_solver.boundary_conditions(0, 0)
    fdm_solution = fdm_solver.solve()
    
    # Create FEM solver
    fem_solver = FEM_PDE(L, T, Nx, Nt, alpha)
    fem_solver.initial_conditions(initial_condition)
    fem_solver.boundary_conditions(0, 0)
    fem_solution = fem_solver.solve()
    
    # Create SPDE solver
    spde_solver = SPDE(L, T, Nx, Nt, alpha, sigma)
    spde_solver.initial_conditions(initial_condition)
    spde_solver.boundary_conditions(0, 0)
    spde_solution = spde_solver.solve()
    
    # Process and plot the results
    processed_fdm_solution = process_pde_solution(fdm_solution)
    processed_fem_solution = process_pde_solution(fem_solution)
    processed_spde_solution = process_pde_solution(spde_solution)
    
    plot_pde_solution(processed_fdm_solution, L, T)
    plot_pde_solution(processed_fem_solution, L, T)
    plot_pde_solution(processed_spde_solution, L, T)


class SpectralMethods:
    def __init__(self, L, N, alpha):
        """
        Initialize the spectral method solver.
        
        Parameters:
        L (float): Domain length.
        N (int): Number of spatial points.
        alpha (float): Diffusion coefficient.
        """
        self.L = L
        self.N = N
        self.alpha = alpha
        self.x = np.linspace(0, L, N, endpoint=False)
        self.k = np.fft.fftfreq(N, d=L/N) * 2 * np.pi  # Wavenumbers

    def initialize_condition(self, initial_func):
        """
        Set the initial condition.
        
        Parameters:
        initial_func (callable): Function to generate initial condition.
        """
        self.u0 = initial_func(self.x)
        self.u_hat = fft(self.u0)

    def solve_heat_equation(self, T, dt):
        """
        Solve the heat equation using the Fourier spectral method.
        
        Parameters:
        T (float): Total time.
        dt (float): Time step.
        
        Returns:
        times (ndarray): Array of time points.
        u_solution (ndarray): Array of solution values at each time point.
        """
        t = 0
        u_hat_solution = [self.u_hat]
        while t < T:
            self.u_hat = self.u_hat * np.exp(-self.alpha * self.k**2 * dt)
            u_hat_solution.append(self.u_hat)
            t += dt

        u_solution = [np.real(ifft(u_hat)) for u_hat in u_hat_solution]
        times = np.arange(0, T + dt, dt)
        return times, np.array(u_solution)

    def plot_solution(self, times, u_solution, num_plots=10):
        """
        Plot the solution of the PDE at different time points.
        
        Parameters:
        times (ndarray): Array of time points.
        u_solution (ndarray): Array of solution values at each time point.
        num_plots (int): Number of time points to plot.
        """
        plt.figure(figsize=(8, 6))
        plot_indices = np.linspace(0, len(times) - 1, num_plots, dtype=int)
        for i in plot_indices:
            plt.plot(self.x, u_solution[i], label=f'Time = {times[i]:.2f}')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.title('Heat Equation Solution Using Fourier Spectral Method')
        plt.show()
"""
# Example usage of the SpectralMethods library
if __name__ == "__main__":
    L = 2 * np.pi  # Domain length
    N = 256  # Number of spatial points
    alpha = 0.02  # Diffusion coefficient
    T = 1.0  # Total time
    dt = 0.001  # Time step

    # Define initial condition function (Gaussian)
    def initial_condition(x):
        return np.exp(-100 * (x - L/2)**2)

    # Create SpectralMethods instance
    spectral_solver = SpectralMethods(L, N, alpha)
    spectral_solver.initialize_condition(initial_condition)

    # Solve the heat equation
    times, u_solution = spectral_solver.solve_heat_equation(T, dt)

    # Plot the solution
    spectral_solver.plot_solution(times, u_solution)
"""