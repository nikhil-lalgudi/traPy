import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import log_execution, measure_time
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_execution(func):
    """Decorator to log function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}...")
        result = func(*args, **kwargs)
        logger.info(f"Finished executing {func.__name__}.")
        return result
    return wrapper

def measure_time(func):
    """Decorator to measure the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class FiniteDifferenceMethods:
    def __init__(self):
        pass

    @log_execution
    @measure_time
    def setup_grid(self, S_max, T, M, N):
        """Set up the spatial and time grids."""
        self.S_max = S_max
        self.T = T
        self.M = M
        self.N = N
        self.dS = S_max / M
        self.dt = T / N
        self.S = np.linspace(0, S_max, M + 1)
        self.t = np.linspace(0, T, N + 1)
        self.grid = np.zeros((M + 1, N + 1))
        logger.info(f"Grid set up with S_max={S_max}, T={T}, M={M}, N={N}")

    @log_execution
    @measure_time
    def set_boundary_conditions(self, K, r):
        """Set boundary and initial conditions."""
        # Boundary condition at maturity
        self.grid[:, -1] = np.maximum(self.S - K, 0)  # For a call option
        # Boundary conditions at S = 0 and S = S_max
        self.grid[0, :] = 0  # For all t
        self.grid[-1, :] = self.S_max - K * np.exp(-r * (self.T - self.t))

    @log_execution
    @measure_time
    def solve_explicit(self, r, sigma):
        """Solve the PDE using the explicit finite difference method."""
        for j in range(self.N - 1, -1, -1):
            for i in range(1, self.M):
                self.grid[i, j] = (self.grid[i, j + 1] +
                                   0.5 * self.dt * (sigma**2 * i**2 - r * i) * self.grid[i - 1, j + 1] +
                                   (1 - self.dt * (sigma**2 * i**2 + r)) * self.grid[i, j + 1] +
                                   0.5 * self.dt * (sigma**2 * i**2 + r * i) * self.grid[i + 1, j + 1])

    @log_execution
    @measure_time
    def solve_implicit(self, r, sigma):
        """Solve the PDE using the implicit finite difference method."""
        A = self.setup_implicit_matrix(r, sigma)
        for j in range(self.N - 1, -1, -1):
            b = self.grid[1:self.M, j + 1]
            b[0] -= 0.5 * self.dt * (sigma**2 * 1**2 - r * 1) * self.grid[0, j + 1]
            b[-1] -= 0.5 * self.dt * (sigma**2 * (self.M - 1)**2 + r * (self.M - 1)) * self.grid[self.M, j + 1]
            self.grid[1:self.M, j] = spla.spsolve(A, b)

    def setup_implicit_matrix(self, r, sigma):
        """Set up the matrix for the implicit finite difference method."""
        M = self.M - 1
        diagonals = [
            0.5 * self.dt * (sigma**2 * np.arange(1, M + 1)**2 - r * np.arange(1, M + 1)),
            1 + self.dt * (sigma**2 * np.arange(1, M + 1)**2 + r),
            0.5 * self.dt * (sigma**2 * np.arange(1, M + 1)**2 + r * np.arange(1, M + 1))
        ]
        A = sp.diags(diagonals, offsets=[-1, 0, 1], shape=(M, M), format='csc')
        return A

    @log_execution
    @measure_time
    def interpolate_option_price(self, S0):
        """Interpolate the option price at the current stock price."""
        return np.interp(S0, self.S, self.grid[:, 0])

    @log_execution
    @measure_time
    def plot_results(self):
        """Plot the results of the finite difference method."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        S, t = np.meshgrid(self.S, self.t)
        ax.plot_surface(S, t, self.grid.T, cmap='viridis')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        plt.title('Finite Difference Method Results')
        plt.show()

# Example usage
if __name__ == "__main__":
    fdm = FiniteDifferenceMethods()
    S_max = 200
    T = 1
    M = 100
    N = 1000
    r = 0.05
    sigma = 0.2
    K = 100
    S0 = 100

    fdm.setup_grid(S_max, T, M, N)
    fdm.set_boundary_conditions(K, r)
    fdm.solve_explicit(r, sigma)
    option_price_explicit = fdm.interpolate_option_price(S0)
    print(f"Option price (explicit) at S0={S0}: {option_price_explicit}")
    fdm.plot_results()

    fdm.setup_grid(S_max, T, M, N)
    fdm.set_boundary_conditions(K, r)
    fdm.solve_implicit(r, sigma)
    option_price_implicit = fdm.interpolate_option_price(S0)
    print(f"Option price (implicit) at S0={S0}: {option_price_implicit}")
    fdm.plot_results()
