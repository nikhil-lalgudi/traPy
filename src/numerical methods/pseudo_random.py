import numpy as np
import matplotlib.pyplot as plt

# Auxiliary function for Inversion method
from scipy.special import erfinv

from config import log_execution, measure_time

class PseudoRandomNumbers:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    # Uniform Pseudo-Random Number Generators
    @log_execution
    @measure_time
    def linear_congruential_generator(self, a, c, m, seed, size=1):
        """Generate uniform pseudo-random numbers using Linear Congruential Generator."""
        numbers = np.zeros(size)
        x = seed
        for i in range(size):
            x = (a * x + c) % m
            numbers[i] = x / m
        return numbers

    @log_execution
    @measure_time
    def fibonacci_generator(self, a, b, m, size=1):
        """Generate uniform pseudo-random numbers using Fibonacci Generator."""
        numbers = np.zeros(size)
        x = a
        y = b
        for i in range(size):
            x, y = y, (x + y) % m
            numbers[i] = x / m
        return numbers

    @log_execution
    @measure_time
    def combined_multiple_recursive_generator(self, a1, a2, m1, m2, seed1, seed2, size=1):
        """Generate uniform pseudo-random numbers using Combined Multiple Recursive Generator."""
        numbers = np.zeros(size)
        x = seed1
        y = seed2
        for i in range(size):
            x = (a1 * x) % m1
            y = (a2 * y) % m2
            z = (x - y) % m1
            if z < 0:
                z += m1
            numbers[i] = z / m1
        return numbers

    # Normal Pseudo-Random Number Generators
    @log_execution
    @measure_time
    def inversion_method(self, size=1):
        """Generate normal pseudo-random numbers using Inversion Method."""
        u = np.random.uniform(0, 1, size)
        return np.sqrt(2) * erfinv(2 * u - 1)

    @log_execution
    @measure_time
    def box_muller_method(self, size=1):
        """Generate normal pseudo-random numbers using Box-Muller Method."""
        u1 = np.random.uniform(0, 1, size)
        u2 = np.random.uniform(0, 1, size)
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
        return z1, z2

    @log_execution
    @measure_time
    def polar_method(self, size=1):
        """Generate normal pseudo-random numbers using Polar Method."""
        z = np.zeros(size)
        for i in range(size):
            while True:
                v1 = 2 * np.random.uniform() - 1
                v2 = 2 * np.random.uniform() - 1
                s = v1**2 + v2**2
                if s < 1 and s > 0:
                    z[i] = v1 * np.sqrt(-2 * np.log(s) / s)
                    break
        return z

    # Correlated Normal Random Vectors
    @log_execution
    @measure_time
    def correlated_normal_random_vectors(self, mean, cov, size=1):
        """Generate correlated normal random vectors."""
        return np.random.multivariate_normal(mean, cov, size)

    @log_execution
    @measure_time
    def plot_histogram(self, data, bins=50, title="Histogram"):
        """Plot histogram of the given data."""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

"""# Example usage
if __name__ == "__main__":
    prng = PseudoRandomNumbers(seed=42)

    # Generate and plot uniform random numbers using different methods
    lc_data = prng.linear_congruential_generator(a=1664525, c=1013904223, m=2**32, seed=42, size=10000)
    prng.plot_histogram(lc_data, title="Linear Congruential Generator")

    fib_data = prng.fibonacci_generator(a=1, b=1, m=2**32, size=10000)
    prng.plot_histogram(fib_data, title="Fibonacci Generator")

    cmrg_data = prng.combined_multiple_recursive_generator(a1=40014, a2=40692, m1=2147483563, m2=2147483399, seed1=42, seed2=43, size=10000)
    prng.plot_histogram(cmrg_data, title="Combined Multiple Recursive Generator")

    # Generate and plot normal random numbers using different methods
    inv_data = prng.inversion_method(size=10000)
    prng.plot_histogram(inv_data, title="Inversion Method")

    bm_data1, bm_data2 = prng.box_muller_method(size=5000)
    prng.plot_histogram(bm_data1, title="Box-Muller Method")

    polar_data = prng.polar_method(size=10000)
    prng.plot_histogram(polar_data, title="Polar Method")

    # Generate and plot correlated normal random vectors
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    correlated_data = prng.correlated_normal_random_vectors(mean, cov, size=10000)
    plt.figure(figsize=(10, 6))
    plt.scatter(correlated_data[:, 0], correlated_data[:, 1], alpha=0.5)
    plt.title("Correlated Normal Random Vectors")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
"""