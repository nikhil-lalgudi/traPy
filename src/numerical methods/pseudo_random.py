import numpy as np
import matplotlib as plt

from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function

class PseudoRandomNumbers:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @log_execution
    @measure_time
    def uniform_random(self, low=0.0, high=1.0, size=1):
        """Generate uniform pseudo-random numbers."""
        return np.random.uniform(low, high, size)

    @log_execution
    @measure_time
    def normal_random(self, mean=0.0, stddev=1.0, size=1):
        """Generate normal pseudo-random numbers."""
        return np.random.normal(mean, stddev, size)

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
"""
# Example usage
if __name__ == "__main__":
    prng = PseudoRandomNumbers(seed=42)

    # Generate and plot uniform random numbers
    uniform_data = prng.uniform_random(low=0, high=10, size=10000)
    prng.plot_histogram(uniform_data, title="Uniform Random Numbers")

    # Generate and plot normal random numbers
    normal_data = prng.normal_random(mean=0, stddev=1, size=10000)
    prng.plot_histogram(normal_data, title="Normal Random Numbers")
    """