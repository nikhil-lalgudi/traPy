import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from config import log_execution, measure_time
%matplotlib inline

class QuasiMonteCarloMethods:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @log_execution
    @measure_time
    def sobol_sequence(self, d, n, scramble=True):
        """Generate a Sobol sequence of `n` points in `d` dimensions."""
        sampler = Sobol(d, scramble=scramble)
        sample = sampler.random(n)
        return sample

    @log_execution
    @measure_time
    def halton_sequence(self, d, n, scramble=True):
        """Generate a Halton sequence of `n` points in `d` dimensions."""
        sampler = Halton(d, scramble=scramble)
        sample = sampler.random(n)
        return sample

    @log_execution
    @measure_time
    def latin_hypercube(self, d, n, scramble=True):
        """Generate a Latin hypercube sample of `n` points in `d` dimensions."""
        sampler = LatinHypercube(d, scramble=scramble)
        sample = sampler.random(n)
        return sample

    @log_execution
    @measure_time
    def poisson_disk(self, d, radius, hypersphere=False):
        """Generate a Poisson disk sample in `d` dimensions."""
        sampler = PoissonDisk(d, radius, hypersphere)
        sample = sampler.sample()
        return sample

    @log_execution
    @measure_time
    def multinomial_qmc(self, pvals, n_trials, engine='sobol', **kwargs):
        """QMC sampling from a multinomial distribution."""
        sampler = MultinomialQMC(pvals, n_trials, engine, **kwargs)
        sample = sampler.sample()
        return sample

    @log_execution
    @measure_time
    def multivariate_normal_qmc(self, mean, cov, engine='sobol', **kwargs):
        """QMC sampling from a multivariate Normal distribution."""
        sampler = MultivariateNormalQMC(mean, cov, engine, **kwargs)
        sample = sampler.sample()
        return sample

    @log_execution
    @measure_time
    def discrepancy(self, sample, iterative=False, method='centered'):
        """Calculate the discrepancy of a given sample."""
        if method == 'centered':
            return self.centered_discrepancy(sample, iterative)
        elif method == 'geometric':
            return self.geometric_discrepancy(sample)
        else:
            raise ValueError("Unsupported method")

    @log_execution
    @measure_time
    def centered_discrepancy(self, sample, iterative=False):
        """Calculate the centered discrepancy of a given sample."""
        n, d = sample.shape
        if iterative:
            return self._centered_discrepancy_iterative(sample)
        else:
            return self._centered_discrepancy(sample, n, d)

    def _centered_discrepancy(self, sample, n, d):
        D_star = 0
        for i in range(n):
            for j in range(n):
                D_star += np.prod(1 - np.maximum(sample[i], sample[j]))
        D_star = (13 / 12) ** d - (2 / n) * D_star + (1 / (n ** 2)) * np.sum(np.prod(1 - sample, axis=1))
        return D_star

    def _centered_discrepancy_iterative(self, sample):
        n, d = sample.shape
        D_star = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    D_star += np.prod(1 - np.maximum(sample[i], sample[j]))
        D_star = (13 / 12) ** d - (2 / n) * D_star + (1 / (n ** 2)) * np.sum(np.prod(1 - sample, axis=1))
        return D_star

    @log_execution
    @measure_time
    def geometric_discrepancy(self, sample, method='L2'):
        """Calculate the geometric discrepancy of a given sample."""
        if method == 'L2':
            return self._L2_discrepancy(sample)
        else:
            raise ValueError("Unsupported method")

    def _L2_discrepancy(self, sample):
        n, d = sample.shape
        L2_star = 0
        for i in range(n):
            for j in range(n):
                prod = 1
                for k in range(d):
                    prod *= (1 - abs(sample[i, k] - sample[j, k]))
                L2_star += prod
        L2_star = (4 / 3) ** d - (2 / n) * L2_star + (1 / (n ** 2)) * np.sum(np.prod(1 - sample, axis=1))
        return L2_star

    @log_execution
    @measure_time
    def update_discrepancy(self, x_new, sample, initial_disc):
        """Update the centered discrepancy with a new sample."""
        n, d = sample.shape
        D_star_new = initial_disc * (n ** 2) + 2 * np.sum(np.prod(1 - np.maximum(x_new, sample), axis=1)) + np.prod(1 - x_new)
        D_star_new /= (n + 1) ** 2
        return D_star_new

    @log_execution
    @measure_time
    def scale(self, sample, l_bounds, u_bounds, reverse=False):
        """Scale the sample from unit hypercube to different bounds."""
        if reverse:
            return (sample - l_bounds) / (u_bounds - l_bounds)
        else:
            return l_bounds + sample * (u_bounds - l_bounds)

    @log_execution
    @measure_time
    def simulate_asset_paths(self, S0, T, mu, sigma, n_sims, n_steps, method='sobol'):
        """Simulate asset paths using quasi-random sequences."""
        dt = T / n_steps
        asset_paths = np.zeros((n_sims, n_steps + 1))
        asset_paths[:, 0] = S0
        if method == 'sobol':
            sample = self.sobol_sequence(n_steps, n_sims)
        elif method == 'halton':
            sample = self.halton_sequence(n_steps, n_sims)
        elif method == 'latin_hypercube':
            sample = self.latin_hypercube(n_steps, n_sims)
        else:
            raise ValueError("Unsupported method. Use 'sobol', 'halton', or 'latin_hypercube'.")
        
        Z = st.norm.ppf(sample)
        for t in range(1, n_steps + 1):
            asset_paths[:, t] = asset_paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
        return asset_paths

    @log_execution
    @measure_time
    def price_european_option(self, S0, K, T, r, sigma, option_type='call', n_sims=10000, n_steps=100, method='sobol'):
        """Price a European option using quasi-Monte Carlo simulation."""
        asset_paths = self.simulate_asset_paths(S0, T, r, sigma, n_sims, n_steps, method)
        if option_type == 'call':
            payoffs = np.maximum(asset_paths[:, -1] - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - asset_paths[:, -1], 0)
        else:
            raise ValueError("Unsupported option type")
        discounted_payoffs = np.exp(-r * T) * payoffs
        return np.mean(discounted_payoffs)

    @log_execution
    @measure_time
    def plot_asset_paths(self, S0, T, mu, sigma, n_sims, n_steps, method='sobol'):
        """Plot simulated asset paths."""
        asset_paths = self.simulate_asset_paths(S0, T, mu, sigma, n_sims, n_steps, method)
        plt.figure(figsize=(10, 6))
        for i in range(n_sims):
            plt.plot(asset_paths[i, :], lw=0.5)
        plt.title(f"Simulated Asset Paths using {method.capitalize()} Sequences")
        plt.xlabel("Time Steps")
        plt.ylabel("Asset Price")
        plt.grid(True)
        plt.show()

    @log_execution
    @measure_time
    def plot_option_prices(self, S0, K, T, r, sigma, n_sims, n_steps, method='sobol'):
        """Plot option prices using quasi-Monte Carlo simulation."""
        strikes = np.linspace(S0 * 0.5, S0 * 1.5, 50)
        prices = [self.price_european_option(S0, K, T, r, sigma, 'call', n_sims, n_steps, method) for K in strikes]
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, prices, label='Call Option Prices')
        plt.title(f"Option Prices using {method.capitalize()} Sequences")
        plt.xlabel("Strike Price")
        plt.ylabel("Option Price")
        plt.legend()
        plt.grid(True)
        plt.show()
"""
# Example usage
if __name__ == "__main__":
    qmcm = QuasiMonteCarloMethods(seed=42)
    S0 = 100  # Initial stock price
    K = 105   # Strike price
    T = 1     # Time to maturity in years
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    call_price_sobol = qmcm.price_european_option(S0, K, T, r, sigma, option_type='call', method='sobol')
    put_price_sobol = qmcm.price_european_option(S0, K, T, r, sigma, option_type='put', method='sobol')

    call_price_halton = qmcm.price_european_option(S0, K, T, r, sigma, option_type='call', method='halton')
    put_price_halton = qmcm.price_european_option(S0, K, T, r, sigma, option_type='put', method='halton')

    print(f"European Call Option Price (Sobol): {call_price_sobol}")
    print(f"European Put Option Price (Sobol): {put_price_sobol}")
    print(f"European Call Option Price (Halton): {call_price_halton}")
    print(f"European Put Option Price (Halton): {put_price_halton}")

    # Plot Asset Paths
    qmcm.plot_asset_paths(S0, T, r, sigma, n_sims=10, n_steps=100, method='sobol')
    qmcm.plot_asset_paths(S0, T, r, sigma, n_sims=10, n_steps=100, method='halton')

    # Plot Option Prices
    qmcm.plot_option_prices(S0, K, T, r, sigma, n_sims=1000, n_steps=100, method='sobol')
    qmcm.plot_option_prices(S0, K, T, r, sigma, n_sims=1000, n_steps=100, method='halton')
    """