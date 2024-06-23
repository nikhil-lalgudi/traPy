import numpy as np
import scipy.stats as st
import time
import functools
import logging

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

class MonteCarloMethods:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @log_execution
    @measure_time
    def generate_random_numbers(self, n, distribution='normal', **kwargs):
        if distribution == 'normal':
            return np.random.normal(kwargs.get('mu', 0), kwargs.get('sigma', 1), n)
        elif distribution == 'uniform':
            return np.random.uniform(kwargs.get('low', 0), kwargs.get('high', 1), n)
        elif distribution == 'lognormal':
            return np.random.lognormal(kwargs.get('mean', 0), kwargs.get('sigma', 1), n)
        else:
            raise ValueError("Unsupported distribution type")

    @log_execution
    @measure_time
    def simulate_asset_paths(self, S0, T, mu, sigma, n_sims, n_steps):
        dt = T / n_steps
        asset_paths = np.zeros((n_sims, n_steps + 1))
        asset_paths[:, 0] = S0
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(n_sims)
            asset_paths[:, t] = asset_paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return asset_paths

    @log_execution
    @measure_time
    def price_european_option(self, S0, K, T, r, sigma, option_type='call', n_sims=10000, n_steps=100):
        asset_paths = self.simulate_asset_paths(S0, T, r, sigma, n_sims, n_steps)
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
    def estimate_var(self, portfolio_returns, alpha=0.05):
        var = np.percentile(portfolio_returns, alpha * 100)
        return var

    @log_execution
    @measure_time
    def estimate_cvar(self, portfolio_returns, alpha=0.05):
        var = self.estimate_var(portfolio_returns, alpha)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return cvar

    @log_execution
    @measure_time
    def black_scholes_price(self, S0, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S0 * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S0 * st.norm.cdf(-d1)
        else:
            raise ValueError("Unsupported option type")
        return price
"""
# Example usage
if __name__ == "__main__":
    mcm = MonteCarloMethods(seed=42)
    S0 = 100  # Initial stock price
    K = 105   # Strike price
    T = 1     # Time to maturity in years
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    call_price = mcm.price_european_option(S0, K, T, r, sigma, option_type='call')
    put_price = mcm.price_european_option(S0, K, T, r, sigma, option_type='put')

    print(f"European Call Option Price: {call_price}")
    print(f"European Put Option Price: {put_price}")

    # Generate random returns for VaR calculation
    portfolio_returns = mcm.generate_random_numbers(10000, 'normal', mu=0.01, sigma=0.02)
    var_95 = mcm.estimate_var(portfolio_returns, alpha=0.05)
    cvar_95 = mcm.estimate_cvar(portfolio_returns, alpha=0.05)

    print(f"Value at Risk (95%): {var_95}")
    print(f"Conditional Value at Risk (95%): {cvar_95}")
"""