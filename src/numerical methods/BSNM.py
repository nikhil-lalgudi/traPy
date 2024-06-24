import numpy as np
import scipy.stats as st
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


from diffusion import Diffusion ## debug NEEDED
from config import log_execution, measure_time


import matplotlib.pyplot as plt

class BlackScholesNumericalMethods:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @log_execution
    @measure_time
    def bs_pricer(self, S0, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S0 * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
        elif option_type == 'put':
            price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S0 * st.norm.cdf(-d1)
        else:
            raise ValueError("Unsupported option type")
        return price

    @log_execution
    @measure_time
    def put_call_parity(self, call_price, S0, K, T, r):
        put_price = call_price - S0 + K * np.exp(-r * T)
        return put_price

    @log_execution
    @measure_time
    def binomial_option_pricing(self, S0, K, T, r, sigma, option_type='call', n=1000):
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        # Initialize asset prices at maturity
        prices = np.zeros(n + 1)
        prices[0] = S0 * (d ** n)
        for i in range(1, n + 1):
            prices[i] = prices[i - 1] * u / d
        
        # Initialize option values at maturity
        values = np.zeros(n + 1)
        if option_type == 'call':
            values = np.maximum(0, prices - K)
        elif option_type == 'put':
            values = np.maximum(0, K - prices)
        
        # Backward induction to price the option
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                values[j] = discount * (p * values[j + 1] + (1 - p) * values[j])
        
        return values[0]

    @log_execution
    @measure_time
    def bs_pde_solver(self, S0, K, T, r, sigma, option_type='call', n=100, m=100):
        dt = T / n
        ds = 2 * S0 / m
        S = np.linspace(0, 2 * S0, m + 1)
        
        A = np.zeros((m + 1, m + 1))
        B = np.zeros(m + 1)
        
        for i in range(1, m):
            A[i, i-1] = 0.5 * dt * (sigma ** 2 * i ** 2 - r * i)
            A[i, i] = 1 - dt * (sigma ** 2 * i ** 2 + r)
            A[i, i+1] = 0.5 * dt * (sigma ** 2 * i ** 2 + r * i)
        
        if option_type == 'call':
            B = np.maximum(S - K, 0)
        elif option_type == 'put':
            B = np.maximum(K - S, 0)
        
        for j in range(n):
            B = np.linalg.solve(A, B)
            B[0] = 0
            B[-1] = 2 * S0 - K * np.exp(-r * (T - j * dt))
        
        return np.interp(S0, S, B)

    @log_execution
    @measure_time
    def derivative_approximation(self, f, x, h=1e-5):
        df_dx = (f(x + h) - f(x - h)) / (2 * h)
        return df_dx

    @log_execution
    @measure_time
    def implicit_discretization(self, S0, K, T, r, sigma, option_type='call', n=100, m=100):
        dt = T / n
        ds = 2 * S0 / m
        S = np.linspace(0, 2 * S0, m + 1)
        
        A = np.zeros((m + 1, m + 1))
        B = np.zeros(m + 1)
        
        for i in range(1, m):
            A[i, i-1] = -0.5 * dt * (sigma ** 2 * i ** 2 - r * i)
            A[i, i] = 1 + dt * (sigma ** 2 * i ** 2 + r)
            A[i, i+1] = -0.5 * dt * (sigma ** 2 * i ** 2 + r * i)
        
        if option_type == 'call':
            B = np.maximum(S - K, 0)
        elif option_type == 'put':
            B = np.maximum(K - S, 0)
        
        for j in range(n):
            B = splinalg.spsolve(sp.csr_matrix(A), B)
            B[0] = 0
            B[-1] = 2 * S0 - K * np.exp(-r * (T - j * dt))
        
        return np.interp(S0, S, B)

@staticmethod
def plot_asset_paths(asset_paths, title="Simulated Asset Paths"):
    plt.figure(figsize=(10, 6))
    for i in range(min(10, len(asset_paths))):  # Plot only a subset if there are many paths
        plt.plot(asset_paths[i], lw=0.5)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.grid(True)
    plt.show()

@staticmethod
def plot_option_price_convergence(prices, title="Option Price Convergence"):
    plt.figure(figsize=(10, 6))
    plt.plot(prices, lw=1)
    plt.title(title)
    plt.xlabel("Simulation Runs")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

@staticmethod
def plot_histogram(data, bins=50, title="Histogram"):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

@staticmethod
def plot_var_distribution(portfolio_returns, var, title="VaR Distribution"):
    plt.figure(figsize=(10, 6))
    plt.hist(portfolio_returns, bins=50, edgecolor='k', alpha=0.7)
    plt.axvline(var, color='r', linestyle='--', label=f'VaR at {var:.2f}')
    plt.title(title)
    plt.xlabel("Portfolio Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

"""
# Example usage
if __name__ == "__main__":
    bsnm = BlackScholesNumericalMethods(seed=42)
    S0 = 100  # Initial stock price
    K = 105   # Strike price
    T = 1     # Time to maturity in years
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility

    call_price = bsnm.bs_pricer(S0, K, T, r, sigma, option_type='call')
    put_price = bsnm.put_call_parity(call_price, S0, K, T, r)

    print(f"Black-Scholes Call Option Price: {call_price}")
    print(f"Put Price using Put-Call Parity: {put_price}")

    binom_call_price = bsnm.binomial_option_pricing(S0, K, T, r, sigma, option_type='call')
    binom_put_price = bsnm.binomial_option_pricing(S0, K, T, r, sigma, option_type='put')

    print(f"Binomial Call Option Price: {binom_call_price}")
    print(f"Binomial Put Option Price: {binom_put_price}")

    pde_call_price = bsnm.bs_pde_solver(S0, K, T, r, sigma, option_type='call')
    print(f"Black-Scholes PDE Call Option Price: {pde_call_price}")

    # Derivative approximation example
    f = lambda x: np.sin(x)
    df_dx = bsnm.derivative_approximation(f, np.pi/4)
    print(f"Derivative of sin(x) at pi/4: {df_dx}")

    implicit_call_price = bsnm.implicit_discretization(S0, K, T, r, sigma, option_type='call')
    print(f"Implicit Discretization Call Option Price: {implicit_call_price}")
"""