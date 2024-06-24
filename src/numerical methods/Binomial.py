import numpy as np
import matplotlib.pyplot as plt

from config import log_execution, measure_time

# Assumptions

"""
(A1) Arbitrage is impossible (no-arbitrage principle)
(A2) There is a risk-free interest rate r > 0 which applies for all credits. Continuous
payment of interest according to (1.1).
(A3) No transaction costs, taxes, etc. Trading is possible at any time. Any fraction of
an asset can be sold. Liquid market, i.e. selling an asset does not change its value
significantly.
(A4) A seller can sell assets he/she does not own yet (“short selling”)
(A5) No dividends on the underlying asset are paid.

"""
class BinomialModel:
    def __init__(self, S0, K, T, r, sigma, N):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
        self.N = N    # Number of time steps
        self.dt = T / N  # Time step size
        self.u, self.d, self.p = self.compute_parameters()
        self.tree = self.build_tree()

    @log_execution
    @measure_time
    def compute_parameters(self):
        """Compute the parameters u, d, and p for the binomial model."""
        beta = 0.5 * (np.exp(-self.r * self.dt) + np.exp((self.r + self.sigma ** 2) * self.dt))
        u = beta + np.sqrt(beta ** 2 - 1)
        d = 1 / u
        p = (np.exp(self.r * self.dt) - d) / (u - d)
        return u, d, p

    @log_execution
    @measure_time
    def build_tree(self):
        """Build the binomial tree for stock prices."""
        tree = np.zeros((self.N + 1, self.N + 1))
        for n in range(self.N + 1):
            for j in range(n + 1):
                tree[j, n] = self.S0 * (self.u ** j) * (self.d ** (n - j))
        return tree

    @log_execution
    @measure_time
    def price_option(self, option_type='european', payoff='call'):
        """Price an option using the binomial model."""
        # Initialize the option values at maturity
        option_values = np.zeros((self.N + 1, self.N + 1))
        if payoff == 'call':
            option_values[:, self.N] = np.maximum(0, self.tree[:, self.N] - self.K)
        elif payoff == 'put':
            option_values[:, self.N] = np.maximum(0, self.K - self.tree[:, self.N])

        # Backward phase
        for n in range(self.N - 1, -1, -1):
            for j in range(n + 1):
                option_values[j, n] = np.exp(-self.r * self.dt) * (
                    self.p * option_values[j + 1, n + 1] + (1 - self.p) * option_values[j, n + 1])
                if option_type == 'american':
                    if payoff == 'call':
                        option_values[j, n] = max(option_values[j, n], self.tree[j, n] - self.K)
                    elif payoff == 'put':
                        option_values[j, n] = max(option_values[j, n], self.K - self.tree[j, n])
        
        return option_values[0, 0]
"""
# Example usage:
if __name__ == "__main__":
    S0 = 100  # Initial stock price
    K = 100   # Strike price
    T = 1     # Time to maturity
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    N = 100   # Number of time steps

    model = BinomialModel(S0, K, T, r, sigma, N)
    print(f"u: {model.u}, d: {model.d}, p: {model.p}")
    
    european_call_price = model.price_option(option_type='european', payoff='call')
    european_put_price = model.price_option(option_type='european', payoff='put')
    american_call_price = model.price_option(option_type='american', payoff='call')
    american_put_price = model.price_option(option_type='american', payoff='put')

    print(f"European Call Option Price: {european_call_price}")
    print(f"European Put Option Price: {european_put_price}")
    print(f"American Call Option Price: {american_call_price}")
    print(f"American Put Option Price: {american_put_price}")

"""


# Note: Cite Numerical methods in mathematical finance, Tobias Jahnke (Karlsruher Institut f¨ur Technologie)
# Version: April 5, 2013