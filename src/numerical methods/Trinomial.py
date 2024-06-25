# extension of Binomial.py

import numpy as np
from config import log_execution, measure_time

class TrinomialTree:
    def __init__(self, S0, K, T, r, sigma, N):
        self.S0 = S0  # Initial stock price
        self.K = K    # Strike price
        self.T = T    # Time to maturity
        self.r = r    # Risk-free rate
        self.sigma = sigma  # Volatility
        self.N = N    # Number of time steps
        self.dt = T / N  # Time step size

    @log_execution
    @measure_time
    def boyle_model(self):
        u = np.exp(self.sigma * np.sqrt(2 * self.dt))
        d = 1 / u
        m = 1
        pu = ((np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt / 2) - np.exp(-self.sigma * np.sqrt(self.dt / 2))) /
              (np.exp(self.sigma * np.sqrt(self.dt / 2)) - np.exp(-self.sigma * np.sqrt(self.dt / 2)))) ** 2
        pd = ((np.exp(self.sigma * np.sqrt(self.dt / 2)) - np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt / 2)) /
              (np.exp(self.sigma * np.sqrt(self.dt / 2)) - np.exp(-self.sigma * np.sqrt(self.dt / 2)))) ** 2
        pm = 1 - pu - pd
        return u, d, m, pu, pm, pd

    @log_execution
    @measure_time
    def kamrad_ritchken_model(self):
        u = np.exp(self.sigma * np.sqrt(3 * self.dt))
        d = 1 / u
        m = 1
        qu = 1 / 6 + (self.r - self.sigma ** 2 / 2) * np.sqrt(self.dt / (12 * self.sigma ** 2))
        qd = 1 / 6 - (self.r - self.sigma ** 2 / 2) * np.sqrt(self.dt / (12 * self.sigma ** 2))
        qm = 2 / 3
        return u, d, m, qu, qm, qd

    @log_execution
    @measure_time
    def price_option(self, model='boyle', option_type='european', payoff='call'):
        if model == 'boyle':
            u, d, m, pu, pm, pd = self.boyle_model()
        elif model == 'kamrad_ritchken':
            u, d, m, pu, pm, pd = self.kamrad_ritchken_model()
        else:
            raise ValueError("Unsupported model")

        # Initialize stock price tree
        stock_price = np.zeros((2 * self.N + 1, self.N + 1))
        stock_price[self.N, 0] = self.S0
        for t in range(1, self.N + 1):
            for i in range(-t, t + 1, 2):
                stock_price[self.N + i, t] = self.S0 * (u ** max(i, 0)) * (d ** max(-i, 0))

        # Initialize option value tree
        option_value = np.zeros((2 * self.N + 1, self.N + 1))
        if payoff == 'call':
            option_value[:, self.N] = np.maximum(0, stock_price[:, self.N] - self.K)
        elif payoff == 'put':
            option_value[:, self.N] = np.maximum(0, self.K - stock_price[:, self.N])
        else:
            raise ValueError("Unsupported payoff")

        # Backward phase
        for t in range(self.N - 1, -1, -1):
            for i in range(-t, t + 1, 2):
                option_value[self.N + i, t] = np.exp(-self.r * self.dt) * (
                    pu * option_value[self.N + i + 1, t + 1] +
                    pm * option_value[self.N + i, t + 1] +
                    pd * option_value[self.N + i - 1, t + 1]
                )
                if option_type == 'american':
                    if payoff == 'call':
                        option_value[self.N + i, t] = max(option_value[self.N + i, t], stock_price[self.N + i, t] - self.K)
                    elif payoff == 'put':
                        option_value[self.N + i, t] = max(option_value[self.N + i, t], self.K - stock_price[self.N + i, t])

        return option_value[self.N, 0]

"""# Example usage
if __name__ == "__main__":
    S0 = 100  # Initial stock price
    K = 100   # Strike price
    T = 1     # Time to maturity
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    N = 100   # Number of time steps

    trinomial_tree = TrinomialTree(S0, K, T, r, sigma, N)
    
    european_call_boyle = trinomial_tree.price_option(model='boyle', option_type='european', payoff='call')
    european_put_boyle = trinomial_tree.price_option(model='boyle', option_type='european', payoff='put')
    american_call_boyle = trinomial_tree.price_option(model='boyle', option_type='american', payoff='call')
    american_put_boyle = trinomial_tree.price_option(model='boyle', option_type='american', payoff='put')

    european_call_kr = trinomial_tree.price_option(model='kamrad_ritchken', option_type='european', payoff='call')
    european_put_kr = trinomial_tree.price_option(model='kamrad_ritchken', option_type='european', payoff='put')
    american_call_kr = trinomial_tree.price_option(model='kamrad_ritchken', option_type='american', payoff='call')
    american_put_kr = trinomial_tree.price_option(model='kamrad_ritchken', option_type='american', payoff='put')

    print("Boyle Model - European Call Option Price:", european_call_boyle)
    print("Boyle Model - European Put Option Price:", european_put_boyle)
    print("Boyle Model - American Call Option Price:", american_call_boyle)
    print("Boyle Model - American Put Option Price:", american_put_boyle)
    print("Kamrad-Ritchken Model - European Call Option Price:", european_call_kr)
    print("Kamrad-Ritchken Model - European Put Option Price:", european_put_kr)
    print("Kamrad-Ritchken Model - American Call Option Price:", american_call_kr)
    print("Kamrad-Ritchken Model - American Put Option Price:", american_put_kr)
"""