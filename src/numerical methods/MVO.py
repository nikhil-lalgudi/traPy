import numpy as np
import scipy.optimize as sco
import matplotlib as plt

from config import log_execution, measure_time

class MeanVarianceOptimization:
    def __init__(self, returns, cov_matrix, risk_free_rate=0.0):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns)
    
    @log_execution
    @measure_time
    def portfolio_return(self, weights):
        return np.dot(weights, self.returns)
    
    @log_execution
    @measure_time
    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    @log_execution
    @measure_time
    def negative_sharpe_ratio(self, weights):
        p_return = self.portfolio_return(weights)
        p_volatility = self.portfolio_volatility(weights)
        return -(p_return - self.risk_free_rate) / p_volatility

    @log_execution
    @measure_time
    def optimize_sharpe_ratio(self):
        num_assets = self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        result = sco.minimize(self.negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    @log_execution
    @measure_time
    def optimal_weights_stocks_bonds(self, stock_weights, bond_weights):
        weights = stock_weights + bond_weights
        return weights / np.sum(weights)

    @log_execution
    @measure_time
    def tangency_portfolio_density(self, weights):
        p_return = self.portfolio_return(weights)
        p_volatility = self.portfolio_volatility(weights)
        sharpe_ratio = (p_return - self.risk_free_rate) / p_volatility
        return sharpe_ratio / np.sqrt(2 * np.pi)

    @log_execution
    @measure_time
    def closed_form_short_positions(self):
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        ones = np.ones(self.num_assets)
        w = inv_cov_matrix @ self.returns / (ones.T @ inv_cov_matrix @ self.returns)
        return w
    
    def portfolio_metrics(weights, returns, cov_matrix):
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_return, portfolio_volatility

"""# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    num_assets = 4
    returns = np.random.rand(num_assets)
    cov_matrix = np.random.rand(num_assets, num_assets)
    cov_matrix = cov_matrix.T @ cov_matrix  # To ensure the covariance matrix is positive semi-definite
    risk_free_rate = 0.01

    mvo = MeanVarianceOptimization(returns, cov_matrix, risk_free_rate)
    
    # Optimize Sharpe ratio
    optimal_sharpe_result = mvo.optimize_sharpe_ratio()
    optimal_sharpe_weights = optimal_sharpe_result.x
    print("Optimal weights (Sharpe ratio):", optimal_sharpe_weights)

    # Optimal weights between stocks and bonds
    stock_weights = np.array([0.6, 0.3, 0, 0])
    bond_weights = np.array([0, 0, 0.7, 0.3])
    optimal_weights = mvo.optimal_weights_stocks_bonds(stock_weights, bond_weights)
    print("Optimal weights (stocks and bonds):", optimal_weights)

    # Probability density of the tangency portfolio
    tangency_density = mvo.tangency_portfolio_density(optimal_sharpe_weights)
    print("Tangency portfolio density:", tangency_density)

    # Short positions - closed formula
    closed_form_weights = mvo.closed_form_short_positions()
    print("Optimal weights (closed formula with short positions):", closed_form_weights)

    # Plotting the efficient frontier
   

    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        p_return, p_volatility = portfolio_metrics(weights, returns, cov_matrix)
        results[0, i] = p_return
        results[1, i] = p_volatility
        results[2, i] = (p_return - risk_free_rate) / p_volatility

    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Efficient Frontier')
    plt.scatter(mvo.portfolio_volatility(optimal_sharpe_weights), mvo.portfolio_return(optimal_sharpe_weights), marker='*', color='r', s=200, label='Maximum Sharpe ratio')
    plt.legend()
    plt.grid(True)
    plt.show()
"""