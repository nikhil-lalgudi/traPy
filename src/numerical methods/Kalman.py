import numpy as np

from numpy.linalg import inv
import matplotlib.pyplot as plt

from config import log_execution, measure_time

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Observation noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    @log_execution
    @measure_time
    def predict(self):
        """Predict the next state and covariance."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    @log_execution
    @measure_time
    def update(self, z):
        """Update the state estimate with the new observation."""
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[1])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

class AR1Process:
    def __init__(self, phi, sigma, n):
        self.phi = phi  # Autoregressive parameter
        self.sigma = sigma  # Noise standard deviation
        self.n = n  # Number of observations
        self.data = self.generate_data()

    @log_execution
    @measure_time
    def generate_data(self):
        """Generate AR(1) process data."""
        data = np.zeros(self.n)
        for t in range(1, self.n):
            data[t] = self.phi * data[t - 1] + np.random.normal(0, self.sigma)
        return data

class RegressionAnalysis:
    @staticmethod
    @log_execution
    @measure_time
    def estimate_parameters(data):
        """Estimate AR(1) parameters using regression analysis."""
        x = data[:-1]
        y = data[1:]
        phi = np.dot(x, y) / np.dot(x, x)
        sigma = np.sqrt(np.mean((y - phi * x) ** 2))
        return phi, sigma

def run_kalman_filter(data, phi, sigma):
    F = np.array([[phi]])
    H = np.array([[1]])
    Q = np.array([[sigma ** 2]])
    R = np.array([[1]])
    x0 = np.array([0])
    P0 = np.array([[1]])
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    estimates = np.zeros(len(data))
    for t in range(len(data)):
        kf.predict()
        kf.update(data[t])
        estimates[t] = kf.x
    return estimates
"""
# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    true_phi = 0.8
    true_sigma = 1.0

    # Generate AR(1) process data
    ar1 = AR1Process(true_phi, true_sigma, n)
    data = ar1.data

    # Estimate parameters using regression analysis
    phi_est, sigma_est = RegressionAnalysis.estimate_parameters(data)
    logger.info(f"Estimated phi: {phi_est}, Estimated sigma: {sigma_est}")

    # Run Kalman filter
    kalman_estimates = run_kalman_filter(data, phi_est, sigma_est)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='AR(1) Process')
    plt.plot(kalman_estimates, label='Kalman Filter Estimate', linestyle='--')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Kalman Filter for AR(1) Process')
    plt.grid(True)
    plt.show()
"""    