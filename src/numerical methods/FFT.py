# Add more inverse stuff

import numpy as np
import scipy.fft as fft
import scipy.integrate as integrate
import scipy.stats as st
import matplotlib.pyplot as plt
from black_scholes import BlackScholes

from config import log_execution, measure_time

%matplotlib inline

class FourierTransformMethods:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    @log_execution
    @measure_time
    def fourier_transform(self, f, x):
        """Compute the Fourier transform of a given function f at point x."""
        integrand = lambda t: f(t) * np.exp(-1j * x * t)
        return integrate.quad(integrand, -np.inf, np.inf)[0]

    @log_execution
    @measure_time
    def inverse_fourier_transform(self, F, x):
        """Compute the inverse Fourier transform of a given function F at point x."""
        integrand = lambda t: F(t) * np.exp(1j * x * t)
        return (1 / (2 * np.pi)) * integrate.quad(integrand, -np.inf, np.inf)[0]

    @log_execution
    @measure_time
    def numerical_inversion(self, characteristic_function, u, T, K, r):
        """Numerical inversion of a characteristic function for option pricing."""
        integrand = lambda phi: (
            np.exp(-1j * phi * np.log(K))
            * characteristic_function(phi - 1j * (u + 1))
            / (1j * phi * u * (u + 1))
        ).real
        integral = integrate.quad(integrand, -np.inf, np.inf)[0]
        return np.exp(-r * T) * (K ** u) * integral / np.pi

    @log_execution
    @measure_time
    def lewis_method(self, characteristic_function, S0, K, T, r, sigma, option_type='call'):
        """Option pricing using Lewis method based on characteristic function."""
        integrand = lambda phi: (
            np.exp(-1j * phi * np.log(K)) 
            * characteristic_function(phi - 1j * 0.5) 
            / (phi ** 2 + 0.25)
        ).real
        integral = integrate.quad(integrand, -np.inf, np.inf)[0]
        price = (S0 * np.exp(-0.5 * r * T) - K * np.exp(-r * T)) * 0.5 + np.exp(-r * T) * integral / np.pi
        if option_type == 'put':
            price = price + K * np.exp(-r * T) - S0 * np.exp(-r * T)
        return price

    @log_execution
    @measure_time
    def fast_fourier_transform(self, data):
        """Compute the Fast Fourier Transform of the given data."""
        return fft.fft(data)

    @log_execution
    @measure_time
    def inverse_fast_fourier_transform(self, data):
        """Compute the Inverse Fast Fourier Transform of the given data."""
        return fft.ifft(data)

    @log_execution
    @measure_time
    def option_pricing_fft(self, characteristic_function, S0, K, T, r, alpha=1.5, N=4096, B=800):
        """Option pricing using FFT based on Carr-Madan method."""
        eta = B / N
        v = np.arange(N) * eta
        lambd = 2 * np.pi / (N * eta)
        k = -N * lambd / 2 + np.arange(N) * lambd

        psi = lambda v: np.exp(-r * T) * characteristic_function(v - (alpha + 1) * 1j) / (alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)
        z = np.exp(-1j * k[0] * v) * psi(v) * eta
        z[0] *= 0.5

        fft_z = self.fast_fourier_transform(z)
        C = np.exp(-alpha * k) / np.pi * fft_z.real

        K_values = np.exp(k) * S0
        return np.interp(K, K_values, C)

    @log_execution
    @measure_time
    def plot_option_price(self, S0, K, T, r, sigma, alpha=1.5, N=4096, B=800):
        """Plot the option prices using the FFT method."""
        characteristic_function = lambda u: BlackScholes.characteristic_function(u, S0, K, T, r, sigma)
        prices = self.option_pricing_fft(characteristic_function, S0, K, T, r, alpha, N, B)

        K_values = np.linspace(S0 * 0.5, S0 * 1.5, len(prices))
        plt.figure(figsize=(10, 6))
        plt.plot(K_values, prices, label='Option Prices')
        plt.xlabel('Strike Price')
        plt.ylabel('Option Price')
        plt.title('Option Prices using FFT Method')
        plt.legend()
        plt.grid(True)
        plt.show()

"""
# Example usage
if __name__ == "__main__":
    ftm = FourierTransformMethods(seed=42)
    
    # Define a characteristic function for the Black-Scholes model
    def bs_characteristic_function(u, S0=100, K=105, T=1, r=0.05, sigma=0.2):
        i = 1j
        phi = np.exp(i * u * (np.log(S0) + (r - 0.5 * sigma ** 2) * T) - 0.5 * sigma ** 2 * u ** 2 * T)
        return phi
    
    call_price = ftm.lewis_method(bs_characteristic_function, 100, 105, 1, 0.05, 0.2, option_type='call')
    print(f"Option Price using Lewis Method: {call_price}")

    # FFT Option Pricing
    fft_price = ftm.option_pricing_fft(bs_characteristic_function, 100, 105, 1, 0.05)
    print(f"Option Price using FFT: {fft_price}")

    # Plot Option Prices
    ftm.plot_option_price(100, 105, 1, 0.05, 0.2)

    # Fast Fourier Transform Example
    data = np.random.rand(1024)
    fft_data = ftm.fast_fourier_transform(data)
    ifft_data = ftm.inverse_fast_fourier_transform(fft_data)

    print(f"FFT of Data: {fft_data[:5]}")
    print(f"Inverse FFT of Data: {ifft_data[:5]}")
"""