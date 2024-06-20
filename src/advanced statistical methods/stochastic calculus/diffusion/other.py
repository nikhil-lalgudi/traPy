import numpy as np
import matplotlib.pyplot as plt
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function
from base import BaseTimeProcess

class HestonModel(BaseProcess):
    def __init__(self, s0, v0, mu, kappa, theta, xi, rho, t, rng=None):
        super().__init__(rng=rng)
        self.s0 = s0
        self.v0 = v0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.t = t

    def sample(self, n):
        dt = self.t / n
        s = np.zeros(n + 1)
        v = np.zeros(n + 1)
        s[0] = self.s0
        v[0] = self.v0
        for i in range(n):
            z1 = self.rng.normal()
            z2 = self.rng.normal()
            dw1 = np.sqrt(dt) * z1
            dw2 = np.sqrt(dt) * (self.rho * z1 + np.sqrt(1 - self.rho**2) * z2)
            v[i+1] = v[i] + self.kappa * (self.theta - v[i]) * dt + self.xi * np.sqrt(v[i]) * dw2
            v[i+1] = max(v[i+1], 0)
            s[i+1] = s[i] * np.exp((self.mu - 0.5 * v[i]) * dt + np.sqrt(v[i]) * dw1)
        return s, v


# GARCH(1,1) Model
class GARCH(BaseProcess):
    def __init__(self, omega, alpha, beta, n, rng=None):
        super().__init__(rng=rng)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.n = n

    def sample(self):
        r = np.zeros(self.n)
        sigma2 = np.zeros(self.n)
        for t in range(1, self.n):
            sigma2[t] = self.omega + self.alpha * r[t-1]**2 + self.beta * sigma2[t-1]
            r[t] = self.rng.normal(0, np.sqrt(sigma2[t]))
        return r, sigma2


# Black-Scholes Model
class BlackScholesModel(BaseProcess):
    def __init__(self, s0, mu, sigma, t, rng=None):
        super().__init__(rng=rng)
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.t = t

    def sample(self, n):
        dt = self.t / n
        s = np.zeros(n + 1)
        s[0] = self.s0
        for i in range(1, n + 1):
            z = self.rng.normal()
            s[i] = s[i-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        return s


# Hull-White Model
class HullWhiteModel(BaseProcess):
    def __init__(self, r0, alpha, sigma, t, rng=None):
        super().__init__(rng=rng)
        self.r0 = r0
        self.alpha = alpha
        self.sigma = sigma
        self.t = t

    def sample(self, n):
        dt = self.t / n
        r = np.zeros(n + 1)
        r[0] = self.r0
        for i in range(1, n + 1):
            z = self.rng.normal()
            r[i] = r[i-1] + self.alpha * (self.r0 - r[i-1]) * dt + self.sigma * np.sqrt(dt) * z
        return r


# Merton Jump-Diffusion Model
class MertonJumpDiffusionModel(BaseProcess):
    def __init__(self, s0, mu, sigma, lam, m, v, t, rng=None):
        super().__init__(rng=rng)
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.m = m
        self.v = v
        self.t = t

    def sample(self, n):
        dt = self.t / n
        s = np.zeros(n + 1)
        s[0] = self.s0
        for i in range(1, n + 1):
            z = self.rng.normal()
            poi = self.rng.poisson(self.lam * dt)
            s[i] = s[i-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z +
                                   poi * (self.m + self.v * self.rng.normal()))
        return s

class KouJumpDiffusionModel(BaseProcess):
    def __init__(self, s0, mu, sigma, lam, p, eta1, eta2, t, rng=None):
        super().__init__(rng=rng)
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2
        self.t = t

    def sample(self, n):
        dt = self.t / n
        s = np.zeros(n + 1)
        s[0] = self.s0
        for i in range(1, n + 1):
            z = self.rng.normal()
            poi = self.rng.poisson(self.lam * dt)
            jumps = np.sum(np.where(self.rng.random(poi) < self.p,
                                    self.rng.exponential(scale=1/self.eta1, size=poi),
                                    -self.rng.exponential(scale=1/self.eta2, size=poi)))
            s[i] = s[i-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z + jumps)
        return s
