import numpy as np
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function

from Gaussian import GaussianNoise  
from base import BaseTimeProcess

class DiffusionProcess(GaussianNoise):
    def __init__(self, speed=1, mean=0, vol=1, volexp=0, t=1, rng=None):
        super().__init__(t=t, rng=rng)
        self.speed = speed
        self.mean = mean
        self.vol = vol
        self.volexp = volexp

    def __str__(self):
        return f"Diffusion process with speed={self.speed}, mean={self.mean}, vol={self.vol}, volexp={self.volexp} on [0, {self.t}]"

    def __repr__(self):
        return f"Diffusion(speed={self.speed}, mean={self.mean}, vol={self.vol}, volexp={self.volexp}, t={self.t})"

    @property
    def speed(self):
        return self._speed

    @speed.setter
    @check_numeric_or_single_arg_callable
    def speed(self, value):
        self._speed = ensure_single_arg_constant_function(value)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    @check_numeric_or_single_arg_callable
    def mean(self, value):
        self._mean = ensure_single_arg_constant_function(value)

    @property
    def vol(self):
        return self._vol

    @vol.setter
    @check_numeric_or_single_arg_callable
    def vol(self, value):
        self._vol = ensure_single_arg_constant_function(value)

    @property
    def volexp(self):
        return self._volexp

    @volexp.setter
    @check_numeric_or_single_arg_callable
    def volexp(self, value):
        self._volexp = ensure_single_arg_constant_function(value)

    def _sample(self, n, initial=1.0):
        check_positive_integer(n)
        check_numeric(initial, "Initial")
        delta_t = 1.0 * self.t / n
        gns = self._sample_gaussian_noise(n)
        s = [initial]
        t = 0
        for k in range(n):
            t += delta_t
            initial += (
                self.speed(t) * (self.mean(t) - initial) * delta_t
                + self.vol(t) * initial ** self.volexp(initial) * gns[k]
            )
            s.append(initial)
        return np.array(s)

    def sample(self, n, initial=1.0):
        return self._sample(n, initial)
