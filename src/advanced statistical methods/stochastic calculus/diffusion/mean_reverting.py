import numpy as np
import matplotlib.pyplot as plt
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function
from base import BaseTimeProcess


from ornstein_uhlenbeck import OrnsteinUhlenbeckProcess

class MeanRevertingProcess(BaseTimeProcess):
    def __init__(self, speed=1, mean=0, vol=1, t=1, rng=None):
        super().__init__(t=t, rng=rng)
        self.speed = speed
        self.mean = mean
        self.vol = vol

    def __str__(self):
        return f"Mean-reverting process with speed={self.speed}, mean={self.mean}, vol={self.vol} on [0, {self.t}]"

    def __repr__(self):
        return f"MeanRevertingProcess(speed={self.speed}, mean={self.mean}, vol={self.vol}, t={self.t})"

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        check_numeric_or_single_arg_callable(value, "speed")
        self._speed = ensure_single_arg_constant_function(value)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        check_numeric_or_single_arg_callable(value, "mean")
        self._mean = ensure_single_arg_constant_function(value)

    @property
    def vol(self):
        return self._vol

    @vol.setter
    def vol(self, value):
        check_numeric_or_single_arg_callable(value, "vol")
        self._vol = ensure_single_arg_constant_function(value)

    def _sample(self, n, initial=1.0):
        check_positive_integer(n)
        check_numeric(initial, "Initial")
        delta_t = 1.0 * self.t / n
        gns = self.rng.normal(size=n)
        s = [initial]
        t = 0
        for k in range(n):
            t += delta_t
            initial += (
                self.speed(t) * (self.mean(t) - initial) * delta_t
                + self.vol(t) * np.sqrt(delta_t) * gns[k]
            )
            s.append(initial)
        return np.array(s)

    def sample(self, n, initial=1.0):
        return self._sample(n, initial)