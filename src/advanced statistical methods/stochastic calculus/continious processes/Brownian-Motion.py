import numpy as np
import matplotlib.pyplot as plt
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function
from base import BaseTimeProcess


import numpy as np
from error_handler import check_numeric, check_positive_number
from base import BaseTimeProcess


class BrownianMotion(BaseTimeProcess):
    def __init__(self, drift=0, scale=1, t=1, rng=None):
        super().__init__(t=t, rng=rng)
        self.drift = drift
        self.scale = scale

    def __str__(self):
        if self.drift == 0 and self.scale == 1:
            return f"Standard Brownian motion on interval [0, {self.t}]"
        return f"Brownian motion with drift {self.drift} and scale {self.scale} on interval [0, {self.t}]"

    def __repr__(self):
        return f"BrownianMotion(drift={self.drift}, scale={self.scale}, t={self.t})"

    @property
    def drift(self):
        """Drift parameter."""
        return self._drift

    @drift.setter
    def drift(self, value):
        check_numeric(value, "Drift")
        self._drift = value

    @property
    def scale(self):
        """Scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, value):
        check_positive_number(value, "Scale")
        self._scale = value

    def _sample_brownian_motion(self, n):
        """Generate a realization of Brownian Motion."""
        dt = self.t / n
        increments = self.scale * np.sqrt(dt) * self.rng.normal(size=n)
        bm = np.cumsum(increments)
        bm = np.insert(bm, 0, 0)

        if self.drift != 0:
            drift_term = self.drift * np.linspace(0, self.t, n + 1)
            bm += drift_term

        return bm

    def sample(self, n):
        """Generate a realization.

        :param int n: the number of increments to generate
        """
        return self._sample_brownian_motion(n)

    def _sample_brownian_motion_at(self, times):
        """Generate a Brownian motion at specified times."""
        increments = np.diff(times)
        bm_increments = self.scale * np.sqrt(increments) * self.rng.normal(size=len(increments))
        bm = np.cumsum(bm_increments)
        bm = np.insert(bm, 0, 0)

        if self.drift != 0:
            drift_term = self.drift * times
            bm += drift_term

        return bm

    def sample_at(self, times):
        """Generate a realization using specified times.

        :param times: a vector of increasing time values at which to generate
            the realization
        """
        if times[0] != 0:
            times = np.insert(times, 0, 0)
        return self._sample_brownian_motion_at(times)
