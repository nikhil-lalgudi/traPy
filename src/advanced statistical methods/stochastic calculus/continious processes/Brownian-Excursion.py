import numpy as np
import matplotlib.pyplot as plt
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function
from base import BaseTimeProcess

from Brownian-Bridge import BrownianBridge

import numpy as np
from brownian_bridge import BrownianBridge
from error_handler import check_numeric, check_positive_integer

class BrownianExcursion(BrownianBridge):

    def __init__(self, t=1, rng=None):
        super().__init__(b=0, t=t, rng=rng)

    def __str__(self):
        return f"Brownian excursion on [0, {self.t}]"

    def __repr__(self):
        return f"BrownianExcursion(t={self.t})"

    def _sample_brownian_excursion(self, n):
        """Generate a Brownian excursion."""
        brownian_bridge = self._sample_brownian_bridge(n)
        idx_min = np.argmin(brownian_bridge)
        excursion = np.roll(brownian_bridge, -idx_min) - brownian_bridge[idx_min]
        return np.maximum(excursion, 0)

    def _sample_brownian_excursion_at(self, times):
        """Generate a Brownian excursion at specified times."""
        brownian_bridge = self._sample_brownian_bridge_at(times)
        idx_min = np.argmin(brownian_bridge)
        excursion = np.roll(brownian_bridge, -idx_min) - brownian_bridge[idx_min]
        return np.maximum(excursion, 0)

    def sample(self, n):
        """Generate a realization.

        :param int n: the number of increments to generate.
        """
        return self._sample_brownian_excursion(n)

    def sample_at(self, times):
        """Generate a realization using specified times.

        :param times: a vector of increasing time values at which to generate
            the realization
        """
        return self._sample_brownian_excursion_at(times)
