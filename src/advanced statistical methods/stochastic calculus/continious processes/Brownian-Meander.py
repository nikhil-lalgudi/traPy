import numpy as np
from error_handler import check_numeric, check_positive_integer
from brownian_bridge import BrownianBridge

class BrownianMeander(BrownianBridge):
    def __init__(self, t=1, rng=None):
        super().__init__(b=0, t=t, rng=rng)

    def __str__(self):
        return f"Brownian meander on [0, {self.t}]"

    def __repr__(self):
        return f"BrownianMeander(t={self.t})"

    def _sample_brownian_meander(self, n):
        """Generate a Brownian meander."""
        brownian_bridge = self._sample_brownian_bridge(n)
        meander = brownian_bridge - np.min(brownian_bridge)
        return meander

    def _sample_brownian_meander_at(self, times):
        """Generate a Brownian meander at specified times."""
        brownian_bridge = self._sample_brownian_bridge_at(times)
        meander = brownian_bridge - np.min(brownian_bridge)
        return meander

    def sample(self, n):
        """Generate a realization.

        :param int n: the number of increments to generate.
        """
        return self._sample_brownian_meander(n)

    def sample_at(self, times):
        """Generate a realization using specified times.

        :param times: a vector of increasing time values at which to generate
            the realization
        """
        return self._sample_brownian_meander_at(times)