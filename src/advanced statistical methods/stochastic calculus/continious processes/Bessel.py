import numpy as np
import matplotlib.pyplot as plt
from error_handler import check_positive_integer
from Brownian-Motion import BrownianMotion

class BesselProcess(BrownianMotion):
    def __init__(self, dimension=1, t=1, rng=None):
        super().__init__(drift=0, scale=1, t=t, rng=rng)
        self.dimension = dimension

    def __str__(self):
        return f"Bessel process in {self.dimension}-dimensional space on [0, {self.t}]"

    def __repr__(self):
        return f"BesselProcess(dimension={self.dimension}, t={self.t})"

    @property
    def dimension(self):
        """Dimension of the Euclidean space."""
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        check_positive_integer(value)
        self._dimension = value

    def _sample_bessel_process(self, n):
        """Generate a realization of a Bessel process."""
        bm = np.array([self._sample_brownian_motion(n) for _ in range(self.dimension)])
        bessel_path = np.sqrt(np.sum(bm**2, axis=0))
        return bessel_path

    def _sample_bessel_process_at(self, times):
        """Generate a realization of a Bessel process at specified times."""
        bm = np.array([self._sample_brownian_motion_at(times) for _ in range(self.dimension)])
        bessel_path = np.sqrt(np.sum(bm**2, axis=0))
        return bessel_path

    def sample(self, n):
        """Generate a realization.

        :param int n: the number of increments to generate.
        """
        return self._sample_bessel_process(n)

    def sample_at(self, times):
        """Generate a realization using specified times.

        :param times: a vector of increasing time values at which to generate
            the realization
        """
        return self._sample_bessel_process_at(times)