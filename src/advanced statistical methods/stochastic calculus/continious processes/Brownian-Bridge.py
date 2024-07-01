import numpy as np
import matplotlib.pyplot as plt
from error_handler import check_numeric

from Brownian-Motion import BrownianMotion

class BrownianBridge(BrownianMotion):
    def __init__(self, b=0, t=1, rng=None):
        super().__init__(drift=0, scale=1, t=t, rng=rng)
        self.b = b

    def __str__(self):
        return f"Brownian bridge from 0 to {self.b} on [0, {self.t}]"

    def __repr__(self):
        return f"BrownianBridge(b={self.b}, t={self.t})"

    @property
    def b(self):
        """Right endpoint value."""
        return self._b

    @b.setter
    def b(self, value):
        check_numeric(value, "Right endpoint value")
        self._b = value

    def _sample_brownian_bridge(self, n, b=None):
        """Generate a realization of a Brownian bridge."""
        if b is None:
            b = self.b
        times = np.linspace(0, self.t, n + 1)
        bm = self._sample_brownian_motion(n)
        bridge = bm + times * (b - bm[-1]) / self.t
        return bridge

    def _sample_brownian_bridge_at(self, times, b=None):
        if b is None:
            b = self.b
        bm = self._sample_brownian_motion_at(times)
        bridge = bm + np.array(times) * (b - bm[-1]) / times[-1]
        return bridge

    def sample(self, n):
        return self._sample_brownian_bridge(n)

    def sample_at(self, times, b=None):
        return self._sample_brownian_bridge_at(times, b)