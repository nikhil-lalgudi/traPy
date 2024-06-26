import numpy as np
from base import BaseTimeProcess
from error_handler import check_positive_integer

class GaussianNoise(BaseTimeProcess):
    def __init__(self, t=1, rng=None):
        super().__init__(t=t, rng=rng)

    def __str__(self):
        return "Gaussian noise generator on interval [0, {t}]".format(t=str(self.t))

    def __repr__(self):
        return "GaussianNoise(t={t})".format(t=str(self.t))

    def _sample_gaussian_noise(self, n): 
        check_positive_integer(n)
        delta_t = 1.0 * self.t / n
        noise = self.rng.normal(scale=np.sqrt(delta_t), size=n)
        return noise

    def _sample_gaussian_noise_at(self, times):    
        if times[0] != 0:
            times = np.concatenate(([0], times))
            
        increments = np.diff(times)
        noise = np.array([self.rng.normal(scale=np.sqrt(inc)) for inc in increments])
        return noise

    def sample(self, n):
        return self._sample_gaussian_noise(n)

    def sample_at(self, times):    
        return self._sample_gaussian_noise_at(times)
