import numpy as np
from error_handler import check_numeric, check_positive_integer, check_numeric_or_single_arg_callable, ensure_single_arg_constant_function
from base import BaseTimeProcess
from Vasicek import VasicekProcess, ExtendedVasicekProcess

class MultifactorVasicekProcess(BaseTimeProcess):
    def __init__(self, factors, t=1, rng=None):
        super().__init__(t=t, rng=rng)
        self.factors = factors

    def __str__(self):
        return f"Multifactor Vasicek process with {len(self.factors)} factors on [0, {self.t}]"

    def __repr__(self):
        return f"MultifactorVasicekProcess(factors={self.factors}, t={self.t})"

    def _sample(self, n, initial):
        check_positive_integer(n)
        delta_t = self.t / n
        num_factors = len(self.factors)
        s = np.zeros((n + 1, num_factors))
        s[0, :] = initial
        for i in range(n):
            t = i * delta_t
            for j, factor in enumerate(self.factors):
                speed = ensure_single_arg_constant_function(factor["speed"])
                mean = ensure_single_arg_constant_function(factor["mean"])
                vol = ensure_single_arg_constant_function(factor["vol"])
                s[i+1, j] = (
                    s[i, j]
                    + speed(t) * (mean(t) - s[i, j]) * delta_t
                    + vol(t) * np.sqrt(delta_t) * self.rng.normal()
                )
        return s

    def sample(self, n, initial):
        if not isinstance(initial, (list, np.ndarray)) or len(initial) != len(self.factors):
            raise ValueError("Initial values must be a list or array with length equal to the number of factors.")
        return self._sample(n, initial)


class MultifactorCIRProcess(BaseTimeProcess):
    def __init__(self, factors, t=1, rng=None):
        super().__init__(t=t, rng=rng)
        self.factors = factors

    def __str__(self):
        return f"Multifactor CIR process with {len(self.factors)} factors on [0, {self.t}]"

    def __repr__(self):
        return f"MultifactorCIRProcess(factors={self.factors}, t={self.t})"

    def _sample(self, n, initial):
        check_positive_integer(n)
        delta_t = self.t / n
        num_factors = len(self.factors)
        s = np.zeros((n + 1, num_factors))
        s[0, :] = initial
        for i in range(n):
            t = i * delta_t
            for j, factor in enumerate(self.factors):
                speed = ensure_single_arg_constant_function(factor["speed"])
                mean = ensure_single_arg_constant_function(factor["mean"])
                vol = ensure_single_arg_constant_function(factor["vol"])
                s[i+1, j] = (
                    s[i, j]
                    + speed(t) * (mean(t) - s[i, j]) * delta_t
                    + vol(t) * np.sqrt(s[i, j]) * np.sqrt(delta_t) * self.rng.normal()
                )
                s[i+1, j] = max(s[i+1, j], 0)
        return s

    def sample(self, n, initial):
        if not isinstance(initial, (list, np.ndarray)) or 
        len(initial) != len(self.factors):
            raise ValueError("Initial values must be a list or array with length equal to the number of factors.")
        return self._sample(n, initial)