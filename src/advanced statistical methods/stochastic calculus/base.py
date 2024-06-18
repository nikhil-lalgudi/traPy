from abc import ABC, abstractmethod
import numpy as np


def check_positive_integer(value):
    if not (isinstance(value, int) and value > 0):
        raise ValueError("Value must be a positive integer")


def check_positive_number(value, name):
    if not (isinstance(value, (int, float)) and value > 0):
        raise ValueError(f"{name} must be a positive number")


def generate_times(t, n):
    """Generate a sequence of times."""
    return np.linspace(0, t, n + 1)


class BaseProcess(ABC):
    def __init__(self, rng=None):
        self._rng = rng

    @property
    def rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    @rng.setter
    def rng(self, value):
        if value is None:
            self._rng = None
        elif isinstance(value, (np.random.RandomState, np.random.Generator)):
            self._rng = value
        else:
            raise TypeError("rng must be of type `numpy.random.Generator`")

    @abstractmethod
    def sample(self, n):  # pragma: no cover
        pass


class BaseSequenceProcess(BaseProcess, ABC):
    pass


class BaseTimeProcess(BaseProcess, ABC):
    """Base class to be subclassed to most process classes.

    Contains properties and functions related to times and continuous-time
    processes.
    """

    def __init__(self, t=1, rng=None):
        super().__init__(rng=rng)
        self.t = t
        self._n = None
        self._times = None

    @property
    def t(self):
        """End time of the process."""
        return self._t

    @t.setter
    def t(self, value):
        check_positive_number(value, "Time end")
        self._t = float(value)

    def _set_times(self, n):
        if self._n != n:
            check_positive_integer(n)
            self._n = n
            self._times = generate_times(self.t, n)

    def times(self, n):
        """Generate times associated with n increments on [0, t].

        :param int n: the number of increments
        """
        self._set_times(n)
        return self._times
