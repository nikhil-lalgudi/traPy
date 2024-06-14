import numpy as np

def step(func):
    """Decorator for a single step of an ODE solver."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class SplitODE:
    def __init__(self, H, G, y0, t0, t1, dt):
        self.H = H  # Hamiltonian part
        self.G = G  # Gradient part
        self.y = y0
        self.t = t0
        self.t1 = t1
        self.dt = dt
        self.steps = int((t1 - t0) / dt)

    @step
    def hamiltonian_step(self, y, dt):
        q, p = np.split(y, 2)
        dqdt = np.gradient(self.H(q, p), q)
        dpdt = np.gradient(self.H(q, p), p)
        q_new = q + dqdt * dt
        p_new = p + dpdt * dt
        return np.concatenate([q_new, p_new])

    @step
    def gradient_step(self, y, dt):
        q, p = np.split(y, 2)
        dqdt = np.gradient(self.G(q, p), q)
        dpdt = np.gradient(self.G(q, p), p)
        q_new = q + dqdt * dt
        p_new = p + dpdt * dt
        return np.concatenate([q_new, p_new])

    def integrate(self):
        for _ in range(self.steps):
            self.y = self.hamiltonian_step(self.y, self.dt / 2)
            self.y = self.gradient_step(self.y, self.dt)
            self.y = self.hamiltonian_step(self.y, self.dt / 2)
            self.t += self.dt
        return self.y

def H(q, p):
    """Hamiltonian function."""
    return 0.5 * np.sum(p**2) + 0.5 * np.sum(q**2)

def G(q, p):
    """Gradient part of the system."""
    return np.sin(q)


add the example usage
