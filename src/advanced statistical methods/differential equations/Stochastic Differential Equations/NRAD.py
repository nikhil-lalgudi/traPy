# * Experimental support for stochastic neutral, retarded, and algebraic delay differential equations (SNDDEs, SRDDEs, and SDDAEs)

import numpy as np

def step(func):
    """Decorator for a single step of an ODE solver."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class StochasticDelayDE:
    def __init__(self, f, g, y0, t0, t1, dt, delay, noise_intensity, eq_type='SRDDE'):
        self.f = f  
        self.g = g 
        self.y = y0
        self.t = t0
        self.t1 = t1
        self.dt = dt
        self.delay = delay
        self.noise_intensity = noise_intensity
        self.eq_type = eq_type
        self.steps = int((t1 - t0) / dt)
        self.history = [(t0 - i * dt, y0) for i in range(int(delay / dt) + 1)]
        
    def get_delayed_state(self, t):
        for (time, state) in self.history:
            if time <= t - self.delay:
                return state
        return self.history[-1][1]

    @step
    def deterministic_step(self, y, delayed_y, dt):
        if self.eq_type == 'SNDDE':
            return y + self.f(y, delayed_y) * dt - self.f(delayed_y, delayed_y) * dt
        elif self.eq_type == 'SRDDE':
            return y + self.f(y, delayed_y) * dt
        elif self.eq_type == 'SDDAE':
            return y + np.linalg.solve(self.f(y, delayed_y), dt)
        else:
            raise ValueError(f"Unsupported equation type: {self.eq_type}")

    @step
    def stochastic_step(self, y, dt):
        noise = np.random.normal(0, np.sqrt(dt), size=y.shape)
        return y + self.g(y) * noise * self.noise_intensity

    def integrate(self):
        for _ in range(self.steps):
            delayed_y = self.get_delayed_state(self.t)
            self.y = self.deterministic_step(self.y, delayed_y, self.dt)
            self.y = self.stochastic_step(self.y, self.dt)
            self.t += self.dt
            self.history.append((self.t, self.y))
        return self.y

def example_usage():
    def f_sndde(y, delayed_y):
        return -0.5 * delayed_y + 0.1 * y

    def f_srdde(y, delayed_y):
        return -0.5 * delayed_y

    def f_sddae(y, delayed_y):
        return np.array([[1, 0], [0, 1]])  # Identity matrix for simplicity

    def g(y):
        return 0.1 * y

    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 10.0
    dt = 0.01
    delay = 0.1
    noise_intensity = 0.1

    # Create instances for each type of equation
    solver_sndde = StochasticDelayDE(f_sndde, g, y0, t0, t1, dt, delay, noise_intensity, eq_type='SNDDE')
    solver_srdde = StochasticDelayDE(f_srdde, g, y0, t0, t1, dt, delay, noise_intensity, eq_type='SRDDE')
    solver_sddae = StochasticDelayDE(f_sddae, g, y0, t0, t1, dt, delay, noise_intensity, eq_type='SDDAE')

    # Integrate each system
    result_sndde = solver_sndde.integrate()
    result_srdde = solver_srdde.integrate()
    result_sddae = solver_sddae.integrate()

    return {
        "SNDDE": result_sndde,
        "SRDDE": result_srdde,
        "SDDAE": result_sddae
    }

# Function to access the results
def get_integration_results():
    results = example_usage()
    return results
