import numpy as np

def step(func):
    """Decorator for a single step of an ODE solver."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class DelayDE:
    def __init__(self, f, y0, t0, t1, dt, delay, eq_type='RDDE'):
        self.f = f  # Deterministic part
        self.y = y0
        self.t = t0
        self.t1 = t1
        self.dt = dt
        self.delay = delay
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
        if self.eq_type == 'NDDE':
            return y + self.f(y, delayed_y) * dt - self.f(delayed_y, delayed_y) * dt
        elif self.eq_type == 'RDDE':
            return y + self.f(y, delayed_y) * dt
        elif self.eq_type == 'DDAE':
            return y + np.linalg.solve(self.f(y, delayed_y), dt)
        else:
            raise ValueError(f"Unsupported equation type: {self.eq_type}")

    def integrate(self):
        for _ in range(self.steps):
            delayed_y = self.get_delayed_state(self.t)
            self.y = self.deterministic_step(self.y, delayed_y, self.dt)
            self.t += self.dt
            self.history.append((self.t, self.y))
        return self.y
"""
# Example usage within a library
def example_usage():
    # Define deterministic parts of the equation
    def f_ndde(y, delayed_y):
        return -0.5 * delayed_y + 0.1 * y

    def f_rdde(y, delayed_y):
        return -0.5 * delayed_y

    def f_ddae(y, delayed_y):
        return np.array([[1, 0], [0, 1]])  # Identity matrix for simplicity

    # Initial conditions
    y0 = np.array([1.0])
    t0 = 0.0
    t1 = 10.0
    dt = 0.01
    delay = 0.1

    # Create instances for each type of equation
    solver_ndde = DelayDE(f_ndde, y0, t0, t1, dt, delay, eq_type='NDDE')
    solver_rdde = DelayDE(f_rdde, y0, t0, t1, dt, delay, eq_type='RDDE')
    solver_ddae = DelayDE(f_ddae, y0, t0, t1, dt, delay, eq_type='DDAE')

    # Integrate each system
    result_ndde = solver_ndde.integrate()
    result_rdde = solver_rdde.integrate()
    result_ddae = solver_ddae.integrate()

    return {
        "NDDE": result_ndde,
        "RDDE": result_rdde,
        "DDAE": result_ddae
    }

# Function to access the results
def get_integration_results():
    results = example_usage()
    return results
"""