import numpy as np
from scipy.integrate import solve_ivp

def solve_rode(prob, **kwargs):
    alg, extra_kwargs = default_algorithm(prob, **kwargs)
    solver = kwargs.get('solver', 'EM')  # Default to Euler-Maruyama for RODEs
    
    t_span = np.array(prob['t_span'])
    u0 = np.array(prob['u0'])
    dt = t_span[1] - t_span[0]
    n_steps = len(t_span)
    u = np.zeros((n_steps, len(u0)))
    u[0] = u0

    for i in range(1, n_steps):
        t = t_span[i-1]
        u_t = u[i-1]
        dw = np.random.normal(0, np.sqrt(dt), size=u0.shape)
        u[i] = u_t + prob['f'](t, u_t) * dt + prob['g'](t, u_t) * dw

    return t_span, u