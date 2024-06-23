import numpy as np
from scipy.integrate import solve_ivp

def default_algorithm(prob, **kwargs):
    # Define default algorithm and additional kwargs if needed
    return 'EM', {}

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
        if solver == 'EM':
            u[i] = u_t + prob['f'](t, u_t) * dt + prob['g'](t, u_t) * dw
        elif solver == 'Milstein':
            g_t = prob['g'](t, u_t)
            u[i] = u_t + prob['f'](t, u_t) * dt + g_t * dw + 0.5 * g_t * g_t * (dw**2 - dt)
        # Add more solvers as needed

    return t_span, u

# Example usage within a library
def example_usage():
    # Define the deterministic part f and stochastic part g of the RODE
    def f(t, u):
        return -0.5 * u

    def g(t, u):
        return 0.1 * u

    # Define the problem dictionary
    prob = {
        'f': f,
        'g': g,
        'u0': [1.0],
        't_span': np.linspace(0, 10, 1001)  # 10 seconds, 1001 steps
    }

    # Solve using the Euler-Maruyama method
    t_span, u_em = solve_rode(prob, solver='EM')

    # Solve using the Milstein method
    t_span, u_milstein = solve_rode(prob, solver='Milstein')

    return t_span, u_em, u_milstein

# Function to access the results
def get_integration_results():
    t_span, u_em, u_milstein = example_usage()
    return t_span, u_em, u_milstein

# Functions to process and visualize the results
def process_rode(t_span, u):
    """Process the results of an RODE simulation."""
    # Calculate basic statistics
    mean_trajectory = np.mean(u, axis=0)
    std_trajectory = np.std(u, axis=0)
    final_state_mean = np.mean(u[-1])
    final_state_std = np.std(u[-1])
    
    return {
        't_span': t_span,
        'trajectory': u,
        'mean_trajectory': mean_trajectory,
        'std_trajectory': std_trajectory,
        'final_state_mean': final_state_mean,
        'final_state_std': final_state_std
    }

def plot_rode(t_span, u, title='RODE Simulation'):
    """Plot the results of an RODE simulation."""
    plt.figure()
    plt.plot(t_span, u)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    t_span, u_em, u_milstein = get_integration_results()
    
    # Process results
    em_results = process_rode(t_span, u_em)
    milstein_results = process_rode(t_span, u_milstein)
    
    # Plot results
    plot_rode(t_span, u_em, title='Euler-Maruyama Method')
    plot_rode(t_span, u_milstein, title='Milstein Method')
