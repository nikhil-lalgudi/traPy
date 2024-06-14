import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

# Decorator for a single step
def step(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Discrete Function Map Class
class DiscreteMap:
    def __init__(self, map_func, y0, steps):
        self.map_func = map_func
        self.y = y0
        self.steps = steps
    
    @step
    def apply_map(self, y):
        return self.map_func(y)
    
    def iterate(self):
        trajectory = [self.y]
        for _ in range(self.steps):
            self.y = self.apply_map(self.y)
            trajectory.append(self.y)
        return np.array(trajectory)

# Gillespie Simulation Class
class GillespieSimulation:
    def __init__(self, reactions, rates, y0, t0, t1):
        self.reactions = reactions
        self.rates = rates
        self.y = np.array(y0)
        self.t = t0
        self.t1 = t1
        self.time = [t0]
        self.trajectory = [y0]
    
    def propensity(self):
        return np.array([rate(self.y) for rate in self.rates])
    
    def next_reaction_time(self, a0):
        return np.random.exponential(1.0 / a0)
    
    def choose_reaction(self, a):
        r = np.random.uniform(0, 1)
        cumulative_sum = np.cumsum(a)
        return np.searchsorted(cumulative_sum, r * cumulative_sum[-1])
    
    def simulate(self):
        while self.t < self.t1:
            a = self.propensity()
            a0 = np.sum(a)
            if a0 == 0:
                break
            dt = self.next_reaction_time(a0)
            self.t += dt
            reaction_index = self.choose_reaction(a)
            self.y += self.reactions[reaction_index]
            self.time.append(self.t)
            self.trajectory.append(self.y.copy())
        return np.array(self.time), np.array(self.trajectory)

# Markov Process Class
class MarkovProcess:
    def __init__(self, transition_matrix, y0, steps):
        self.transition_matrix = transition_matrix
        self.y = y0
        self.steps = steps
    
    def next_state(self):
        return np.random.choice(len(self.transition_matrix), p=self.transition_matrix[self.y])
    
    def simulate(self):
        trajectory = [self.y]
        for _ in range(self.steps):
            self.y = self.next_state()
            trajectory.append(self.y)
        return np.array(trajectory)

# Example usage within a library
def example_usage():
    # Logistic map as a discrete map example
    def logistic_map(x, r=3.9):
        return r * x * (1 - x)
    
    y0 = 0.1
    steps = 100
    map_solver = DiscreteMap(logistic_map, y0, steps)
    map_trajectory = map_solver.iterate()
    
    # Gillespie simulation example
    reactions = np.array([[-1, 1], [1, -1]])  # Reaction stoichiometries
    rates = [
        lambda x: 0.1 * x[0],  # Rate for the first reaction
        lambda x: 0.02 * x[1]  # Rate for the second reaction
    ]
    y0 = [50, 100]
    t0, t1 = 0, 10
    gillespie_solver = GillespieSimulation(reactions, rates, y0, t0, t1)
    gillespie_time, gillespie_trajectory = gillespie_solver.simulate()
    
    # Markov process example
    transition_matrix = np.array([
        [0.9, 0.1],
        [0.5, 0.5]
    ])
    y0 = 0
    steps = 100
    markov_solver = MarkovProcess(transition_matrix, y0, steps)
    markov_trajectory = markov_solver.simulate()
    
    return {
        "DiscreteMap": map_trajectory,
        "Gillespie": (gillespie_time, gillespie_trajectory),
        "MarkovProcess": markov_trajectory
    }

# Function to access the results
def get_integration_results():
    results = example_usage()
    return results

# Functions to process and visualize the results
def process_discrete_map(map_trajectory):
    """Process the trajectory of a discrete map."""
    # Calculate basic statistics
    mean_value = np.mean(map_trajectory)
    std_deviation = np.std(map_trajectory)
    min_value = np.min(map_trajectory)
    max_value = np.max(map_trajectory)
    
    # Detect fixed points
    fixed_points = np.where(np.diff(map_trajectory) == 0)[0]
    
    return {
        'trajectory': map_trajectory,
        'mean': mean_value,
        'std_deviation': std_deviation,
        'min': min_value,
        'max': max_value,
        'fixed_points': fixed_points
    }

def process_gillespie(gillespie_time, gillespie_trajectory):
    """Process the results of a Gillespie simulation."""
    # Calculate basic statistics
    final_states = gillespie_trajectory[-1]
    mean_final_state = np.mean(final_states)
    std_final_state = np.std(final_states)
    min_final_state = np.min(final_states)
    max_final_state = np.max(final_states)
    
    # Reaction rates over time
    reaction_rates = np.diff(gillespie_trajectory, axis=0) / np.diff(gillespie_time[:, None], axis=0)
    
    return {
        'time': gillespie_time,
        'trajectory': gillespie_trajectory,
        'mean_final_state': mean_final_state,
        'std_final_state': std_final_state,
        'min_final_state': min_final_state,
        'max_final_state': max_final_state,
        'reaction_rates': reaction_rates
    }

def process_markov(markov_trajectory):
    """Process the trajectory of a Markov process."""
    # Calculate basic statistics
    state_counts = np.bincount(markov_trajectory)
    state_probabilities = state_counts / len(markov_trajectory)
    
    # State transitions
    transitions = np.zeros((len(state_counts), len(state_counts)))
    for i in range(len(markov_trajectory) - 1):
        transitions[markov_trajectory[i], markov_trajectory[i + 1]] += 1
    
    transition_probabilities = transitions / np.sum(transitions, axis=1, keepdims=True)
    
    return {
        'trajectory': markov_trajectory,
        'state_counts': state_counts,
        'state_probabilities': state_probabilities,
        'transitions': transitions,
        'transition_probabilities': transition_probabilities
    }

def plot_discrete_map(map_trajectory):
    plt.figure()
    plt.plot(map_trajectory)
    plt.title('Discrete Map Trajectory')
    plt.xlabel('Step')
    plt.ylabel('State')
    plt.grid(True)
    plt.show()

def plot_gillespie(gillespie_time, gillespie_trajectory):
    plt.figure()
    plt.plot(gillespie_time, gillespie_trajectory)
    plt.title('Gillespie Simulation')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.grid(True)
    plt.show()

def plot_markov(markov_trajectory):
    plt.figure()
    plt.plot(markov_trajectory)
    plt.title('Markov Process Trajectory')
    plt.xlabel('Step')
    plt.ylabel('State')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    results = get_integration_results()
    
    # Process results
    map_trajectory = process_discrete_map(results["DiscreteMap"])
    gillespie_time, gillespie_trajectory = process_gillespie(*results["Gillespie"])
    markov_trajectory = process_markov(results["MarkovProcess"])
    
    # Plot results
    plot_discrete_map(map_trajectory)
    plot_gillespie(gillespie_time, gillespie_trajectory)
    plot_markov(markov_trajectory)
