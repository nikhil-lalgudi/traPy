import numpy as np
import matplotlib.pyplot as plt

class HamiltonianSystem:
    def __init__(self, H, dH_dq, dH_dp, q0, p0, t0, t1, dt):
        self.H = H          # Hamiltonian function
        self.dH_dq = dH_dq  # Partial derivative of H with respect to q
        self.dH_dp = dH_dp  # Partial derivative of H with respect to p
        self.q = q0         # Initial condition for position
        self.p = p0         # Initial condition for momentum
        self.t = t0         # Initial time
        self.t1 = t1        # Final time
        self.dt = dt        # Time step size
        self.steps = int((t1 - t0) / dt)
        self.trajectory_q = [q0]
        self.trajectory_p = [p0]
        self.time = [t0]
    
    def symplectic_euler_step(self, q, p, dt):
        p_new = p - self.dH_dq(q) * dt
        q_new = q + self.dH_dp(p_new) * dt
        return q_new, p_new
    
    def solve(self):
        for _ in range(self.steps):
            self.q, self.p = self.symplectic_euler_step(self.q, self.p, self.dt)
            self.t += self.dt
            self.trajectory_q.append(self.q)
            self.trajectory_p.append(self.p)
            self.time.append(self.t)
        return np.array(self.time), np.array(self.trajectory_q), np.array(self.trajectory_p)

# Example usage within a library
def example_usage():
    # Define the Hamiltonian function and its partial derivatives
    def H(q, p):
        return 0.5 * p**2 + 0.5 * q**2  # Harmonic oscillator
    
    def dH_dq(q):
        return q
    
    def dH_dp(p):
        return p
    
    # Initial conditions
    q0 = np.array([1.0])
    p0 = np.array([0.0])
    t0 = 0.0
    t1 = 10.0
    dt = 0.01
    
    # Create Hamiltonian system solver
    ham_system = HamiltonianSystem(H, dH_dq, dH_dp, q0, p0, t0, t1, dt)
    
    # Solve Hamiltonian system
    time, trajectory_q, trajectory_p = ham_system.solve()
    
    return time, trajectory_q, trajectory_p

# Functions to process and visualize the results
def process_hamiltonian_solution(time, trajectory_q, trajectory_p):
    # Basic statistics or analysis can be added here
    return time, trajectory_q, trajectory_p
"""
def plot_hamiltonian_solution(time, trajectory_q, trajectory_p):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, trajectory_q)
    plt.title('Hamiltonian System Solution - Position')
    plt.xlabel('Time')
    plt.ylabel('q')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, trajectory_p)
    plt.title('Hamiltonian System Solution - Momentum')
    plt.xlabel('Time')
    plt.ylabel('p')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    time, trajectory_q, trajectory_p = example_usage()
    
    # Process results
    processed_time, processed_trajectory_q, processed_trajectory_p = process_hamiltonian_solution(time, trajectory_q, trajectory_p)
    
    # Plot results
    plot_hamiltonian_solution(processed_time, processed_trajectory_q, processed_trajectory_p)
"""