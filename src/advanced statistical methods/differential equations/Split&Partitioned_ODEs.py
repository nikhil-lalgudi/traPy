import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

%matplotlib inline

class Part1:
    def __init__(self, f):
        """
        Initialize the first part of the system.
        
        Parameters:
        f (callable): Function defining the differential equation for q.
        """
        self.f = f

    def __call__(self, y, t):
        q, p = y[:len(y)//2], y[len(y)//2:]
        dq_dt = self.f(p, t)
        return np.concatenate((dq_dt, np.zeros_like(dq_dt)))


class Part2:
    def __init__(self, g):
    
        self.g = g

    def __call__(self, y, t):
        q, p = y[:len(y)//2], y[len(y)//2:]
        dp_dt = self.g(q, t)
        return np.concatenate((np.zeros_like(dp_dt), dp_dt))


class SplitPartitionedODEs:
    def __init__(self, part1, part2):
        """
        Initialize the solver for split and partitioned ODEs.
        
        Parameters:
        part1 (callable): Instance of class defining the first part of the split ODE.
        part2 (callable): Instance of class defining the second part of the split ODE.
        """
        self.part1 = part1
        self.part2 = part2

    def solve(self, y0, t_span, dt):
        """
        Solve the split ODE using a simple splitting method.
        
        Parameters:
        y0 (ndarray): Initial conditions.
        t_span (tuple): Time interval (start, end).
        dt (float): Time step.
        
        Returns:
        t (ndarray): Time points.
        y (ndarray): Solution array.
        """
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            y_half = y[i-1] + dt/2 * self.part1(y[i-1], t[i-1])
            y[i] = y_half + dt * self.part2(y_half, t[i-1] + dt/2)
            y[i] = y[i] + dt/2 * self.part1(y[i], t[i])
        
        return t, y

    def plot_solution(self, t, y):
        """
        Plot the solution of the split ODE.
        
        Parameters:
        t (ndarray): Time points.
        y (ndarray): Solution array.
        """
        plt.figure(figsize=(12, 6))
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i], label=f'Variable {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.legend()
        plt.title('Solution of Split ODE')
        plt.show()


class SymplecticIntegrator:
    def __init__(self, f, g):
        """
        Initialize the symplectic integrator.
        
        Parameters:
        f (callable): Function defining the differential equation for q.
        g (callable): Function defining the differential equation for p.
        """
        self.f = f
        self.g = g

    def solve(self, q0, p0, t_span, dt):
        """
        Solve the Hamiltonian system using the symplectic Euler method.
        
        Parameters:
        q0 (float): Initial condition for q.
        p0 (float): Initial condition for p.
        t_span (tuple): Time interval (start, end).
        dt (float): Time step.
        
        Returns:
        t (ndarray): Time points.
        q (ndarray): Solution for q.
        p (ndarray): Solution for p.
        """
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        q = np.zeros(len(t))
        p = np.zeros(len(t))
        q[0] = q0
        p[0] = p0

        for i in range(1, len(t)):
            p[i] = p[i-1] - dt * self.f(q[i-1], t[i-1])
            q[i] = q[i-1] + dt * self.g(p[i], t[i-1])
        
        return t, q, p

    def plot_solution(self, t, q, p):
        """
        Plot the solution of the Hamiltonian system.
        
        Parameters:
        t (ndarray): Time points.
        q (ndarray): Solution for q.
        p (ndarray): Solution for p.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(t, q, label='q')
        plt.plot(t, p, label='p')
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.legend()
        plt.title('Solution of Hamiltonian System')
        plt.show()


class IMEXMethod:
    def __init__(self, f_explicit, f_implicit, max_iter=10, tol=1e-6):
        """
        Initialize the IMEX method solver.
        
        Parameters:
        f_explicit (callable): Function defining the explicit part of the ODE.
        f_implicit (callable): Function defining the implicit part of the ODE.
        max_iter (int): Maximum number of iterations for the implicit solver.
        tol (float): Tolerance for convergence of the implicit solver.
        """
        self.f_explicit = f_explicit
        self.f_implicit = f_implicit
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, y0, t_span, dt):
        """
        Solve the ODE using the IMEX method.
        
        Parameters:
        y0 (ndarray): Initial conditions.
        t_span (tuple): Time interval (start, end).
        dt (float): Time step.
        
        Returns:
        t (ndarray): Time points.
        y (ndarray): Solution array.
        """
        t = np.arange(t_span[0], t_span[1] + dt, dt)
        y = np.zeros((len(t), len(y0)))
        y[0] = y0

        for i in range(1, len(t)):
            y_explicit = y[i-1] + dt * self.f_explicit(y[i-1], t[i-1])
            y_implicit = y_explicit
            for _ in range(self.max_iter):
                y_new = y_explicit + dt * self.f_implicit(y_implicit, t[i])
                if np.linalg.norm(y_new - y_implicit) < self.tol:
                    break
                y_implicit = y_new
            y[i] = y_implicit
        
        return t, y

    def plot_solution(self, t, y):
        """
        Plot the solution of the ODE solved by the IMEX method.
        
        Parameters:
        t (ndarray): Time points.
        y (ndarray): Solution array.
        """
        plt.figure(figsize=(12, 6))
        for i in range(y.shape[1]):
            plt.plot(t, y[:, i], label=f'Variable {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.legend()
        plt.title('Solution of ODE using IMEX Method')
        plt.show()

"""
# Example usage of the AdvancedODEs library
if __name__ == "__main__":
    # Split and Partitioned ODEs example
    def f1(p, t):
        return np.array([p[0]])

    def f2(q, t):
        return np.array([-q[0]])

    y0 = [1.0, 0.0]
    t_span = (0, 10)
    dt = 0.01

    part1 = HamiltonianPart1(f1)
    part2 = HamiltonianPart2(f2)

    split_solver = SplitPartitionedODEs(part1, part2)
    t, y = split_solver.solve(y0, t_span, dt)
    split_solver.plot_solution(t, y)

    # Symplectic Integrator example
    def f(q, t):
        return q

    def g(p, t):
        return p

    q0 = 1.0
    p0 = 0.0
    t_span = (0, 10)
    dt = 0.01

    symplectic_solver = SymplecticIntegrator(f, g)
    t, q, p = symplectic_solver.solve(q0, p0, t_span, dt)
    symplectic_solver.plot_solution(t, q, p)

    # IMEX Method example
    def f_explicit(y, t):
        return -y

    def f_implicit(y, t):
        return -0.5 * y

    y0 = [1.0]
    t_span = (0, 10)
    dt = 0.01

    imex_solver = IMEXMethod(f_explicit, f_implicit)
    t, y = imex_solver.solve(y0, t_span, dt)
    imex_solver.plot_solution(t, y)
"""