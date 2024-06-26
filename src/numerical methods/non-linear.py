import numpy as np

from config import log_execution, measure_time

class NonLinearSolver:
    
    @staticmethod
    @log_execution
    @measure_time
    def bisection_method(f, a, b, tol=1e-5, max_iter=100):
        """Bisection method for solving non-linear equations."""
        if f(a) * f(b) >= 0:
            raise ValueError("Function values at the interval endpoints must have opposite signs.")
        
        for _ in range(max_iter):
            c = (a + b) / 2
            if f(c) == 0 or (b - a) / 2 < tol:
                return c
            elif f(c) * f(a) < 0:
                b = c
            else:
                a = c
        
        raise ValueError("Maximum number of iterations reached without convergence.")

    @staticmethod
    @log_execution
    @measure_time
    def newton_raphson_method(f, df, x0, tol=1e-5, max_iter=100):
        """Newton-Raphson method for solving non-linear equations."""
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            dfx = df(x)
            if dfx == 0:
                raise ValueError("Derivative is zero.")
            x_new = x - fx / dfx
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        
        raise ValueError("Maximum number of iterations reached without convergence.")

    @staticmethod
    @log_execution
    @measure_time
    def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
        """Secant method for solving non-linear equations."""
        for _ in range(max_iter):
            f0, f1 = f(x0), f(x1)
            if f1 - f0 == 0:
                raise ValueError("Zero denominator.")
            x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            if abs(x_new - x1) < tol:
                return x_new
            x0, x1 = x1, x_new
        
        raise ValueError("Maximum number of iterations reached without convergence.")

    @staticmethod
    @log_execution
    @measure_time
    def fixed_point_algorithm(g, x0, tol=1e-5, max_iter=100):
        """Fixed-point algorithm for solving non-linear equations."""
        x = x0
        for _ in range(max_iter):
            x_new = g(x)
            if abs(x_new - x) < tol:
                return x_new
            x = x_new
        
        raise ValueError("Maximum number of iterations reached without convergence.")
"""
# Example usage
if __name__ == "__main__":
    def f(x):
        return x**3 - x - 2
    
    def df(x):
        return 3*x**2 - 1
    
    def g(x):
        return np.cbrt(x + 2)
    
    solver = NonLinearSolver()
    
    # Bisection method
    root_bisection = solver.bisection_method(f, 1, 2)
    print(f"Bisection method root: {root_bisection}")
    
    # Newton-Raphson method
    root_newton_raphson = solver.newton_raphson_method(f, df, 1.5)
    print(f"Newton-Raphson method root: {root_newton_raphson}")
    
    # Secant method
    root_secant = solver.secant_method(f, 1, 2)
    print(f"Secant method root: {root_secant}")
    
    # Fixed-point algorithm
    root_fixed_point = solver.fixed_point_algorithm(g, 1)
    print(f"Fixed-point algorithm root: {root_fixed_point}")
"""