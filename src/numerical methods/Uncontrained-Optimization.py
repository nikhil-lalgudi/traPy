import numpy as np

from config import log_execution, measure_time

class UnconstrainedOptimization:
    def __init__(self, f, grad_f, hess_f=None, grad_g=None):
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.grad_g = grad_g

    @log_execution
    @measure_time
    def first_order_conditions(self, x):
        grad = self.grad_f(x)
        return np.all(np.isclose(grad, 0))

    @log_execution
    @measure_time
    def second_order_conditions(self, x):
        if self.hess_f is None:
            raise ValueError("Hessian function is not provided.")
        grad = self.grad_f(x)
        hess = self.hess_f(x)
        return np.all(np.isclose(grad, 0)) and np.all(np.linalg.eigvals(hess) > 0)

    @log_execution
    @measure_time
    def line_search(self, x0, alpha0, p, c1=1e-4, c2=0.9, max_iter=100):
        alpha = alpha0
        x = x0
        for _ in range(max_iter):
            if self.f(x + alpha * p) <= self.f(x) + c1 * alpha * np.dot(self.grad_f(x), p) and \
               np.dot(self.grad_f(x + alpha * p), p) >= c2 * np.dot(self.grad_f(x), p):
                return x + alpha * p
            alpha *= 0.5
        return x + alpha * p

    @log_execution
    @measure_time
    def newton_method(self, x0, tol=1e-5, max_iter=100):
        if self.hess_f is None:
            raise ValueError("Hessian function is not provided.")
        x = x0
        for _ in range(max_iter):
            grad = self.grad_f(x)
            hess = self.hess_f(x)
            if np.linalg.norm(grad) < tol:
                return x
            x -= np.linalg.solve(hess, grad)
        raise ValueError("Maximum number of iterations reached without convergence.")

    @log_execution
    @measure_time
    def constrained_optimization(self, g, x0, lambda0, tol=1e-5, max_iter=100):
        if self.grad_g is None:
            raise ValueError("Gradient of the constraint function is not provided.")
        
        def lagrangian(x, lambdas):
            return self.f(x) + np.dot(lambdas, g(x))
        
        def grad_lagrangian(x, lambdas):
            return self.grad_f(x) + np.dot(self.grad_g(x).T, lambdas)
        
        x = x0
        lambdas = lambda0
        for _ in range(max_iter):
            grad_l = grad_lagrangian(x, lambdas)
            hess_l = self.hess_f(x) if self.hess_f is not None else np.eye(len(x))
            if np.linalg.norm(grad_l) < tol:
                return x, lambdas
            x -= np.linalg.solve(hess_l, grad_l)
            lambdas += g(x)
        raise ValueError("Maximum number of iterations reached without convergence.")
    
    @staticmethod
    def plot_contour(f, x_range, y_range, levels=50):
        x = np.linspace(*x_range, 400)
        y = np.linspace(*y_range, 400)
        X, Y = np.meshgrid(x, y)
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=levels, cmap='viridis')
        plt.colorbar()