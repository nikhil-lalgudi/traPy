#include "PDE_solver.hpp"

// Path: src/advanced%20statistical%20methods/time%20series%20forecasting/C%2B%2B/PDE_solver.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

double* SOR_aabbcc(double aa, double bb, double cc, double* x_old, double* help_ptr, double* x_new, int size, double w, double eps, int N_max); // Placeholder for your SOR function

double PDE_solver(int Ns, int Nt, double S, double K, double T, double sig, double r, double w) {
    const double eps = 1e-10;
    const int N_max = 600;
  
    double S_max = 3 * K;
    double S_min = K / 3;
    double x_max = log(S_max);
    double x_min = log(S_min);

    double dx = (x_max - x_min) / (Ns - 1);
    double dt = T / Nt;

    double sig2 = sig * sig;
    double dxx = dx * dx;
    double aa = ((dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx));
    double bb = (1 + dt * (sig2 / dxx + r));
    double cc = (-(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx));

    // vector allocations  
    std::vector<double> x(Ns);
    std::vector<double> x_old(Ns - 2);
    std::vector<double> x_new(Ns - 2);
    std::vector<double> help_ptr(Ns - 2);

    for (unsigned int i = 0; i < Ns; ++i) // price vector
        x[i] = exp(x_min + i * dx);
  
    for (unsigned int i = 0; i < Ns - 2; ++i) // payoff
        x_old[i] = std::max(x[i + 1] - K, 0.0);

    // Backward iteration
    for (int k = Nt - 1; k >= 0; --k) {
        x_old[Ns - 3] -= cc * (S_max - K * exp(-r * (T - k * dt))); // offset
        x_new = SOR_aabbcc(aa, bb, cc, x_old.data(), help_ptr.data(), x_new.data(), Ns - 2, w, eps, N_max); // SOR solver

        if (k != 0) // swap the pointers (we don't need to allocate new memory) 
            std::swap(x_old, x_new);
    }

    // x_new is the solution!! 

    // binary search: Search for the points for the interpolation  
    int low = 1;
    int high = Ns - 2;
    int mid;
    double result = -1;

    if (S > x[high] || S < x[low]) {
        std::cerr << "error: Price S out of grid." << std::endl;
        return result;
    }
  
    while ((low + 1) != high) {
        mid = (low + high) / 2;
      
        if (std::fabs(x[mid] - S) < 1e-10) {
            result = x_new[mid - 1];
            return result;
        } else if (x[mid] < S) {
            low = mid;
        } else {
            high = mid;
        }
    }

    // linear interpolation
    result = x_new[low - 1] + (S - x[low]) * (x_new[high - 1] - x_new[low - 1]) / (x[high] - x[low]);
    return result;
}

