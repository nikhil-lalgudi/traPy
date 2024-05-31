#ifndef PDE_solver.hpp
#define PDE_solver.hpp

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <string>

double PDE_solver(int Ns, int Nt, double S, double K, double T, double sig, double r, double w);

#endif