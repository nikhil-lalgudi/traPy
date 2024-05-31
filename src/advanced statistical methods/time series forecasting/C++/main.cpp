#include "solv.hpp"
#include "PDE_solver.hpp"

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace std;

int main()
{
   double A[4][4] = { {10, 5, 0, 0},
                      {2, 10, 5, 0},
                      {0, 2, 10, 5},
                      {0, 0, 2, 10} };
   double b[4] = {20, 37, 54, 46};

   double aa = 2, bb = 10, cc = 5;

   const double w = 1;
   const double eps = 1e-10;
   const int N_max = 100;

   cout << "Matrix A: \n";
   print_matrix(A);
   cout << "Vector b: \n";
   print_array(b);
   
   double* x0 = new double[N]();
   double* x1 = new double[N]();
   double* x = SOR_aabbcc(aa, bb, cc, b, x0, x1, N, w, eps, N_max);
   cout << "Solution x: \n";
   print_array(x);
   delete[] x0;
   delete[] x1;
  double r = 0.1;
  double sig = 0.2;
  double S = 100.0;
  double K = 100.0;
  double T = 1.;

  int Nspace = 3000;    
  int Ntime = 2000;     

  double w = 1.68;      
  
  cout << "The price is: " << PDE_solver(Nspace, Ntime, S, K, T, sig, r, w) << endl;

  return 0;
}

