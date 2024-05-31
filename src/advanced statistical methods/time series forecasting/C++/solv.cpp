#include "solv.hpp"

// Path: src/advanced%20statistical%20methods/time%20series%20forecasting/C%2B%2B/solv.cpp

#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cstdlib>

using namespace std;

double dist_square(double *a, double *b){
    double sum = 0;
    for (int i = 0; i < N; i++){
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

double dist_square2(double* a, double* b, int NN){
    double sum = 0;
    for (int i = 0; i < NN; i++){
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

void print_array(double *arr){
    for (int i = 0; i < N; i++){
        cout << arr[i] << "\t";
    }
    cout << endl;
}

void print_array_2(double* arr, int NN){
    for (int i = 0; i < NN; i++){
        cout << arr[i] << "\t";
    }
    cout << endl;
}

double dist_square(double* a, double* b)
{
 double dist = 0;
 
 for (int i=0; i<N; ++i)
   {
     dist += (a[i] - b[i]) * (a[i] - b[i]);
   }
 return dist;
}

double dist_square2(double* a, double* b, int NN)
{
 double dist = 0;
 
 for (int i=0; i<NN; ++i)
   {
     dist += (a[i] - b[i]) * (a[i] - b[i]);
   }
 return dist;
}

double* SOR(double A[N][N], double* b, double w, double eps, int N_max)
{
 double* x0 = new double[N]();
 double* x1 = new double[N]();
 double S;
 
 for (int k=0; k<N_max+1; ++k)
   {
     for (int i=0; i<N; ++i)
   {
     S = 0;
     for (int j=0; j<N; ++j)
       {
         if (j!=i)
   	S += A[i][j] * x1[j];
       }
     x1[i] = (1-w)*x1[i] + (w / A[i][i]) * (b[i] - S);  
   }
     if (dist_square(x0,x1) < eps*eps)
   return x1;
     
     for (int i=0; i<N; ++i)
   x0[i] = x1[i];
   }
 cout << "Fail to converge in " << N_max << " iterations" << endl;
 delete[] x0;
 return x1;
}

double* SOR_trid(double A[N][N], double* b, double w, double eps, int N_max)
{
 double* x0 = new double[N]();
 double* x1 = new double[N]();
 double S;
 
 for (int k=0; k<N_max+1; ++k)
   {
     for (int i=0; i<N; ++i)
   {
     if (i==0)
       S = A[0][1] * x1[1];
     else if (i==N-1)
       S = A[N-1][N-2] * x1[N-2];
     else
       S = A[i][i-1] * x1[i-1] + A[i][i+1] * x1[i+1];
     
     x1[i] = (1-w)*x1[i] + (w / A[i][i]) * (b[i] - S);  
   }
     if (dist_square(x0,x1) < eps*eps)
   return x1;
     
     for (int i=0; i<N; ++i)
   x0[i] = x1[i];
   }
 cout << "Fail to converge in " << N_max << " iterations" << endl;
 delete[] x0;
 return x1;
}

double* SOR_abc(double aa, double bb, double cc, 
double* b, int NN, double w, double eps, int N_max)
{
 double* x0 = new double[NN]();
 double* x1 = new double[NN]();
 double S;
 
 for (int k=0; k<N_max+1; ++k)
   {
     for (int i=0; i<NN; ++i)
   {
     if (i==0)
       S = cc * x1[1];
     else if (i==NN-1)
       S = aa * x1[NN-2];
     else
       S = aa * x1[i-1] + cc * x1[i+1];
     
     x1[i] = (1-w)*x1[i] + (w / bb) * (b[i] - S);  
   }
     if (dist_square2(x0,x1,NN) < eps*eps)
   {
     delete[] x0;
     return x1;
   }
     
     for (int i=0; i<NN; ++i)
   x0[i] = x1[i];
   }
 cout << "Fail to converge in " << N_max << " iterations" << endl;
 delete[] x0;
 return x1;
}

double* SOR_aabbcc(double aa, double bb, double cc, double *b, double *x_old, double *x_new,
                  int NN, double w, double eps, int N_max)
{
   double S;

   for (int k=0; k<N_max+1; ++k)
   {
       for (int i=0; i<NN; ++i)
       {
           if (i==0)
               S = cc * x_new[1];
           else if (i==NN-1)
               S = aa * x_new[NN-2];
           else
               S = aa * x_new[i-1] + cc * x_new[i+1];

           x_new[i] = (1-w)*x_new[i] + (w / bb) * (b[i] - S);
       }
       if (dist_square2(x_old, x_new, NN) < eps*eps)
       {
           return x_new;
       }

       for (int i=0; i<NN; ++i)
           x_old[i] = x_new[i];
   }
   cout << "Fail to converge in " << N_max << " iterations" << endl;
   return x_new;
}

void print_matrix(double arr[N][N])
{
for (int i=0; i<N; ++i)
   {
     for (int j=0; j<N; ++j)
   cout << arr[i][j] << "\t";
     cout << endl;
   }
}