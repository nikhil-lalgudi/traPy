#ifndef SOLV_HPP
#define SOLBE_HPP

#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cstdlib>

static const int N = 4;

void print_matrix(double arr[N][N]);
void print_array(double *arr);
double* SOR(double A[N][N], double* b, double w, double eps, int N_max);
double* SOR_trid(double A[N][N], double* b, double w, double eps, int N_max);
double* SOR_abc(double aa, double bb, double cc, double* b, int NN,
 double w, double eps, int N_max);

double* SOR_aabbcc(double aa, double bb, double cc, double* b, double* x_old, double* x_new, 
int NN, double w, double eps, int N_max);
double dist_square(double *arr1, double *arr2);
double dist_square2(double* arr1, double* arr2, int NN);
void print_array_2(double* arr, int NN);


#endif