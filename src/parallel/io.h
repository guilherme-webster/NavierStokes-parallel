// This header file declares the functions for input and output operations.

#ifndef IO_H
#define IO_H

int init(int* problem, double* f, int* i_max, int* j_max, double* a, double* b, double* Re, double* T, double* g_x, double* g_y, double* tau, double* omega, double* epsilon, int* max_it, int* n_print, const char* param_file);
void output_results(int i_max, int j_max, double** u, double** v, double** p, double t, double a, double b, const char* out_prefix);

#endif // IO_H