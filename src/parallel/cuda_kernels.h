#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, double** res, double** RHS, double omega, double eps, int max_it);
double cudaL2(double** m, int i_max, int j_max);
void cudaFG(double** F, double** G, double** u, double** v, int i_max, int j_max, 
           double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma);
void cudaUpdateVelocity(double** u, double** v, double** F, double** G, double** p, 
                      int i_max, int j_max, double delta_t, double delta_x, double delta_y);
void cudaCalculateRHS(double** F, double** G, double** RHS, 
                   int i_max, int j_max, double delta_t, double delta_x, double delta_y);
#endif // CUDA_KERNELS_H