#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

// Declare CUDA kernels
__global__ void NavierStokesStepKernel(
    double* u, double* v, double* p, 
    double* F, double* G, double* RHS, double* res,
    int i_max, int j_max, double Re, double g_x, double g_y,
    double delta_t, double delta_x, double delta_y, double gamma,
    double omega, double eps, int max_sor_iter);

__global__ void CalculateRHSKernel(double* F, double* G, double* RHS, 
                               int i_max, int j_max, 
                               double delta_t, double delta_x, double delta_y);

__global__ void RedSORKernel(double* p, double* RHS, int i_max, int j_max, 
                         double omega, double dxdx, double dydy);

__global__ void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, 
                          double omega, double dxdx, double dydy);

__global__ void UpdateBoundaryKernel(double* p, int i_max, int j_max);

__global__ void CalculateResidualKernel(double* p, double* res, double* RHS, 
                                    int i_max, int j_max, double dxdx, double dydy);

__global__ void UpdateVelocityKernel(double* u, double* v, double* F, double* G, double* p,
                                 int i_max, int j_max, double delta_t, double delta_x, double delta_y);

// Error checking helper
void check_cuda(cudaError_t error, const char *filename, const int line);
#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

#endif // CUDA_KERNELS_H