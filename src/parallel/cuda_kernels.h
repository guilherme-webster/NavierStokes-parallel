#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// CUDA kernels for SOR pressure solver
__global__ void SORKernel(double* p, double* res, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
__global__ void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy);

// Host wrapper functions
int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, double** res, double** RHS, double omega, double eps, int max_it);

#endif // CUDA_KERNELS_H