#include "cuda_kernels.h"
#include <cuda_runtime.h>

__global__ void SORKernel(double* p, double* res, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    if (i <= i_max && j <= j_max) {
        p[i * (j_max + 2) + j] = (1.0 - omega) * p[i * (j_max + 2) + j] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((p[(i + 1) * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j]);
    }
}

__global__ void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    if (i <= i_max && j <= j_max) {
        res[i * (j_max + 2) + j] = (p[(i + 1) * (j_max + 2) + j] - 2.0 * p[i * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] - 2.0 * p[i * (j_max + 2) + j] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j];
    }
}

void launchSORKernel(double* d_p, double* d_res, double* d_RHS, int i_max, int j_max, double omega, double delta_x, double delta_y) {
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + 1 + blockSize.x - 1) / blockSize.x, (j_max + 1 + blockSize.y - 1) / blockSize.y);

    double dxdx = delta_x * delta_x;
    double dydy = delta_y * delta_y;

    SORKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, i_max, j_max, omega, dxdx, dydy);
    cudaDeviceSynchronize();
}

void launchCalculateResidualKernel(double* d_p, double* d_res, double* d_RHS, int i_max, int j_max, double delta_x, double delta_y) {
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + 1 + blockSize.x - 1) / blockSize.x, (j_max + 1 + blockSize.y - 1) / blockSize.y);

    double dxdx = delta_x * delta_x;
    double dydy = delta_y * delta_y;

    CalculateResidualKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, i_max, j_max, dxdx, dydy);
    cudaDeviceSynchronize();
}