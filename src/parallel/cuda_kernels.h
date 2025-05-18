#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "integration.h"

// CUDA kernel for the SOR method
__global__ void SORKernel(double** p, double** res, double** RHS, double omega, double delta_x, double delta_y, int i_max, int j_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    if (i <= i_max && j <= j_max) {
        double dydy = delta_y * delta_y;
        double dxdx = delta_x * delta_x;

        p[i][j] = (1.0 - omega) * p[i][j] + omega / (2.0 * (1.0 / dxdx + 1.0 / dydy))
                    * ((p[i + 1][j] + p[i - 1][j]) / dxdx + (p[i][j + 1] + p[i][j - 1]) / dydy - RHS[i][j]);
    }
}

// CUDA kernel for calculating the residual
__global__ void ResidualKernel(double** p, double** res, double** RHS, double delta_x, double delta_y, int i_max, int j_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    if (i <= i_max && j <= j_max) {
        double dydy = delta_y * delta_y;
        double dxdx = delta_x * delta_x;

        res[i][j] = (p[i + 1][j] - 2.0 * p[i][j] + p[i - 1][j]) / dxdx + (p[i][j + 1] - 2.0 * p[i][j] + p[i][j - 1]) / dydy - RHS[i][j];
    }
}

// Function to launch the SOR kernel
void launchSORKernel(double** d_p, double** d_res, double** d_RHS, double omega, double delta_x, double delta_y, int i_max, int j_max) {
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + 15) / 16, (j_max + 15) / 16);
    SORKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, omega, delta_x, delta_y, i_max, j_max);
}

// Function to launch the residual kernel
void launchResidualKernel(double** d_p, double** d_res, double** d_RHS, double delta_x, double delta_y, int i_max, int j_max) {
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + 15) / 16, (j_max + 15) / 16);
    ResidualKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, delta_x, delta_y, i_max, j_max);
}

#endif // CUDA_KERNELS_H