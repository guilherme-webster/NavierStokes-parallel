#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Add CUDA error checking
void check_cuda(cudaError_t error, const char *filename, const int line)
{
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

__global__ void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    // Only update red cells (i+j is even)
    if (i <= i_max && j <= j_max && (i + j) % 2 == 0) {
        p[i * (j_max + 2) + j] = (1.0 - omega) * p[i * (j_max + 2) + j] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((p[(i + 1) * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j]);
    }
}

__global__ void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    // Only update black cells (i+j is odd)
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
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

int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = L2(p, i_max, j_max);
    
    // Allocate device memory for flattened arrays
    double *d_p, *d_res, *d_RHS;
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_p, size));
    CUDACHECK(cudaMalloc((void**)&d_res, size));
    CUDACHECK(cudaMalloc((void**)&d_RHS, size));
    
    // Create flattened host arrays
    double *h_p = (double*)malloc(size);
    double *h_res = (double*)malloc(size);
    double *h_RHS = (double*)malloc(size);
    
    // Copy from 2D arrays to flattened arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            h_p[i * (j_max + 2) + j] = p[i][j];
            h_RHS[i * (j_max + 2) + j] = RHS[i][j];
        }
    }
    
    // Copy data to device
    CUDACHECK(cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_RHS, h_RHS, size, cudaMemcpyHostToDevice));
    //CUDACHECK(cudaMemcpy(d_res, h_res, size, cudaMemcpyHostToDevice));
    // Initialize residual to zero
    CUDACHECK(cudaMemset(d_res, 0, size));
    
    // Grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Run Red-Black SOR iterations
    while (it < max_it) {
        // Update boundary conditions on device
        for (int i = 1; i <= i_max; i++) {
            h_p[i * (j_max + 2) + 0] = h_p[i * (j_max + 2) + 1];
            h_p[i * (j_max + 2) + (j_max + 1)] = h_p[i * (j_max + 2) + j_max];
        }
        for (int j = 1; j <= j_max; j++) {
            h_p[0 * (j_max + 2) + j] = h_p[1 * (j_max + 2) + j];
            h_p[(i_max + 1) * (j_max + 2) + j] = h_p[i_max * (j_max + 2) + j];
        }
        CUDACHECK(cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice));
        
        // Red points
        RedSORKernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Black points update
        BlackSORKernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Calculate residual and check convergence
        CalculateResidualKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, i_max, j_max, dxdx, dydy);
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Copy results back
        CUDACHECK(cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost));
        
        // Check for convergence
        double res_norm = 0.0;
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                res_norm += h_res[i * (j_max + 2) + j] * h_res[i * (j_max + 2) + j];
            }
        }
        res_norm = sqrt(res_norm / (i_max * j_max));
        
        if (res_norm <= eps * (norm_p + 1e-10)) {
            break; // Converged
        }
        
        it++;
    }
    
    // Copy final result back to 2D arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            p[i][j] = h_p[i * (j_max + 2) + j];
            res[i][j] = h_res[i * (j_max + 2) + j];
        }
    }
    
    // Free memory
    CUDACHECK(cudaFree(d_p));
    CUDACHECK(cudaFree(d_res));
    CUDACHECK(cudaFree(d_RHS));
    free(h_p);
    free(h_res);
    free(h_RHS);
    
    return (it < max_it) ? 0 : -1;
}

// implement two kernels: red and black, check L2 for early convergence 