#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include "integration.h"
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

// Variáveis globais para arrays unificados
double *unified_p = NULL;
double *unified_res = NULL;
double *unified_RHS = NULL;
size_t cuda_array_size = 0;
int grid_i_max = 0;
int grid_j_max = 0;

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
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // Only update black cells (i+j is odd)
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
        p[i * (j_max + 2) + j] = (1.0 - omega) * p[i * (j_max + 2) + j] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((p[(i + 1) * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j]);
    }
}

__global__ void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= i_max && j <= j_max) {
        res[i * (j_max + 2) + j] = (p[(i + 1) * (j_max + 2) + j] - 2.0 * p[i * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] - 2.0 * p[i * (j_max + 2) + j] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j];
    }
}

// Initialize CUDA arrays once
int initCudaArrays(int i_max, int j_max) {
    // Salvar dimensões da grade
    grid_i_max = i_max;
    grid_j_max = j_max;
    
    // Calcular tamanho necessário
    cuda_array_size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    // Alocar memória unificada
    CUDACHECK(cudaMallocManaged(&unified_p, cuda_array_size));
    CUDACHECK(cudaMallocManaged(&unified_res, cuda_array_size));
    CUDACHECK(cudaMallocManaged(&unified_RHS, cuda_array_size));
    
    // Inicializar com zeros
    CUDACHECK(cudaMemset(unified_p, 0, cuda_array_size));
    CUDACHECK(cudaMemset(unified_res, 0, cuda_array_size));
    CUDACHECK(cudaMemset(unified_RHS, 0, cuda_array_size));
    
    // Prefetch para GPU
    int device = -1;
    CUDACHECK(cudaGetDevice(&device));
    CUDACHECK(cudaMemPrefetchAsync(unified_p, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_RHS, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_res, cuda_array_size, device, NULL));
    
    return 0;
}

// Free CUDA arrays once at the end
void freeCudaArrays() {
    if (unified_p) CUDACHECK(cudaFree(unified_p));
    if (unified_res) CUDACHECK(cudaFree(unified_res));
    if (unified_RHS) CUDACHECK(cudaFree(unified_RHS));
    
    unified_p = NULL;
    unified_res = NULL;
    unified_RHS = NULL;
}

int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = L2(p, i_max, j_max);
    
    // Copiar dados dos arrays 2D para os arrays unificados
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            unified_p[i * (j_max + 2) + j] = p[i][j];
            unified_RHS[i * (j_max + 2) + j] = RHS[i][j];
            unified_res[i * (j_max + 2) + j] = 0.0;
        }
    }
    
    // Grid e block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Executar iterações Red-Black SOR
    while (it < max_it) {
        // Atualizar condições de contorno
        for (int i = 1; i <= i_max; i++) {
            unified_p[i * (j_max + 2) + 0] = unified_p[i * (j_max + 2) + 1];
            unified_p[i * (j_max + 2) + (j_max + 1)] = unified_p[i * (j_max + 2) + j_max];
        }
        for (int j = 1; j <= j_max; j++) {
            unified_p[0 * (j_max + 2) + j] = unified_p[1 * (j_max + 2) + j];
            unified_p[(i_max + 1) * (j_max + 2) + j] = unified_p[i_max * (j_max + 2) + j];
        }
        
        // Sincronizar antes de computação
        CUDACHECK(cudaDeviceSynchronize());
        
        // Red points
        RedSORKernel<<<gridSize, blockSize>>>(unified_p, unified_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Black points
        BlackSORKernel<<<gridSize, blockSize>>>(unified_p, unified_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Calcular resíduos
        CalculateResidualKernel<<<gridSize, blockSize>>>(unified_p, unified_res, unified_RHS, i_max, j_max, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Verificar convergência
        double res_norm = 0.0;
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                res_norm += unified_res[i * (j_max + 2) + j] * unified_res[i * (j_max + 2) + j];
            }
        }
        res_norm = sqrt(res_norm / (i_max * j_max));
        
        if (res_norm <= eps * (norm_p + 0.01)) {
            break; // Convergência atingida
        }
        
        it++;
    }
    
    // Copiar resultados de volta para os arrays 2D originais
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            p[i][j] = unified_p[i * (j_max + 2) + j];
        }
    }
    
    return (it < max_it) ? 0 : -1;
}