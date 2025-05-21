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
#define BLOCK_SIZE 16
#define OVERLAP 2  // Overlap size between subdomains

// Integrated kernel that handles an entire Navier-Stokes timestep
__global__ void NavierStokesStepKernel(
    double* u, double* v, double* p, double* F, double* G, double* RHS, double* res,
    int i_max, int j_max, double Re, double g_x, double g_y,
    double delta_t, double delta_x, double delta_y, double gamma,
    double omega, double eps, int max_sor_iter) {
    
    // Each thread handles one grid point
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Skip boundary points and points outside the valid domain
    if (i > i_max || j > j_max) return;
    
    // Calculate F term
    if (i < i_max and j > 0) {
        double u_central = u[i * (j_max + 2) + j];
        double v_average = (v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j] + 
                           v[i * (j_max + 2) + (j-1)] + v[(i+1) * (j_max + 2) + (j-1)]) * 0.25;
        
        double du2_dx = ((u[i * (j_max + 2) + j] + u[(i+1) * (j_max + 2) + j])
                       * (u[i * (j_max + 2) + j] + u[(i+1) * (j_max + 2) + j]) -
                        (u[(i-1) * (j_max + 2) + j] + u[i * (j_max + 2) + j])
                       * (u[(i-1) * (j_max + 2) + j] + u[i * (j_max + 2) + j]))
                      / (4.0 * delta_x);
        
        double duv_dy = ((v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j])
                       * (u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) -
                        (v[i * (j_max + 2) + (j-1)] + v[(i+1) * (j_max + 2) + (j-1)])
                       * (u[i * (j_max + 2) + (j-1)] + u[i * (j_max + 2) + j]))
                      / (4.0 * delta_y);
        
        double d2u_dx2 = (u[(i+1) * (j_max + 2) + j] - 2.0 * u[i * (j_max + 2) + j] + u[(i-1) * (j_max + 2) + j])
                        / (delta_x * delta_x);
        
        double d2u_dy2 = (u[i * (j_max + 2) + (j+1)] - 2.0 * u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j-1)])
                        / (delta_y * delta_y);
        
        F[i * (j_max + 2) + j] = u_central + delta_t * (
                               1.0/Re * (d2u_dx2 + d2u_dy2) -
                               du2_dx - duv_dy + g_x);
    }
    
    // Calculate G term
    if (j < j_max && i > 0) {
        double v_central = v[i * (j_max + 2) + j];
        double u_average = (u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)] + 
                           u[(i-1) * (j_max + 2) + j] + u[(i-1) * (j_max + 2) + (j+1)]) * 0.25;
        
        double duv_dx = ((u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)])
                       * (v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) -
                        (u[(i-1) * (j_max + 2) + j] + u[(i-1) * (j_max + 2) + (j+1)])
                       * (v[(i-1) * (j_max + 2) + j] + v[i * (j_max + 2) + j]))
                      / (4.0 * delta_x);
        
        double dv2_dy = ((v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j+1)])
                       * (v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j+1)]) -
                        (v[i * (j_max + 2) + (j-1)] + v[i * (j_max + 2) + j])
                       * (v[i * (j_max + 2) + (j-1)] + v[i * (j_max + 2) + j]))
                      / (4.0 * delta_y);
        
        double d2v_dx2 = (v[(i+1) * (j_max + 2) + j] - 2.0 * v[i * (j_max + 2) + j] + v[(i-1) * (j_max + 2) + j])
                        / (delta_x * delta_x);
        
        double d2v_dy2 = (v[i * (j_max + 2) + (j+1)] - 2.0 * v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j-1)])
                        / (delta_y * delta_y);
        
        G[i * (j_max + 2) + j] = v_central + delta_t * (
                               1.0/Re * (d2v_dx2 + d2v_dy2) -
                               duv_dx - dv2_dy + g_y);
    }
    
    // Global synchronization not available, use separate kernels for RHS and SOR
}

__global__ void CalculateRHSKernel(double* F, double* G, double* RHS, 
                                 int i_max, int j_max, 
                                 double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max && i > 0 && j > 0) {
        RHS[i * (j_max + 2) + j] = 1.0 / delta_t * (
            (F[i * (j_max + 2) + j] - F[(i-1) * (j_max + 2) + j]) / delta_x + 
            (G[i * (j_max + 2) + j] - G[i * (j_max + 2) + (j-1)]) / delta_y
        );
    }
}

// Standard SOR kernels - using the Red-Black approach
__global__ void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= i_max && j <= j_max && i > 0 && j > 0 && (i + j) % 2 == 0) {
        p[i * (j_max + 2) + j] = (1.0 - omega) * p[i * (j_max + 2) + j] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((p[(i + 1) * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j]);
    }
}

__global__ void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= i_max && j <= j_max && i > 0 && j > 0 && (i + j) % 2 == 1) {
        p[i * (j_max + 2) + j] = (1.0 - omega) * p[i * (j_max + 2) + j] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((p[(i + 1) * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j]);
    }
}

__global__ void UpdateBoundaryKernel(double* p, int i_max, int j_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bottom and top boundaries
    if (i >= 1 && i <= i_max && j == 0) {
        p[i * (j_max + 2) + j] = p[i * (j_max + 2) + j+1]; // Bottom
    }
    if (i >= 1 && i <= i_max && j == j_max+1) {
        p[i * (j_max + 2) + j] = p[i * (j_max + 2) + j-1]; // Top
    }
    
    // Left and right boundaries
    if (j >= 1 && j <= j_max && i == 0) {
        p[i * (j_max + 2) + j] = p[(i+1) * (j_max + 2) + j]; // Left
    }
    if (j >= 1 && j <= j_max && i == i_max+1) {
        p[i * (j_max + 2) + j] = p[(i-1) * (j_max + 2) + j]; // Right
    }
}

__global__ void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= i_max && j <= j_max && i > 0 && j > 0) {
        res[i * (j_max + 2) + j] = (p[(i + 1) * (j_max + 2) + j] - 2.0 * p[i * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] - 2.0 * p[i * (j_max + 2) + j] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j];
    }
}

__global__ void UpdateVelocityKernel(double* u, double* v, double* F, double* G, double* p,
                                    int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < i_max && j <= j_max && j > 0) {
        u[i * (j_max + 2) + j] = F[i * (j_max + 2) + j] - delta_t * (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_x;
    }
    
    if (i <= i_max && j < j_max) {
        v[i * (j_max + 2) + j] = G[i * (j_max + 2) + j] - delta_t * (p[i * (j_max + 2) + (j+1)] - p[i * (j_max + 2) + j]) / delta_y;
    }
}