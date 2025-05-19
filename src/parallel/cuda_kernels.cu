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

// 1. Shared memory optimized kernels
__global__ void RedSORKernelShared(double* p, double* p_out, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    // Shared memory for tile plus halo cells
    __shared__ double s_p[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices within shared memory (with offset for halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load center tile
    if (i <= i_max && j <= j_max) {
        s_p[ty][tx] = p[i * (j_max + 2) + j];
    }
    
    // Load halo cells (each thread loads its neighborhood)
    if (threadIdx.x == 0 && i > 1) { // Left halo
        s_p[ty][0] = p[(i-1) * (j_max + 2) + j];
    }
    if (threadIdx.x == BLOCK_SIZE-1 && i < i_max) { // Right halo
        s_p[ty][tx+1] = p[(i+1) * (j_max + 2) + j];
    }
    if (threadIdx.y == 0 && j > 1) { // Bottom halo
        s_p[0][tx] = p[i * (j_max + 2) + (j-1)];
    }
    if (threadIdx.y == BLOCK_SIZE-1 && j < j_max) { // Top halo
        s_p[ty+1][tx] = p[i * (j_max + 2) + (j+1)];
    }
    
    __syncthreads();  // Ensure all data is loaded
    
    // Red SOR update
    if (i <= i_max && j <= j_max && (i + j) % 2 == 0) {
        double new_p = (1.0 - omega) * s_p[ty][tx] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((s_p[ty][tx+1] + s_p[ty][tx-1]) / dxdx + 
             (s_p[ty+1][tx] + s_p[ty-1][tx]) / dydy - 
             RHS[i * (j_max + 2) + j]);
        
        // Write back to output buffer
        p_out[i * (j_max + 2) + j] = new_p;
    } else if (i <= i_max && j <= j_max) {
        // Just copy value for black points
        p_out[i * (j_max + 2) + j] = p[i * (j_max + 2) + j];
    }
}

__global__ void BlackSORKernelShared(double* p, double* p_out, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    // Shared memory for tile plus halo cells
    __shared__ double s_p[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices within shared memory (with offset for halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load center tile
    if (i <= i_max && j <= j_max) {
        s_p[ty][tx] = p[i * (j_max + 2) + j];
    }
    
    // Load halo cells (each thread loads its neighborhood)
    if (threadIdx.x == 0 && i > 1) { // Left halo
        s_p[ty][0] = p[(i-1) * (j_max + 2) + j];
    }
    if (threadIdx.x == BLOCK_SIZE-1 && i < i_max) { // Right halo
        s_p[ty][tx+1] = p[(i+1) * (j_max + 2) + j];
    }
    if (threadIdx.y == 0 && j > 1) { // Bottom halo
        s_p[0][tx] = p[i * (j_max + 2) + (j-1)];
    }
    if (threadIdx.y == BLOCK_SIZE-1 && j < j_max) { // Top halo
        s_p[ty+1][tx] = p[i * (j_max + 2) + (j+1)];
    }
    
    __syncthreads();  // Ensure all data is loaded
    
    // Black SOR update
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
        double new_p = (1.0 - omega) * s_p[ty][tx] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((s_p[ty][tx+1] + s_p[ty][tx-1]) / dxdx + 
             (s_p[ty+1][tx] + s_p[ty-1][tx]) / dydy - 
             RHS[i * (j_max + 2) + j]);
        
        // Write back to output buffer
        p_out[i * (j_max + 2) + j] = new_p;
    } else if (i <= i_max && j <= j_max) {
        // Just copy value for red points
        p_out[i * (j_max + 2) + j] = p[i * (j_max + 2) + j];
    }
}

// 2. Domain decomposition kernels
__global__ void UpdateBoundaryKernel(double* p, int i_max, int j_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Update bottom and top boundaries
    if (i <= i_max && j == 1) {
        p[i * (j_max + 2) + 0] = p[i * (j_max + 2) + 1]; // Bottom boundary
    }
    if (i <= i_max && j == j_max) {
        p[i * (j_max + 2) + (j_max + 1)] = p[i * (j_max + 2) + j_max]; // Top boundary
    }
    
    // Update left and right boundaries
    if (j <= j_max && i == 1) {
        p[0 * (j_max + 2) + j] = p[1 * (j_max + 2) + j]; // Left boundary
    }
    if (j <= j_max && i == i_max) {
        p[(i_max + 1) * (j_max + 2) + j] = p[i_max * (j_max + 2) + j]; // Right boundary
    }
}

__global__ void SubdomainSORKernel(double* p, double* p_out, double* RHS, 
                                  int i_max, int j_max, double omega, double dxdx, double dydy,
                                  int subdomain_idx, int num_subdomains_x, int num_subdomains_y,
                                  int iter_per_subdomain) {
    // Shared memory for subdomain
    __shared__ double s_p[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    
    // Calculate subdomain boundaries (assume equal-sized subdomains)
    int subdomain_width = (i_max + num_subdomains_x - 1) / num_subdomains_x;
    int subdomain_height = (j_max + num_subdomains_y - 1) / num_subdomains_y;
    
    // Get subdomain coordinates
    int subdomain_x = subdomain_idx % num_subdomains_x;
    int subdomain_y = subdomain_idx / num_subdomains_x;
    
    // Calculate subdomain start indices (including overlap)
    int i_start = max(1, subdomain_x * subdomain_width - OVERLAP);
    int j_start = max(1, subdomain_y * subdomain_height - OVERLAP);
    
    // Calculate subdomain end indices (including overlap)
    int i_end = min(i_max, (subdomain_x + 1) * subdomain_width + OVERLAP);
    int j_end = min(j_max, (subdomain_y + 1) * subdomain_height + OVERLAP);
    
    // Global and local indices
    int i = i_start + blockIdx.x * blockDim.x + threadIdx.x;
    int j = j_start + blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    
    // Don't overrun subdomain
    if (i > i_end || j > j_end) return;
    
    // Local SOR iterations within this subdomain
    double* src = p;
    double* dst = p_out;
    
    for (int iter = 0; iter < iter_per_subdomain; iter++) {
        // Load data into shared memory
        if (i <= i_end && j <= j_end) {
            s_p[ty * s_width + tx] = src[i * (j_max + 2) + j];
        }
        
        // Load halo cells correctly
        if (threadIdx.x == 0 && i > i_start) {
            s_p[ty * s_width + 0] = src[(i-1) * (j_max + 2) + j]; // Left halo
        }
        if (threadIdx.x == blockDim.x - 1 && i < i_end) {
            s_p[ty * s_width + (tx+1)] = src[(i+1) * (j_max + 2) + j]; // Right halo
        }
        if (threadIdx.y == 0 && j > j_start) {
            s_p[0 * s_width + tx] = src[i * (j_max + 2) + (j-1)]; // Bottom halo
        }
        if (threadIdx.y == blockDim.y - 1 && j < j_end) {
            s_p[(ty+1) * s_width + tx] = src[i * (j_max + 2) + (j+1)]; // Top halo
        }
        
        __syncthreads();
        
        // RED POINTS UPDATE
        if (i <= i_end && j <= j_end && (i + j) % 2 == 0) {
            double new_p = (1.0 - omega) * s_p[ty * s_width + tx] + 
                omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
                ((s_p[ty * s_width + (tx+1)] + s_p[ty * s_width + (tx-1)]) / dxdx + 
                (s_p[(ty+1) * s_width + tx] + s_p[(ty-1) * s_width + tx]) / dydy - 
                RHS[i * (j_max + 2) + j]);
                
            dst[i * (j_max + 2) + j] = new_p;
        } else if (i <= i_end && j <= j_end) {
            dst[i * (j_max + 2) + j] = src[i * (j_max + 2) + j];
        }
        
        __syncthreads();
        
        // We need to reload shared memory with the updated values
        if (i <= i_end && j <= j_end) {
            s_p[ty * s_width + tx] = dst[i * (j_max + 2) + j];
        }

        // Reload halo cells with updated values
        if (threadIdx.x == 0 && i > i_start) {
            s_p[ty * s_width + 0] = dst[(i-1) * (j_max + 2) + j]; // Left halo
        }
        if (threadIdx.x == blockDim.x - 1 && i < i_end) {
            s_p[ty * s_width + (tx+1)] = dst[(i+1) * (j_max + 2) + j]; // Right halo
        }
        if (threadIdx.y == 0 && j > j_start) {
            s_p[0 * s_width + tx] = dst[i * (j_max + 2) + (j-1)]; // Bottom halo
        }
        if (threadIdx.y == blockDim.y - 1 && j < j_end) {
            s_p[(ty+1) * s_width + tx] = dst[i * (j_max + 2) + (j+1)]; // Top halo
        }
        
        __syncthreads();
        
        // BLACK POINTS UPDATE
        if (i <= i_end && j <= j_end && (i + j) % 2 == 1) {
            double new_p = (1.0 - omega) * s_p[ty * s_width + tx] + 
                omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
                ((s_p[ty * s_width + (tx+1)] + s_p[ty * s_width + (tx-1)]) / dxdx + 
                (s_p[(ty+1) * s_width + tx] + s_p[(ty-1) * s_width + tx]) / dydy - 
                RHS[i * (j_max + 2) + j]);
                
            src[i * (j_max + 2) + j] = new_p;  // Write to src (already swapped)
        } else if (i <= i_end && j <= j_end) {
            src[i * (j_max + 2) + j] = dst[i * (j_max + 2) + j];
        }
        
        __syncthreads();
        
        // Swap pointers for next iteration
        double* temp = src;
        src = dst;
        dst = temp;
    }
    
    // Copy final result back to p_out if needed
    if (iter_per_subdomain % 2 == 1 && i <= i_end && j <= j_end) {
        p_out[i * (j_max + 2) + j] = src[i * (j_max + 2) + j];
    }
}

// Original kernels kept for reference (removed in final deployment)
__global__ void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

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

// Kernel to compute residual norm with reduction
__global__ void CalculateResidualNormKernel(double* res, int i_max, int j_max, double* d_norm) {
    __shared__ double s_norm[BLOCK_SIZE * BLOCK_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    double local_norm = 0.0;
    if (i <= i_max && j <= j_max) {
        double r = res[i * (j_max + 2) + j];
        local_norm = r * r;
    }
    
    s_norm[tid] = local_norm;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_norm[tid] += s_norm[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes result
    if (tid == 0) {
        atomicAdd(d_norm, s_norm[0]);
    }
}

int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = 0.0;
    
    printf("Starting optimized CUDA SOR solver with shared memory and domain decomposition...\n");
    
    // Allocate device memory for flattened arrays
    double *d_p, *d_p_tmp, *d_res, *d_RHS, *d_norm;
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_p, size));
    CUDACHECK(cudaMalloc((void**)&d_p_tmp, size)); // Second buffer for double buffering
    CUDACHECK(cudaMalloc((void**)&d_res, size));
    CUDACHECK(cudaMalloc((void**)&d_RHS, size));
    CUDACHECK(cudaMalloc((void**)&d_norm, sizeof(double)));
    
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
    CUDACHECK(cudaMemcpy(d_p_tmp, h_p, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_RHS, h_RHS, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_res, 0, size));
    
    // Calculate initial norm
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            norm_p += p[i][j] * p[i][j];
        }
    }
    norm_p = sqrt(norm_p / (i_max * j_max));
    double convergence_threshold = eps * (norm_p + 1e-10);
    
    printf("Initial norm: %e, convergence threshold: %e\n", norm_p, convergence_threshold);
    
    // Grid and block dimensions for regular kernels
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Domain decomposition parameters
    int num_subdomains_x = 3; // Number of horizontal subdomains
    int num_subdomains_y = 3; // Number of vertical subdomains
    int total_subdomains = num_subdomains_x * num_subdomains_y;
    int iter_per_subdomain = 5; // How many iterations per subdomain before synchronization
    
    // Run iterations
    double *device_ptr = d_p;
    double *buffer_ptr = d_p_tmp;
    double h_res_norm = 1.0;
    
    while (it < max_it && h_res_norm > convergence_threshold) {
        // Update boundary conditions directly on the device
        UpdateBoundaryKernel<<<gridSize, blockSize>>>(device_ptr, i_max, j_max);
        CUDACHECK(cudaGetLastError());
        
        // Process all subdomains (domain decomposition approach)
        for (int subdomain = 0; subdomain < total_subdomains; subdomain++) {
            // Calculate subdomain dimensions
            int subdomain_width = (i_max + num_subdomains_x - 1) / num_subdomains_x;
            int subdomain_height = (j_max + num_subdomains_y - 1) / num_subdomains_y;
            
            // Get subdomain coordinates
            int subdomain_x = subdomain % num_subdomains_x;
            int subdomain_y = subdomain / num_subdomains_x;
            
            // Calculate actual subdomain size including overlap
            int i_start = max(1, subdomain_x * subdomain_width - OVERLAP);
            int j_start = max(1, subdomain_y * subdomain_height - OVERLAP);
            int i_end = min(i_max, (subdomain_x + 1) * subdomain_width + OVERLAP);
            int j_end = min(j_max, (subdomain_y + 1) * subdomain_height + OVERLAP);
            
            // Subdomain grid dimensions
            int subdomain_blocks_x = (i_end - i_start + blockSize.x) / blockSize.x;
            int subdomain_blocks_y = (j_end - j_start + blockSize.y) / blockSize.y;
            dim3 subdomainGrid(subdomain_blocks_x, subdomain_blocks_y);
            
            // Process this subdomain with multiple iterations
            SubdomainSORKernel<<<subdomainGrid, blockSize>>>(
                device_ptr, buffer_ptr, d_RHS,
                i_max, j_max, omega, dxdx, dydy,
                subdomain, num_subdomains_x, num_subdomains_y,
                iter_per_subdomain
            );
            CUDACHECK(cudaGetLastError());
        }
        
        // Swap buffers
        double *temp = device_ptr;
        device_ptr = buffer_ptr;
        buffer_ptr = temp;
        
        // Check for convergence every 10 iterations
        if (it % 10 == 0) {
            CUDACHECK(cudaMemset(d_norm, 0, sizeof(double)));
            
            // Calculate residual
            CalculateResidualKernel<<<gridSize, blockSize>>>(device_ptr, d_res, d_RHS, i_max, j_max, dxdx, dydy);
            CUDACHECK(cudaGetLastError());
            
            // Calculate residual norm
            CalculateResidualNormKernel<<<gridSize, blockSize>>>(d_res, i_max, j_max, d_norm);
            CUDACHECK(cudaGetLastError());
            
            // Get norm from device
            double norm_value = 0.0;
            CUDACHECK(cudaMemcpy(&norm_value, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
            h_res_norm = sqrt(norm_value / (i_max * j_max));
            
            if (it % 100 == 0) {
                printf("Iteration %d: residual norm = %e\n", it, h_res_norm);
            }
            
            if (h_res_norm <= convergence_threshold) {
                printf("Converged after %d iterations. Final residual: %e\n", it, h_res_norm);
                break;
            }
        }
        
        it++;
    }
    
    // Copy final results back
    CUDACHECK(cudaMemcpy(h_p, device_ptr, size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost));
    
    // Copy from flattened arrays back to 2D arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            p[i][j] = h_p[i * (j_max + 2) + j];
            res[i][j] = h_res[i * (j_max + 2) + j];
        }
    }
    
    // Free memory
    CUDACHECK(cudaFree(d_p));
    CUDACHECK(cudaFree(d_p_tmp));
    CUDACHECK(cudaFree(d_res));
    CUDACHECK(cudaFree(d_RHS));
    CUDACHECK(cudaFree(d_norm));
    free(h_p);
    free(h_res);
    free(h_RHS);
    
    return (it < max_it) ? 0 : -1;
}