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
    
    // Create flattened host arrays and speed variables
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
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error in SUBDOMAINSORKERNEL: %s\n", cudaGetErrorString(err));
                exit(-1);
}
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
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error in CALCULATERESIDUALKERNEL: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
            
            // Calculate residual norm
            CalculateResidualNormKernel<<<gridSize, blockSize>>>(d_res, i_max, j_max, d_norm);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error in CALCULATERESIDUALNORMKERNEL: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
            
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

__global__ void L2NormKernel(double* m, int i_max, int j_max, double* d_norm) {
    __shared__ double s_norm[BLOCK_SIZE * BLOCK_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    double local_norm = 0.0;
    if (i <= i_max && j <= j_max) {
        double val = m[i * (j_max + 2) + j];
        local_norm = val * val;
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

double cudaL2(double** m, int i_max, int j_max) {
    // Allocate device memory
    double *d_m, *d_norm;
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_m, size));
    CUDACHECK(cudaMalloc((void**)&d_norm, sizeof(double)));
    
    // Create flattened host array
    double *h_m = (double*)malloc(size);
    
    // Copy from 2D array to flattened array
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            h_m[i * (j_max + 2) + j] = m[i][j];
        }
    }
    
    // Copy data to device
    CUDACHECK(cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(d_norm, 0, sizeof(double)));
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    L2NormKernel<<<gridSize, blockSize>>>(d_m, i_max, j_max, d_norm);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in L2NORMKERNELn: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // Get result from device
    double norm_value = 0.0;
    CUDACHECK(cudaMemcpy(&norm_value, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Calculate final norm
    double l2_norm = sqrt(norm_value / (i_max * j_max));
    
    // Free memory
    CUDACHECK(cudaFree(d_m));
    CUDACHECK(cudaFree(d_norm));
    free(h_m);
    
    return l2_norm;
}

__global__ void FGKernel(double* F, double* G, double* u, double* v, 
                        int i_max, int j_max, double Re, double g_x, double g_y, 
                        double delta_t, double delta_x, double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i > i_max || j > j_max) return;
    
    double du2dx, duvdy, d2udx2, d2udy2;
    double duvdx, dv2dy, d2vdx2, d2vdy2;
    double alpha = 0.9;  // Donor cell parameter
    
    // F calculation for internal points
    if (i < i_max) {
        // Convection terms
        du2dx = ((u[i * (j_max + 2) + j] + u[(i+1) * (j_max + 2) + j]) * 
                (u[i * (j_max + 2) + j] + u[(i+1) * (j_max + 2) + j]) / 4.0 - 
                (u[(i-1) * (j_max + 2) + j] + u[i * (j_max + 2) + j]) * 
                (u[(i-1) * (j_max + 2) + j] + u[i * (j_max + 2) + j]) / 4.0) / delta_x;
        
        du2dx += alpha * ((fabs(u[i * (j_max + 2) + j] + u[(i+1) * (j_max + 2) + j]) * 
                         (u[i * (j_max + 2) + j] - u[(i+1) * (j_max + 2) + j]) / 4.0) - 
                        (fabs(u[(i-1) * (j_max + 2) + j] + u[i * (j_max + 2) + j]) * 
                         (u[(i-1) * (j_max + 2) + j] - u[i * (j_max + 2) + j]) / 4.0)) / delta_x;
        
        duvdy = ((v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) * 
                (u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) / 4.0 - 
                (v[i * (j_max + 2) + (j-1)] + v[(i+1) * (j_max + 2) + (j-1)]) * 
                (u[i * (j_max + 2) + (j-1)] + u[i * (j_max + 2) + j]) / 4.0) / delta_y;
        
        duvdy += alpha * ((fabs(v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) * 
                         (u[i * (j_max + 2) + j] - u[i * (j_max + 2) + (j+1)]) / 4.0) - 
                        (fabs(v[i * (j_max + 2) + (j-1)] + v[(i+1) * (j_max + 2) + (j-1)]) * 
                         (u[i * (j_max + 2) + (j-1)] - u[i * (j_max + 2) + j]) / 4.0)) / delta_y;
        
        // Diffusion terms
        d2udx2 = (u[(i+1) * (j_max + 2) + j] - 2.0 * u[i * (j_max + 2) + j] + 
                 u[(i-1) * (j_max + 2) + j]) / (delta_x * delta_x);
        
        d2udy2 = (u[i * (j_max + 2) + (j+1)] - 2.0 * u[i * (j_max + 2) + j] + 
                 u[i * (j_max + 2) + (j-1)]) / (delta_y * delta_y);
        
        F[i * (j_max + 2) + j] = u[i * (j_max + 2) + j] + 
                                delta_t * (1.0 / Re * (d2udx2 + d2udy2) - du2dx - duvdy + g_x);
    }
    
    // G calculation for internal points
    if (j < j_max) {
        // Convection terms
        duvdx = ((u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) * 
                (v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) / 4.0 - 
                (u[(i-1) * (j_max + 2) + j] + u[(i-1) * (j_max + 2) + (j+1)]) * 
                (v[(i-1) * (j_max + 2) + j] + v[i * (j_max + 2) + j]) / 4.0) / delta_x;
        
        duvdx += alpha * ((fabs(u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) * 
                         (v[i * (j_max + 2) + j] - v[(i+1) * (j_max + 2) + j]) / 4.0) - 
                        (fabs(u[(i-1) * (j_max + 2) + j] + u[(i-1) * (j_max + 2) + (j+1)]) * 
                         (v[(i-1) * (j_max + 2) + j] - v[i * (j_max + 2) + j]) / 4.0)) / delta_x;
        
        dv2dy = ((v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j+1)]) * 
                (v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j+1)]) / 4.0 - 
                (v[i * (j_max + 2) + (j-1)] + v[i * (j_max + 2) + j]) * 
                (v[i * (j_max + 2) + (j-1)] + v[i * (j_max + 2) + j]) / 4.0) / delta_y;
        
        dv2dy += alpha * ((fabs(v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j+1)]) * 
                         (v[i * (j_max + 2) + j] - v[i * (j_max + 2) + (j+1)]) / 4.0) - 
                        (fabs(v[i * (j_max + 2) + (j-1)] + v[i * (j_max + 2) + j]) * 
                         (v[i * (j_max + 2) + (j-1)] - v[i * (j_max + 2) + j]) / 4.0)) / delta_y;
        
        // Diffusion terms
        d2vdx2 = (v[(i+1) * (j_max + 2) + j] - 2.0 * v[i * (j_max + 2) + j] + 
                 v[(i-1) * (j_max + 2) + j]) / (delta_x * delta_x);
        
        d2vdy2 = (v[i * (j_max + 2) + (j+1)] - 2.0 * v[i * (j_max + 2) + j] + 
                 v[i * (j_max + 2) + (j-1)]) / (delta_y * delta_y);
        
        G[i * (j_max + 2) + j] = v[i * (j_max + 2) + j] + 
                                delta_t * (1.0 / Re * (d2vdx2 + d2vdy2) - duvdx - dv2dy + g_y);
    }
}

void cudaFG(double** F, double** G, double** u, double** v, int i_max, int j_max, 
           double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    // Allocate device memory
    double *d_F, *d_G, *d_u, *d_v;
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_F, size));
    CUDACHECK(cudaMalloc((void**)&d_G, size));
    CUDACHECK(cudaMalloc((void**)&d_u, size));
    CUDACHECK(cudaMalloc((void**)&d_v, size));
    
    // Create flattened host arrays
    double *h_F = (double*)malloc(size);
    double *h_G = (double*)malloc(size);
    double *h_u = (double*)malloc(size);
    double *h_v = (double*)malloc(size);
    
    // Copy from 2D arrays to flattened arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            h_F[i * (j_max + 2) + j] = F[i][j];
            h_G[i * (j_max + 2) + j] = G[i][j];
            h_u[i * (j_max + 2) + j] = u[i][j];
            h_v[i * (j_max + 2) + j] = v[i][j];
        }
    }
    
    // Copy data to device
    CUDACHECK(cudaMemcpy(d_F, h_F, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    FGKernel<<<gridSize, blockSize>>>(d_F, d_G, d_u, d_v, i_max, j_max, Re, g_x, g_y, 
                                    delta_t, delta_x, delta_y, gamma);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in FGKERNEl: %s\n", cudaGetErrorString(err));
        exit(-1);
}
    
    // Copy results back
    CUDACHECK(cudaMemcpy(h_F, d_F, size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_G, d_G, size, cudaMemcpyDeviceToHost));
    
    // Copy from flattened arrays back to 2D arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            F[i][j] = h_F[i * (j_max + 2) + j];
            G[i][j] = h_G[i * (j_max + 2) + j];
        }
    }
    
    // Free memory
    CUDACHECK(cudaFree(d_F));
    CUDACHECK(cudaFree(d_G));
    CUDACHECK(cudaFree(d_u));
    CUDACHECK(cudaFree(d_v));
    free(h_F);
    free(h_G);
    free(h_u);
    free(h_v);
}

// Add this after the FG implementation

__global__ void UpdateVelocityKernel(double* u, double* v, double* F, double* G, double* p, 
                                    int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i > i_max || j > j_max) return;
    
    // Update u velocity (using forward difference for dp/dx)
    if (i < i_max) {
        double dp_dx = (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_x;
        u[i * (j_max + 2) + j] = F[i * (j_max + 2) + j] - delta_t * dp_dx;
    }
    
    // Update v velocity (using forward difference for dp/dy)
    if (j < j_max) {
        double dp_dy = (p[i * (j_max + 2) + (j+1)] - p[i * (j_max + 2) + j]) / delta_y;
        v[i * (j_max + 2) + j] = G[i * (j_max + 2) + j] - delta_t * dp_dy;
    }
}

void cudaUpdateVelocity(double** u, double** v, double** F, double** G, double** p, 
                      int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    // Allocate device memory
    double *d_u, *d_v, *d_F, *d_G, *d_p;
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_u, size));
    CUDACHECK(cudaMalloc((void**)&d_v, size));
    CUDACHECK(cudaMalloc((void**)&d_F, size));
    CUDACHECK(cudaMalloc((void**)&d_G, size));
    CUDACHECK(cudaMalloc((void**)&d_p, size));
    
    // Create flattened host arrays
    double *h_u = (double*)malloc(size);
    double *h_v = (double*)malloc(size);
    double *h_F = (double*)malloc(size);
    double *h_G = (double*)malloc(size);
    double *h_p = (double*)malloc(size);
    
    // Copy from 2D arrays to flattened arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            h_u[i * (j_max + 2) + j] = u[i][j];
            h_v[i * (j_max + 2) + j] = v[i][j];
            h_F[i * (j_max + 2) + j] = F[i][j];
            h_G[i * (j_max + 2) + j] = G[i][j];
            h_p[i * (j_max + 2) + j] = p[i][j];
        }
    }
    
    // Copy data to device
    CUDACHECK(cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_F, h_F, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    UpdateVelocityKernel<<<gridSize, blockSize>>>(d_u, d_v, d_F, d_G, d_p, i_max, j_max, delta_t, delta_x, delta_y);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in UPDATEVELOCITYKERNEL: %s\n", cudaGetErrorString(err));
        exit(-1);
}
    
    // Copy results back
    CUDACHECK(cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost));
    
    // Copy from flattened arrays back to 2D arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            u[i][j] = h_u[i * (j_max + 2) + j];
            v[i][j] = h_v[i * (j_max + 2) + j];
        }
    }
    
    // Free memory
    CUDACHECK(cudaFree(d_u));
    CUDACHECK(cudaFree(d_v));
    CUDACHECK(cudaFree(d_F));
    CUDACHECK(cudaFree(d_G));
    CUDACHECK(cudaFree(d_p));
    free(h_u);
    free(h_v);
    free(h_F);
    free(h_G);
    free(h_p);
}

// Add this after the UpdateVelocityKernel implementation

__global__ void CalculateRHSKernel(double* F, double* G, double* RHS,
                                int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        RHS[i * (j_max + 2) + j] = 1.0 / delta_t * 
                                  ((F[i * (j_max + 2) + j] - F[(i-1) * (j_max + 2) + j]) / delta_x + 
                                   (G[i * (j_max + 2) + j] - G[i * (j_max + 2) + (j-1)]) / delta_y);
    }
}

void cudaCalculateRHS(double** F, double** G, double** RHS, 
                   int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    // Allocate device memory
    double *d_F, *d_G, *d_RHS;
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_F, size));
    CUDACHECK(cudaMalloc((void**)&d_G, size));
    CUDACHECK(cudaMalloc((void**)&d_RHS, size));
    
    // Create flattened host arrays
    double *h_F = (double*)malloc(size);
    double *h_G = (double*)malloc(size);
    double *h_RHS = (double*)malloc(size);
    
    // Copy from 2D arrays to flattened arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            h_F[i * (j_max + 2) + j] = F[i][j];
            h_G[i * (j_max + 2) + j] = G[i][j];
        }
    }
    
    // Copy data to device
    CUDACHECK(cudaMemcpy(d_F, h_F, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_G, h_G, size, cudaMemcpyHostToDevice));
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    CalculateRHSKernel<<<gridSize, blockSize>>>(d_F, d_G, d_RHS, i_max, j_max, delta_t, delta_x, delta_y);
   cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in CALCULATERHSKERNEL: %s\n", cudaGetErrorString(err));
        exit(-1);
}
    
    // Copy results back
    CUDACHECK(cudaMemcpy(h_RHS, d_RHS, size, cudaMemcpyDeviceToHost));
    
    // Copy from flattened array back to 2D array
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            RHS[i][j] = h_RHS[i * (j_max + 2) + j];
        }
    }
    
    // Free memory
    CUDACHECK(cudaFree(d_F));
    CUDACHECK(cudaFree(d_G));
    CUDACHECK(cudaFree(d_RHS));
    free(h_F);
    free(h_G);
    free(h_RHS);
}