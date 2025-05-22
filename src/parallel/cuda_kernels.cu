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
    
    // Define shared memory - with halo regions for stencil operations
    __shared__ double s_u[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    __shared__ double s_v[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices for shared memory (with offset for halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    
    // Load center values into shared memory
    int s_idx = tx * s_width + ty;
    int g_idx = i * (j_max + 2) + j;
    
    // Initialize shared memory with zeros for safety
    s_u[s_idx] = 0.0;
    s_v[s_idx] = 0.0;
    
    // Load valid data points
    if (i <= i_max+1 && j <= j_max+1) {
        s_u[s_idx] = u[g_idx];
        s_v[s_idx] = v[g_idx];
    }
    
    // Load halo regions - left and right columns
    if (threadIdx.x == 0 && i > 1) {
        s_u[(tx-1) * s_width + ty] = u[(i-1) * (j_max + 2) + j];
        s_v[(tx-1) * s_width + ty] = v[(i-1) * (j_max + 2) + j];
    }
    if (threadIdx.x == blockDim.x-1 && i <= i_max) {
        s_u[(tx+1) * s_width + ty] = u[(i+1) * (j_max + 2) + j];
        s_v[(tx+1) * s_width + ty] = v[(i+1) * (j_max + 2) + j];
    }
    
    // Load halo regions - top and bottom rows
    if (threadIdx.y == 0 && j > 1) {
        s_u[tx * s_width + (ty-1)] = u[i * (j_max + 2) + (j-1)];
        s_v[tx * s_width + (ty-1)] = v[i * (j_max + 2) + (j-1)];
    }
    if (threadIdx.y == blockDim.y-1 && j <= j_max) {
        s_u[tx * s_width + (ty+1)] = u[i * (j_max + 2) + (j+1)];
        s_v[tx * s_width + (ty+1)] = v[i * (j_max + 2) + (j+1)];
    }
    
    __syncthreads();
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        // Calculate F term using shared memory
        // Original condition: if (i < i_max && j > 0)
        // Corrected: F should be calculated for all i in [1, i_max]
        // The outer if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) already covers this.
        // No inner 'if' needed if F is cell-centered and defined everywhere u is.
        // However, to match the update domain of u, F is needed for u(i,j) where i goes to i_max-1.
        // If u is updated up to i_max, F must be computed up to i_max.
        // Let's assume F, G, u, v are all cell-centered and should be computed for all interior cells.

        // Calculate F term
        double u_central_for_F = s_u[tx * s_width + ty];
        double v_average_for_F = (s_v[tx * s_width + ty] + s_v[(tx+1) * s_width + ty] + 
                                  s_v[tx * s_width + (ty-1)] + s_v[(tx+1) * s_width + (ty-1)]) * 0.25;
        
        double du2_dx_for_F = ((s_u[tx * s_width + ty] + s_u[(tx+1) * s_width + ty]) * (s_u[tx * s_width + ty] + s_u[(tx+1) * s_width + ty]) -
                               (s_u[(tx-1) * s_width + ty] + s_u[tx * s_width + ty]) * (s_u[(tx-1) * s_width + ty] + s_u[tx * s_width + ty]))
                              / (4.0 * delta_x);
        
        double duv_dy_for_F = ((s_v[tx * s_width + ty] + s_v[(tx+1) * s_width + ty]) * (s_u[tx * s_width + ty] + s_u[tx * s_width + (ty+1)]) -
                               (s_v[tx * s_width + (ty-1)] + s_v[(tx+1) * s_width + (ty-1)]) * (s_u[tx * s_width + (ty-1)] + s_u[tx * s_width + ty]))
                              / (4.0 * delta_y);
        
        double d2u_dx2_for_F = (s_u[(tx+1) * s_width + ty] - 2.0 * s_u[tx * s_width + ty] + s_u[(tx-1) * s_width + ty])
                               / (delta_x * delta_x);
        
        double d2u_dy2_for_F = (s_u[tx * s_width + (ty+1)] - 2.0 * s_u[tx * s_width + ty] + s_u[tx * s_width + (ty-1)])
                               / (delta_y * delta_y);
        
        F[g_idx] = u_central_for_F + delta_t * (
                                     1.0/Re * (d2u_dx2_for_F + d2u_dy2_for_F) -
                                     du2_dx_for_F - duv_dy_for_F + g_x);
    
        // Calculate G term
        double v_central_for_G = s_v[tx * s_width + ty];
        double u_average_for_G = (s_u[tx * s_width + ty] + s_u[tx * s_width + (ty+1)] + 
                                  s_u[(tx-1) * s_width + ty] + s_u[(tx-1) * s_width + (ty+1)]) * 0.25;
        
        double duv_dx_for_G = ((s_u[tx * s_width + ty] + s_u[tx * s_width + (ty+1)]) * (s_v[tx * s_width + ty] + s_v[(tx+1) * s_width + ty]) -
                               (s_u[(tx-1) * s_width + ty] + s_u[(tx-1) * s_width + (ty+1)]) * (s_v[(tx-1) * s_width + ty] + s_v[tx * s_width + ty]))
                              / (4.0 * delta_x);
        
        double dv2_dy_for_G = ((s_v[tx * s_width + ty] + s_v[tx * s_width + (ty+1)]) * (s_v[tx * s_width + ty] + s_v[tx * s_width + (ty+1)]) -
                               (s_v[tx * s_width + (ty-1)] + s_v[tx * s_width + ty]) * (s_v[tx * s_width + (ty-1)] + s_v[tx * s_width + ty]))
                              / (4.0 * delta_y);
        
        double d2v_dx2_for_G = (s_v[(tx+1) * s_width + ty] - 2.0 * s_v[tx * s_width + ty] + s_v[(tx-1) * s_width + ty])
                               / (delta_x * delta_x);
        
        double d2v_dy2_for_G = (s_v[tx * s_width + (ty+1)] - 2.0 * s_v[tx * s_width + ty] + s_v[tx * s_width + (ty-1)])
                               / (delta_y * delta_y);
        
        G[g_idx] = v_central_for_G + delta_t * (
                                     1.0/Re * (d2v_dx2_for_G + d2v_dy2_for_G) -
                                     duv_dx_for_G - dv2_dy_for_G + g_y);
    }
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
    __shared__ double s_p[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    __shared__ double s_RHS[BLOCK_SIZE * BLOCK_SIZE];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    int s_idx = tx * s_width + ty;
    int g_idx = i * (j_max + 2) + j;
    int rhs_idx = threadIdx.x * blockDim.x + threadIdx.y;
    
    // Load center values
    if (i <= i_max+1 && j <= j_max+1) {
        s_p[s_idx] = p[g_idx];
        if (i <= i_max && j <= j_max && i > 0 && j > 0) {
            s_RHS[rhs_idx] = RHS[g_idx];
        }
    }
    
    // Load halo regions
    if (threadIdx.x == 0 && i > 1) {
        s_p[(tx-1) * s_width + ty] = p[(i-1) * (j_max + 2) + j];
    }
    if (threadIdx.x == blockDim.x-1 && i <= i_max) {
        s_p[(tx+1) * s_width + ty] = p[(i+1) * (j_max + 2) + j];
    }
    if (threadIdx.y == 0 && j > 1) {
        s_p[tx * s_width + (ty-1)] = p[i * (j_max + 2) + (j-1)];
    }
    if (threadIdx.y == blockDim.y-1 && j <= j_max) {
        s_p[tx * s_width + (ty+1)] = p[i * (j_max + 2) + (j+1)];
    }
    
    __syncthreads();
    
    // Update red points (i+j is even)
    if (i <= i_max && j <= j_max && i > 0 && j > 0 && (i + j) % 2 == 0) {
        s_p[s_idx] = (1.0 - omega) * s_p[s_idx] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((s_p[(tx+1) * s_width + ty] + s_p[(tx-1) * s_width + ty]) / dxdx + 
            (s_p[tx * s_width + (ty+1)] + s_p[tx * s_width + (ty-1)]) / dydy - 
            s_RHS[rhs_idx]);
            
        // Write back to global memory
        p[g_idx] = s_p[s_idx];
    }
}

__global__ void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    __shared__ double s_p[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    __shared__ double s_RHS[BLOCK_SIZE * BLOCK_SIZE];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    int s_idx = tx * s_width + ty;
    int g_idx = i * (j_max + 2) + j;
    int rhs_idx = threadIdx.x * blockDim.x + threadIdx.y;
    
    // Load center values
    if (i <= i_max+1 && j <= j_max+1) {
        s_p[s_idx] = p[g_idx];
        if (i <= i_max && j <= j_max && i > 0 && j > 0) {
            s_RHS[rhs_idx] = RHS[g_idx];
        }
    }
    
    // Load halo regions
    if (threadIdx.x == 0 && i > 1) {
        s_p[(tx-1) * s_width + ty] = p[(i-1) * (j_max + 2) + j];
    }
    if (threadIdx.x == blockDim.x-1 && i <= i_max) {
        s_p[(tx+1) * s_width + ty] = p[(i+1) * (j_max + 2) + j];
    }
    if (threadIdx.y == 0 && j > 1) {
        s_p[tx * s_width + (ty-1)] = p[i * (j_max + 2) + (j-1)];
    }
    if (threadIdx.y == blockDim.y-1 && j <= j_max) {
        s_p[tx * s_width + (ty+1)] = p[i * (j_max + 2) + (j+1)];
    }
    
    __syncthreads();
    
    // Update black points (i+j is odd)
    if (i <= i_max && j <= j_max && i > 0 && j > 0 && (i + j) % 2 == 1) {
        s_p[s_idx] = (1.0 - omega) * s_p[s_idx] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((s_p[(tx+1) * s_width + ty] + s_p[(tx-1) * s_width + ty]) / dxdx + 
            (s_p[tx * s_width + (ty+1)] + s_p[tx * s_width + (ty-1)]) / dydy - 
            s_RHS[rhs_idx]);
            
        // Write back to global memory
        p[g_idx] = s_p[s_idx];
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
    __shared__ double s_p[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    __shared__ double s_RHS[BLOCK_SIZE * BLOCK_SIZE];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    int s_idx = tx * s_width + ty;
    int g_idx = i * (j_max + 2) + j;
    int rhs_idx = threadIdx.x * blockDim.x + threadIdx.y;
    
    // Load center values and RHS
    if (i <= i_max+1 && j <= j_max+1) {
        s_p[s_idx] = p[g_idx];
        if (i <= i_max && j <= j_max && i > 0 && j > 0) {
            s_RHS[rhs_idx] = RHS[g_idx];
        }
    }
    
    // Load halo regions
    if (threadIdx.x == 0 && i > 1) {
        s_p[(tx-1) * s_width + ty] = p[(i-1) * (j_max + 2) + j];
    }
    if (threadIdx.x == blockDim.x-1 && i <= i_max) {
        s_p[(tx+1) * s_width + ty] = p[(i+1) * (j_max + 2) + j];
    }
    if (threadIdx.y == 0 && j > 1) {
        s_p[tx * s_width + (ty-1)] = p[i * (j_max + 2) + (j-1)];
    }
    if (threadIdx.y == blockDim.y-1 && j <= j_max) {
        s_p[tx * s_width + (ty+1)] = p[i * (j_max + 2) + (j+1)];
    }
    
    __syncthreads();
    
    // Calculate residual using shared memory
    if (i <= i_max && j <= j_max && i > 0 && j > 0) {
        res[g_idx] = (s_p[(tx+1) * s_width + ty] - 2.0 * s_p[s_idx] + s_p[(tx-1) * s_width + ty]) / dxdx + 
                    (s_p[tx * s_width + (ty+1)] - 2.0 * s_p[s_idx] + s_p[tx * s_width + (ty-1)]) / dydy - 
                    s_RHS[rhs_idx];
    }
}

__global__ void UpdateVelocityKernel(double* u, double* v, double* F, double* G, double* p,
                                    int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        u[i * (j_max + 2) + j] = F[i * (j_max + 2) + j] - delta_t * (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_x;
    }
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        v[i * (j_max + 2) + j] = G[i * (j_max + 2) + j] - delta_t * (p[i * (j_max + 2) + (j+1)] - p[i * (j_max + 2) + j]) / delta_y;
    }
}

__global__ void MultiStepSORKernel(double* p, double* RHS, double* res, 
                                   int i_max, int j_max, double omega, 
                                   double dxdx, double dydy, int iterations) {
    __shared__ double s_p[(BLOCK_SIZE+2) * (BLOCK_SIZE+2)];
    __shared__ double s_RHS[BLOCK_SIZE * BLOCK_SIZE];
    
    // Global indices
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Local indices for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int s_width = blockDim.x + 2;
    int s_idx = tx * s_width + ty;
    int g_idx = i * (j_max + 2) + j;
    int rhs_idx = threadIdx.x * blockDim.x + threadIdx.y;
    
    // Check bounds once
    bool valid = (i <= i_max && j <= j_max && i > 0 && j > 0);
    bool valid_halo_left = (threadIdx.x == 0 && i > 1);
    bool valid_halo_right = (threadIdx.x == blockDim.x-1 && i <= i_max); //< to <=
    bool valid_halo_bottom = (threadIdx.y == 0 && j > 1);
    bool valid_halo_top = (threadIdx.y == blockDim.y-1 && j <= j_max); // < to <=
    bool is_red = ((i + j) % 2 == 0);
    bool is_black = ((i + j) % 2 == 1);
    
    // Initial load of RHS (doesn't change during iterations)
    if (valid) {
        s_RHS[rhs_idx] = RHS[g_idx];
    }
    
    // Perform multiple iterations
    for (int iter = 0; iter < iterations; iter++) {
        // Load center values
        if (i <= i_max+1 && j <= j_max+1) {
            s_p[s_idx] = p[g_idx];
        }
        
        // Load halo regions
        if (valid_halo_left) {
            s_p[(tx-1) * s_width + ty] = p[(i-1) * (j_max + 2) + j];
        }
        if (valid_halo_right) {
            s_p[(tx+1) * s_width + ty] = p[(i+1) * (j_max + 2) + j];
        }
        if (valid_halo_bottom) {
            s_p[tx * s_width + (ty-1)] = p[i * (j_max + 2) + (j-1)];
        }
        if (valid_halo_top) {
            s_p[tx * s_width + (ty+1)] = p[i * (j_max + 2) + (j+1)];
        }
        
        __syncthreads();
        
        // Update red points
        if (valid && is_red) {
            s_p[s_idx] = (1.0 - omega) * s_p[s_idx] + 
                omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
                ((s_p[(tx+1) * s_width + ty] + s_p[(tx-1) * s_width + ty]) / dxdx + 
                (s_p[tx * s_width + (ty+1)] + s_p[tx * s_width + (ty-1)]) / dydy - 
                s_RHS[rhs_idx]);
                
            // Write back to global memory
            p[g_idx] = s_p[s_idx];
        }
        
        __syncthreads();
        
        // Update boundary conditions between red and black updates
        if (i >= 1 && i <= i_max) {
            if (j == 0) p[i * (j_max + 2) + j] = p[i * (j_max + 2) + (j+1)]; // Bottom
            if (j == j_max+1) p[i * (j_max + 2) + j] = p[i * (j_max + 2) + (j-1)]; // Top
        }
        if (j >= 1 && j <= j_max) {
            if (i == 0) p[i * (j_max + 2) + j] = p[(i+1) * (j_max + 2) + j]; // Left
            if (i == i_max+1) p[i * (j_max + 2) + j] = p[(i-1) * (j_max + 2) + j]; // Right
        }
        
        __syncthreads();
        
        // Reload shared memory after red points and boundary updates
        if (i <= i_max+1 && j <= j_max+1) {
            s_p[s_idx] = p[g_idx];
        }
        
        // Reload halo regions
        if (valid_halo_left) {
            s_p[(tx-1) * s_width + ty] = p[(i-1) * (j_max + 2) + j];
        }
        if (valid_halo_right) {
            s_p[(tx+1) * s_width + ty] = p[(i+1) * (j_max + 2) + j];
        }
        if (valid_halo_bottom) {
            s_p[tx * s_width + (ty-1)] = p[i * (j_max + 2) + (j-1)];
        }
        if (valid_halo_top) {
            s_p[tx * s_width + (ty+1)] = p[i * (j_max + 2) + (j+1)];
        }
        
        __syncthreads();
        
        // Update black points
        if (valid && is_black) {
            s_p[s_idx] = (1.0 - omega) * s_p[s_idx] + 
                omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
                ((s_p[(tx+1) * s_width + ty] + s_p[(tx-1) * s_width + ty]) / dxdx + 
                (s_p[tx * s_width + (ty+1)] + s_p[tx * s_width + (ty-1)]) / dydy - 
                s_RHS[rhs_idx]);
                
            // Write back to global memory
            p[g_idx] = s_p[s_idx];
        }
        
        __syncthreads();
        
        // Update boundary conditions after black update
        if (i >= 1 && i <= i_max) {
            if (j == 0) p[i * (j_max + 2) + j] = p[i * (j_max + 2) + (j+1)]; // Bottom
            if (j == j_max+1) p[i * (j_max + 2) + j] = p[i * (j_max + 2) + (j-1)]; // Top
        }
        if (j >= 1 && j <= j_max) {
            if (i == 0) p[i * (j_max + 2) + j] = p[(i+1) * (j_max + 2) + j]; // Left
            if (i == i_max+1) p[i * (j_max + 2) + j] = p[(i-1) * (j_max + 2) + j]; // Right
        }
        
        __syncthreads();
    }
    
    // Calculate residual if needed
    if (res != NULL && valid) {
        double residual = (p[(i+1) * (j_max + 2) + j] - 2.0 * p[g_idx] + p[(i-1) * (j_max + 2) + j]) / dxdx + 
                          (p[i * (j_max + 2) + (j+1)] - 2.0 * p[g_idx] + p[i * (j_max + 2) + (j-1)]) / dydy - 
                          RHS[g_idx];
        res[g_idx] = residual;
    }
}

// Add boundary conditions for F and G
__global__ void setBoundaryFGKernel(double* F, double* G, int i_max, int j_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // F boundary conditions
    if (j >= 1 && j <= j_max && i == 0) {
        F[i * (j_max + 2) + j] = 0.0; // Left
    }
    if (j >= 1 && j <= j_max && i == i_max) {
        F[i * (j_max + 2) + j] = 0.0; // Right
    }
    
    // G boundary conditions
    if (i >= 1 && i <= i_max && j == 0) {
        G[i * (j_max + 2) + j] = 0.0; // Bottom
    }
    if (i >= 1 && i <= i_max && j == j_max) {
        G[i * (j_max + 2) + j] = 0.0; // Top
    }
}