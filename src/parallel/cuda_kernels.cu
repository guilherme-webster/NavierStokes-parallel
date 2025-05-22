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

// Kernel to calculate F values (intermediate velocity in x-direction)
__global__ void CalculateFKernel(double* u, double* v, double* F, 
                                int i_max, int j_max, double Re, double g_x,
                                double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // F is calculated for internal points only
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        int idx = i * (j_max + 2) + j;
        
        // Convection terms: du²/dx and duv/dy
        double du2_dx = 0.0, duv_dy = 0.0;
        double d2u_dx2 = 0.0, d2u_dy2 = 0.0;
        
        // du²/dx = ∂(u²)/∂x
        if (i > 1 && i < i_max) {
            double u_east = 0.5 * (u[idx] + u[(i+1)*(j_max+2) + j]);
            double u_west = 0.5 * (u[idx] + u[(i-1)*(j_max+2) + j]);
            du2_dx = (u_east * u_east - u_west * u_west) / delta_x;
        }
        
        // duv/dy = ∂(uv)/∂y  
        if (j > 1 && j < j_max) {
            double uv_north = 0.25 * (u[idx] + u[i*(j_max+2) + (j+1)]) * 
                                    (v[idx] + v[(i+1)*(j_max+2) + j]);
            double uv_south = 0.25 * (u[idx] + u[i*(j_max+2) + (j-1)]) * 
                                    (v[i*(j_max+2) + (j-1)] + v[(i+1)*(j_max+2) + (j-1)]);
            duv_dy = (uv_north - uv_south) / delta_y;
        }
        
        // Diffusion terms: ∂²u/∂x² and ∂²u/∂y²
        if (i > 1 && i < i_max) {
            d2u_dx2 = (u[(i+1)*(j_max+2) + j] - 2.0*u[idx] + u[(i-1)*(j_max+2) + j]) / (delta_x * delta_x);
        }
        if (j > 1 && j < j_max) {
            d2u_dy2 = (u[i*(j_max+2) + (j+1)] - 2.0*u[idx] + u[i*(j_max+2) + (j-1)]) / (delta_y * delta_y);
        }
        
        // Calculate F = u + dt * (viscous - convective + gravity)
        F[idx] = u[idx] + delta_t * (
            (1.0/Re) * (d2u_dx2 + d2u_dy2) - du2_dx - duv_dy + g_x
        );
    }
}

// Kernel to calculate G values (intermediate velocity in y-direction)
__global__ void CalculateGKernel(double* u, double* v, double* G, 
                                int i_max, int j_max, double Re, double g_y,
                                double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // G is calculated for internal points only
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        int idx = i * (j_max + 2) + j;
        
        // Convection terms: duv/dx and dv²/dy
        double duv_dx = 0.0, dv2_dy = 0.0;
        double d2v_dx2 = 0.0, d2v_dy2 = 0.0;
        
        // duv/dx = ∂(uv)/∂x
        if (i > 1 && i < i_max) {
            double uv_east = 0.25 * (u[idx] + u[i*(j_max+2) + (j+1)]) * 
                                   (v[idx] + v[(i+1)*(j_max+2) + j]);
            double uv_west = 0.25 * (u[(i-1)*(j_max+2) + j] + u[(i-1)*(j_max+2) + (j+1)]) * 
                                   (v[idx] + v[(i-1)*(j_max+2) + j]);
            duv_dx = (uv_east - uv_west) / delta_x;
        }
        
        // dv²/dy = ∂(v²)/∂y
        if (j > 1 && j < j_max) {
            double v_north = 0.5 * (v[idx] + v[i*(j_max+2) + (j+1)]);
            double v_south = 0.5 * (v[idx] + v[i*(j_max+2) + (j-1)]);
            dv2_dy = (v_north * v_north - v_south * v_south) / delta_y;
        }
        
        // Diffusion terms: ∂²v/∂x² and ∂²v/∂y²
        if (i > 1 && i < i_max) {
            d2v_dx2 = (v[(i+1)*(j_max+2) + j] - 2.0*v[idx] + v[(i-1)*(j_max+2) + j]) / (delta_x * delta_x);
        }
        if (j > 1 && j < j_max) {
            d2v_dy2 = (v[i*(j_max+2) + (j+1)] - 2.0*v[idx] + v[i*(j_max+2) + (j-1)]) / (delta_y * delta_y);
        }
        
        // Calculate G = v + dt * (viscous - convective + gravity)
        G[idx] = v[idx] + delta_t * (
            (1.0/Re) * (d2v_dx2 + d2v_dy2) - duv_dx - dv2_dy + g_y
        );
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