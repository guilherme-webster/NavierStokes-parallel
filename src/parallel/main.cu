/**
 * @file main.cu - CUDA Managed Memory Version
 * @author Hollweck, Wigg
 * @date 10 April 2019
 * @brief Main file with managed memory for Navier-Stokes simulation.
 */

#include "memory.h"
#include "io.h"
#include "vector"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    int i;
    int j;
    int side;
} BoundaryPoint;

// Managed memory pointers - accessible from both host and device
double *d_F, *d_G;
double *d_RHS, *d_res;
double *d_u, *d_v, *d_p;

// Simulation parameters - managed memory
double *d_delta_t, *d_delta_x, *d_delta_y, *d_gamma;
double *du_max, *dv_max;
int *d_i_max, *d_j_max;
double *d_tau, *d_Re;
BoundaryPoint *d_boundary_indices;
double *d_gx, *d_gy;
double *d_omega, *d_epsilon;
int *d_max_it;
double *d_norm_p, *d_norm_res;

void init_memory_managed(int i_max, int j_max, BoundaryPoint* h_boundary_indices, int total_points, 
                        double tau, double Re, double g_x, double g_y, double omega, double epsilon, int max_it);
void free_memory_managed();
double orchestration_managed(int i_max, int j_max);

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while(0)

#define KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while(0)

// Device functions for finite differences
__device__ double du2_dx_managed(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_i_plus = (i+1) * (j_max + 2) + j;
    int idx_i_minus = (i-1) * (j_max + 2) + j;
    
    double stencil1 = 0.5 * (u[idx] + u[idx_i_plus]);
    double stencil2 = 0.5 * (u[idx_i_minus] + u[idx]);
    double stencil3 = fabs(stencil1) * 0.5 * (u[idx] - u[idx_i_plus]);
    double stencil4 = fabs(stencil2) * 0.5 * (u[idx_i_minus] - u[idx]);
    
    return (1.0/delta_x) * (stencil1*stencil1 - stencil2*stencil2) + (gamma/delta_x) * (stencil3 - stencil4);
}

__device__ double duv_dy_managed(double* u, double* v, int i, int j, double delta_y, double gamma, int j_max) {
    int u_idx = i * (j_max + 2) + j;
    int u_idx_j_plus = i * (j_max + 2) + (j+1);
    int u_idx_j_minus = i * (j_max + 2) + (j-1);
    
    int v_idx = i * (j_max + 2) + j;
    int v_idx_i_plus = (i+1) * (j_max + 2) + j;
    int v_idx_j_minus = i * (j_max + 2) + (j-1);
    int v_idx_i_plus_j_minus = (i+1) * (j_max + 2) + (j-1);

    double stencil1 = 0.5 * (v[v_idx] + v[v_idx_i_plus]);
    double stencil2 = 0.5 * (v[v_idx_j_minus] + v[v_idx_i_plus_j_minus]);
    double stencil3 = stencil1 * 0.5 * (u[u_idx] + u[u_idx_j_plus]);
    double stencil4 = stencil2 * 0.5 * (u[u_idx_j_minus] + u[u_idx]);
    double stencil5 = fabs(stencil1) * 0.5 * (u[u_idx] - u[u_idx_j_plus]);
    double stencil6 = fabs(stencil2) * 0.5 * (u[u_idx_j_minus] - u[u_idx]);

    return (1.0/delta_y) * (stencil3 - stencil4) + (gamma/delta_y) * (stencil5 - stencil6);
}

__device__ double dv2_dy_managed(double* v, double* u, int i, int j, double delta_y, double gamma, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_j_plus = i * (j_max + 2) + (j+1);
    int idx_j_minus = i * (j_max + 2) + (j-1);
    
    double stencil1 = 0.5 * (v[idx] + v[idx_j_plus]);
    double stencil2 = 0.5 * (v[idx_j_minus] + v[idx]);
    double stencil3 = fabs(stencil1) * 0.5 * (v[idx] - v[idx_j_plus]);
    double stencil4 = fabs(stencil2) * 0.5 * (v[idx_j_minus] - v[idx]);

    return (1.0/delta_y) * (stencil1*stencil1 - stencil2*stencil2) + (gamma/delta_y) * (stencil3 - stencil4);
}

__device__ double duv_dx_managed(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max) {
    int u_idx = i * (j_max + 2) + j;
    int u_idx_j_plus = i * (j_max + 2) + (j+1);
    int u_idx_i_minus = (i-1) * (j_max + 2) + j;
    int u_idx_i_minus_j_plus = (i-1) * (j_max + 2) + (j+1);
    
    int v_idx = i * (j_max + 2) + j;
    int v_idx_i_plus = (i+1) * (j_max + 2) + j;
    int v_idx_i_minus = (i-1) * (j_max + 2) + j;

    double stencil1 = 0.5 * (u[u_idx] + u[u_idx_j_plus]);
    double stencil2 = 0.5 * (u[u_idx_i_minus] + u[u_idx_i_minus_j_plus]);
    double stencil3 = stencil1 * 0.5 * (v[v_idx] + v[v_idx_i_plus]);
    double stencil4 = stencil2 * 0.5 * (v[v_idx_i_minus] + v[v_idx]);
    double stencil5 = fabs(stencil1) * 0.5 * (v[v_idx] - v[v_idx_i_plus]);
    double stencil6 = fabs(stencil2) * 0.5 * (v[v_idx_i_minus] - v[v_idx]);

    return (1.0/delta_x) * (stencil3 - stencil4) + (gamma/delta_x) * (stencil5 - stencil6);
}

__device__ double d2u_dx2_managed(double* u, int i, int j, double delta_x, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_i_plus = (i+1) * (j_max + 2) + j;
    int idx_i_minus = (i-1) * (j_max + 2) + j;
    return (u[idx_i_plus] - 2.0 * u[idx] + u[idx_i_minus]) / (delta_x*delta_x);
}

__device__ double d2u_dy2_managed(double* u, int i, int j, double delta_y, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_j_plus = i * (j_max + 2) + (j+1);
    int idx_j_minus = i * (j_max + 2) + (j-1);
    return (u[idx_j_plus] - 2.0 * u[idx] + u[idx_j_minus]) / (delta_y*delta_y);
}

__device__ double d2v_dx2_managed(double* v, int i, int j, double delta_x, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_i_plus = (i+1) * (j_max + 2) + j;
    int idx_i_minus = (i-1) * (j_max + 2) + j;
    return (v[idx_i_plus] - 2.0 * v[idx] + v[idx_i_minus]) / (delta_x*delta_x);
}

__device__ double d2v_dy2_managed(double* v, int i, int j, double delta_y, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_j_plus = i * (j_max + 2) + (j+1);
    int idx_j_minus = i * (j_max + 2) + (j-1);
    return (v[idx_j_plus] - 2.0 * v[idx] + v[idx_j_minus]) / (delta_y*delta_y);
}

// Managed memory kernels
__global__ void calculate_F_managed(double* F, double* u, double* v, int i_max, int j_max, double Re,
    double g_x, double delta_t, double delta_x, double delta_y, double gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= i_max * j_max) return;
    
    int i = (idx / j_max) + 1;
    int j = (idx % j_max) + 1;
    
    // Bounds check for F calculation (only for i=1 to i_max-1)
    if (i >= 1 && i <= i_max-1 && j >= 1 && j <= j_max) {
        int f_idx = i * (j_max + 2) + j;
        F[f_idx] = u[f_idx] + delta_t * (
            (1.0/Re) * (d2u_dx2_managed(u, i, j, delta_x, j_max) + d2u_dy2_managed(u, i, j, delta_y, j_max)) 
            - du2_dx_managed(u, v, i, j, delta_x, gamma, j_max) 
            - duv_dy_managed(u, v, i, j, delta_y, gamma, j_max) 
            + g_x
        );
    }
}

__global__ void calculate_G_managed(double* G, double* u, double* v, int i_max, int j_max, double Re,
    double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= i_max * j_max) return;
    
    int i = (idx / j_max) + 1;
    int j = (idx % j_max) + 1;
    
    // Bounds check for G calculation (only for j=1 to j_max-1)
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max-1) {
        int g_idx = i * (j_max + 2) + j;
        G[g_idx] = v[g_idx] + delta_t * (
            (1.0/Re) * (d2v_dx2_managed(v, i, j, delta_x, j_max) + d2v_dy2_managed(v, i, j, delta_y, j_max)) 
            - duv_dx_managed(u, v, i, j, delta_x, gamma, j_max) 
            - dv2_dy_managed(v, u, i, j, delta_y, gamma, j_max) 
            + g_y
        );
    }
}

__global__ void setBoundaryFG_managed(double* F, double* G, int i_max, int j_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * (i_max + j_max)) return;
    
    if (idx < i_max) {
        // Top boundary F
        int i = idx + 1;
        F[i * (j_max + 2) + (j_max + 1)] = F[i * (j_max + 2) + j_max];
        // Bottom boundary F
        F[i * (j_max + 2) + 0] = F[i * (j_max + 2) + 1];
    } else if (idx < 2 * i_max) {
        // Left and right boundaries for G
        int i = (idx - i_max) + 1;
        if (i <= i_max) {
            G[0 * (j_max + 2) + i] = G[1 * (j_max + 2) + i];
            G[(i_max + 1) * (j_max + 2) + i] = G[i_max * (j_max + 2) + i];
        }
    }
}

__global__ void calculate_RHS_managed(double* RHS, double* F, double* G, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= i_max * j_max) return;
    
    int i = (idx / j_max) + 1;
    int j = (idx % j_max) + 1;
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        int rhs_idx = i * (j_max + 2) + j;
        RHS[rhs_idx] = (1.0 / delta_t) * (
            (F[rhs_idx] - F[(i-1) * (j_max + 2) + j]) / delta_x + 
            (G[rhs_idx] - G[i * (j_max + 2) + (j-1)]) / delta_y
        );
    }
}

__global__ void update_boundaries_managed(double* u, double* v, int i_max, int j_max, double lid_velocity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * (i_max + j_max)) return;
    
    if (idx < i_max) {
        // Top boundary (lid-driven)
        int i = idx + 1;
        v[i * (j_max + 2) + j_max] = 0.0;
        u[i * (j_max + 2) + (j_max + 1)] = 2.0 * lid_velocity - u[i * (j_max + 2) + j_max];
        
        // Bottom boundary (no-slip)
        v[i * (j_max + 2) + 0] = 0.0;
        u[i * (j_max + 2) + 0] = -u[i * (j_max + 2) + 1];
    } else if (idx < 2 * i_max) {
        // Left and right boundaries
        int j = (idx - i_max) + 1;
        if (j <= j_max) {
            // Left boundary
            u[0 * (j_max + 2) + j] = 0.0;
            v[0 * (j_max + 2) + j] = -v[1 * (j_max + 2) + j];
            
            // Right boundary
            u[(i_max + 1) * (j_max + 2) + j] = 0.0;
            v[(i_max + 1) * (j_max + 2) + j] = -v[i_max * (j_max + 2) + j];
        }
    }
}

__global__ void SOR_managed(double* p, double* RHS, int i_max, int j_max,
    double delta_x, double delta_y, double omega) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= i_max * j_max) return;
    
    int i = (idx / j_max) + 1;
    int j = (idx % j_max) + 1;
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        double dxdx = delta_x * delta_x;
        double dydy = delta_y * delta_y;
        int p_idx = i * (j_max + 2) + j;
        
        double p_new = (1.0 - omega) * p[p_idx] +
            omega / (2.0 * (1.0/dxdx + 1.0/dydy)) * (
                (p[(i+1) * (j_max + 2) + j] + p[(i-1) * (j_max + 2) + j]) / dxdx + 
                (p[i * (j_max + 2) + (j+1)] + p[i * (j_max + 2) + (j-1)]) / dydy - 
                RHS[p_idx]
            );
        
        p[p_idx] = p_new;
    }
}

__global__ void calculate_residual_managed(double* res, double* p, double* RHS, int i_max, int j_max,
    double delta_x, double delta_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= i_max * j_max) return;
    
    int i = (idx / j_max) + 1;
    int j = (idx % j_max) + 1;
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        int res_idx = i * (j_max + 2) + j;
        res[res_idx] = ((p[(i+1) * (j_max + 2) + j] - 2.0 * p[res_idx] + p[(i-1) * (j_max + 2) + j]) / (delta_x * delta_x) +
            (p[i * (j_max + 2) + (j+1)] - 2.0 * p[res_idx] + p[i * (j_max + 2) + (j-1)]) / (delta_y * delta_y)) - RHS[res_idx];
    }
}

__global__ void update_velocity_managed(double* u, double* v, double* F, double* G, double* p, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= i_max * j_max) return;
    
    int i = (idx / j_max) + 1;
    int j = (idx % j_max) + 1;
    
    if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
        // Update u velocity (for i=1 to i_max-1)
        if (i <= i_max - 1) {
            int u_idx = i * (j_max + 2) + j;
            u[u_idx] = F[u_idx] - delta_t * (p[(i+1) * (j_max + 2) + j] - p[u_idx]) / delta_x;
        }
        
        // Update v velocity (for j=1 to j_max-1)
        if (j <= j_max - 1) {
            int v_idx = i * (j_max + 2) + j;
            v[v_idx] = G[v_idx] - delta_t * (p[i * (j_max + 2) + (j+1)] - p[v_idx]) / delta_y;
        }
    }
}

__global__ void calculate_max_values(double* u, double* v, double* u_max, double* v_max, int i_max, int j_max) {
    extern __shared__ double sdata[];
    double* s_umax = sdata;
    double* s_vmax = &sdata[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_umax = 0.0;
    double local_vmax = 0.0;
    
    if (idx < i_max * j_max) {
        int i = (idx / j_max) + 1;
        int j = (idx % j_max) + 1;
        local_umax = fabs(u[i * (j_max + 2) + j]);
        local_vmax = fabs(v[i * (j_max + 2) + j]);
    }
    
    s_umax[tid] = local_umax;
    s_vmax[tid] = local_vmax;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_umax[tid] = fmax(s_umax[tid], s_umax[tid + s]);
            s_vmax[tid] = fmax(s_vmax[tid], s_vmax[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax((unsigned long long*)u_max, __double_as_longlong(s_umax[0]));
        atomicMax((unsigned long long*)v_max, __double_as_longlong(s_vmax[0]));
    }
}

__global__ void calculate_norm_managed(double* arr, double* norm, int i_max, int j_max) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_sum = 0.0;
    if (idx < i_max * j_max) {
        int i = (idx / j_max) + 1;
        int j = (idx % j_max) + 1;
        double val = arr[i * (j_max + 2) + j];
        local_sum = val * val;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(norm, sdata[0]);
    }
}

void init_memory_managed(int i_max, int j_max, BoundaryPoint* h_boundary_indices, int total_points, 
                        double tau, double Re, double g_x, double g_y, double omega, double epsilon, int max_it) {
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    // Allocate managed memory for arrays
    CUDA_CHECK(cudaMallocManaged((void**)&d_u, size));
    CUDA_CHECK(cudaMallocManaged((void**)&d_v, size));
    CUDA_CHECK(cudaMallocManaged((void**)&d_p, size));
    CUDA_CHECK(cudaMallocManaged((void**)&d_F, size));
    CUDA_CHECK(cudaMallocManaged((void**)&d_G, size));
    CUDA_CHECK(cudaMallocManaged((void**)&d_res, size));
    CUDA_CHECK(cudaMallocManaged((void**)&d_RHS, size));
    
    // Initialize arrays to zero
    CUDA_CHECK(cudaMemset(d_u, 0, size));
    CUDA_CHECK(cudaMemset(d_v, 0, size));
    CUDA_CHECK(cudaMemset(d_p, 0, size));
    CUDA_CHECK(cudaMemset(d_F, 0, size));
    CUDA_CHECK(cudaMemset(d_G, 0, size));
    CUDA_CHECK(cudaMemset(d_res, 0, size));
    CUDA_CHECK(cudaMemset(d_RHS, 0, size));
    
    // Allocate managed memory for parameters
    CUDA_CHECK(cudaMallocManaged((void**)&d_delta_t, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_delta_x, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_delta_y, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_gamma, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&du_max, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&dv_max, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_i_max, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_j_max, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_tau, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_Re, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_gx, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_gy, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_omega, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_epsilon, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_max_it, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_norm_p, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_norm_res, sizeof(double)));
    CUDA_CHECK(cudaMallocManaged((void**)&d_boundary_indices, total_points * sizeof(BoundaryPoint)));
    
    // Initialize parameters
    *d_i_max = i_max;
    *d_j_max = j_max;
    *d_tau = tau;
    *d_Re = Re;
    *d_gx = g_x;
    *d_gy = g_y;
    *d_omega = omega;
    *d_epsilon = epsilon;
    *d_max_it = max_it;
    *du_max = 0.0;
    *dv_max = 0.0;
    *d_norm_p = 0.0;
    *d_norm_res = 0.0;
    
    // Copy boundary indices
    for (int i = 0; i < total_points; i++) {
        d_boundary_indices[i] = h_boundary_indices[i];
    }
    
    printf("Managed memory initialized for grid size %d x %d\n", i_max, j_max);
}

void free_memory_managed() {
    if (d_u) {
        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaFree(d_F));
        CUDA_CHECK(cudaFree(d_G));
        CUDA_CHECK(cudaFree(d_res));
        CUDA_CHECK(cudaFree(d_RHS));
        CUDA_CHECK(cudaFree(d_delta_t));
        CUDA_CHECK(cudaFree(d_delta_x));
        CUDA_CHECK(cudaFree(d_delta_y));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(du_max));
        CUDA_CHECK(cudaFree(dv_max));
        CUDA_CHECK(cudaFree(d_i_max));
        CUDA_CHECK(cudaFree(d_j_max));
        CUDA_CHECK(cudaFree(d_tau));
        CUDA_CHECK(cudaFree(d_Re));
        CUDA_CHECK(cudaFree(d_gx));
        CUDA_CHECK(cudaFree(d_gy));
        CUDA_CHECK(cudaFree(d_omega));
        CUDA_CHECK(cudaFree(d_epsilon));
        CUDA_CHECK(cudaFree(d_max_it));
        CUDA_CHECK(cudaFree(d_norm_p));
        CUDA_CHECK(cudaFree(d_norm_res));
        CUDA_CHECK(cudaFree(d_boundary_indices));
        printf("Managed memory freed\n");
    }
}

double orchestration_managed(int i_max, int j_max) {
    int threads = 256;
    int blocks = (i_max * j_max + threads - 1) / threads;
    int shared_mem_size = 2 * threads * sizeof(double);
    
    // Calculate maximum values
    *du_max = 0.0;
    *dv_max = 0.0;
    calculate_max_values<<<blocks, threads, shared_mem_size>>>(d_u, d_v, du_max, dv_max, i_max, j_max);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate time step
    double u_max_val = *du_max;
    double v_max_val = *dv_max;
    double min_val = fmin(*d_Re / 2.0 / (1.0 / (*d_delta_x * *d_delta_x) + 1.0 / (*d_delta_y * *d_delta_y)), 
                         *d_delta_x / fabs(u_max_val));
    min_val = fmin(min_val, *d_delta_y / fabs(v_max_val));
    min_val = fmin(min_val, 3.0);
    *d_delta_t = *d_tau * min_val;
    *d_gamma = fmax(u_max_val * *d_delta_t / *d_delta_x, v_max_val * *d_delta_t / *d_delta_y);
    
    printf("Time step: dt=%f, gamma=%f, u_max=%f, v_max=%f\n", *d_delta_t, *d_gamma, u_max_val, v_max_val);
    
    // Set boundary conditions
    update_boundaries_managed<<<blocks, threads>>>(d_u, d_v, i_max, j_max, 1.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate F and G
    calculate_F_managed<<<blocks, threads>>>(d_F, d_u, d_v, i_max, j_max, *d_Re, *d_gx, *d_delta_t, *d_delta_x, *d_delta_y, *d_gamma);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    calculate_G_managed<<<blocks, threads>>>(d_G, d_u, d_v, i_max, j_max, *d_Re, *d_gy, *d_delta_t, *d_delta_x, *d_delta_y, *d_gamma);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Set F and G boundary conditions
    setBoundaryFG_managed<<<blocks, threads>>>(d_F, d_G, i_max, j_max);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate RHS
    calculate_RHS_managed<<<blocks, threads>>>(d_RHS, d_F, d_G, i_max, j_max, *d_delta_t, *d_delta_x, *d_delta_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Calculate initial pressure norm
    *d_norm_p = 0.0;
    calculate_norm_managed<<<blocks, threads, threads * sizeof(double)>>>(d_p, d_norm_p, i_max, j_max);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double norm_p = sqrt(*d_norm_p / (i_max * j_max));
    double convergence_threshold = *d_epsilon * (norm_p + 0.01);
    
    // SOR iteration loop
    int it = 0;
    double residual_norm = 1.0;
    
    while (it < *d_max_it && residual_norm > convergence_threshold) {
        // Apply SOR iteration
        SOR_managed<<<blocks, threads>>>(d_p, d_RHS, i_max, j_max, *d_delta_x, *d_delta_y, *d_omega);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check convergence every 10 iterations
        if (it % 10 == 0) {
            calculate_residual_managed<<<blocks, threads>>>(d_res, d_p, d_RHS, i_max, j_max, *d_delta_x, *d_delta_y);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            *d_norm_res = 0.0;
            calculate_norm_managed<<<blocks, threads, threads * sizeof(double)>>>(d_res, d_norm_res, i_max, j_max);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            residual_norm = sqrt(*d_norm_res / (i_max * j_max));
            
            if (it % 100 == 0) {
                printf("SOR it %d: residual=%.8e\n", it, residual_norm);
            }
        }
        it++;
    }
    
    printf("SOR completed after %d iterations. Final residual: %.8e\n", it, residual_norm);
    
    // Update velocities
    update_velocity_managed<<<blocks, threads>>>(d_u, d_v, d_F, d_G, d_p, i_max, j_max, *d_delta_t, *d_delta_x, *d_delta_y);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Print center values
    int center_idx = (i_max/2) * (j_max + 2) + (j_max/2);
    printf("U-CENTER: %.6f\n", d_u[center_idx]);
    printf("V-CENTER: %.6f\n", d_v[center_idx]);
    printf("P-CENTER: %.6f\n", d_p[center_idx]);
    
    return *d_delta_t;
}

BoundaryPoint* generate_boundary_indices(int i_max, int j_max, int* total_points) {
    *total_points = 2 * (i_max + j_max);
    BoundaryPoint* h_boundary_indices = (BoundaryPoint*)malloc(*total_points * sizeof(BoundaryPoint));
    int idx = 0;

    // Top boundary (j = j_max)
    for (int i = 1; i <= i_max; i++) {
        h_boundary_indices[idx++] = (BoundaryPoint){i, j_max, 0};
    }

    // Bottom boundary (j = 0)
    for (int i = 1; i <= i_max; i++) {
        h_boundary_indices[idx++] = (BoundaryPoint){i, 0, 1};
    }

    // Left boundary (i = 0)
    for (int j = 1; j <= j_max; j++) {
        h_boundary_indices[idx++] = (BoundaryPoint){0, j, 2};
    }

    // Right boundary (i = i_max + 1)
    for (int j = 1; j <= j_max; j++) {
        h_boundary_indices[idx++] = (BoundaryPoint){i_max + 1, j, 3};
    }

    return h_boundary_indices;
}

int main(int argc, char* argv[]) {
    // Grid pointers
    double** u, **v, **p, **F, **G, **res, **RHS;

    // Simulation parameters
    int i_max, j_max;
    double a, b, Re, delta_x, delta_y, gamma, T, g_x, g_y, tau, omega, epsilon;
    int max_it, n_print, problem;
    double f;

    const char* param_file = "parameters.txt";
    
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "CUDA: Could not open param_file\n");
        } else {
            param_file = argv[1];
            fclose(fp);
        }
    }
    
    // Initialize parameters
    init(&problem, &f, &i_max, &j_max, &a, &b, &Re, &T, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);
    printf("Initialized!\n");

    // Set step size in space
    delta_x = a / i_max;
    delta_y = b / j_max;

    // Allocate host memory for compatibility
    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);
    printf("Memory allocated.\n");

    // Initialize managed memory
    int total_points;
    BoundaryPoint* boundary_points = generate_boundary_indices(i_max, j_max, &total_points);
    init_memory_managed(i_max, j_max, boundary_points, total_points, tau, Re, g_x, g_y, omega, epsilon, max_it);
    
    // Set grid spacing
    *d_delta_x = delta_x;
    *d_delta_y = delta_y;

    // Time loop
    double t = 0;
    int n = 0;
    clock_t start = clock();
    
    printf("Starting 1000 timesteps simulation...\n");
    
    while (t < T && n < 1000) {
        if (n % 100 == 0) {
            printf("Step %d of 1000 (%.1f%%)\n", n, 100.0 * n / 1000.0);
        }
        
        double dt = orchestration_managed(i_max, j_max);
        t += dt;
        n++;
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n==================== FINAL RESULTS ====================\n");
    printf("Simulation completed: %d steps in %f seconds\n", n, time_spent);
    printf("Final time reached: %f\n", t);
    
    int center_idx = (i_max/2) * (j_max + 2) + (j_max/2);
    printf("U-CENTER: %.6f\n", d_u[center_idx]);
    printf("V-CENTER: %.6f\n", d_v[center_idx]);
    printf("P-CENTER: %.6f\n", d_p[center_idx]);
    printf("Average time per step: %f seconds\n", time_spent / n);
    printf("========================================================\n");

    fprintf(stderr, "%.6f", time_spent);

    // Clean up
    free_memory_managed();
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    free(boundary_points);
    
    return 0;
}