#include "kernel.h"
#include "vector"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef struct {
    int i, j;
    int side; // TOP=0, BOTTOM=1, LEFT=2, RIGHT=3
} BoundaryPoint;

double du_max,dv_max;
double* d_u, d_v, d_p;
double d_delta_t, d_delta_x, d_delta_y;
double d_tau, d_gamma,d_Re;
int* d_boundary_index;
BoundaryPoint* d_boundary_indices;
int di_max, dj_max;
double* d_F, d_G;
double* d_RHS;
double dg_x, dg_y;

void init_memory(int i_max, int j_max, double* delta_t, double delta_x, double delta_y, double Re, BoundaryPoint* h_boundary_indices) {
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    cudaMalloc((void**)&du_max * sizeof(double));
    cudaMalloc((void**)&dv_max * sizeof(double));
    cudaMalloc((void**) d_u * size);
    cudaMalloc((void**) d_v * size);
    cudaMalloc((void**) d_p * size);
    cudaMalloc((void*) d_Re * sizeof(double));
    cudaMalloc((void*) d_tau * sizeof(double));
    cudaMalloc((void*) d_gamma * sizeof(double));
    cudaMalloc((void*) d_delta_t * sizeof(double));
    cudaMalloc((void*) d_delta_x * sizeof(double));
    cudaMalloc((void*) d_delta_y * sizeof(double));
    cudaMalloc((void*) d_boundary_index * sizeof(int));
    cudaMalloc((void**)&d_boundary_indices, total_points * sizeof(BoundaryPoint));
    cudaMalloc((void*)Re * sizeof(double));
    cudaMalloc((void**)d_RHS * size);
    cudaMalloc((void**)d_F * size);
    cudaMalloc((void**)d_G * size);
    cudaMalloc((void*) d_gy * sizeof(double));
    cudaMalloc((void*) d_gx * sizeof(double));

    cudaMemcpy(d_boundary_indices, h_boundary_indices, total_points * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    cudaMemset(d_u,0, sizeof(double));
    cudaMemset(d_v,0, sizeof(double));
    cudaMemset(d_p,0, sizeof(double));
    cudaMemset(dv_max,0, sizeof(double));
    cudaMemset(du_max,0, sizeof(double));
    cudaMemset(d_Re,0, sizeof(double));
    cudaMemset(d_tau,0, sizeof(double));
    cudaMemset(d_gamma,0, sizeof(double));
    cudaMemset(d_delta_t,0, sizeof(double));
    cudaMemset(d_delta_x,0, sizeof(double));
    cudaMemset(d_delta_y,0, sizeof(double));
    cudaMemset(d_boundary_index,0, sizeof(int));
    cudaMemset(d_boundary_indices,0, sizeof(BoundaryPoint));
    cudaMemset(d_F,0, sizeof(double));
    cudaMemset(d_G,0, sizeof(double));
    cudaMemset(d_RHS,0, sizeof(double));
    cudaMemset(dg_x,0, sizeof(double));
    cudaMemset(dg_y,0, sizeof(double));
}



void orquestration(double** u, double** v, double** p, double** res, double** RHS, double** F, double** G,
    int i_max, int j_max, double* delta_t, double delta_x, double delta_y, double Re, double g_x, double g_y,
    double tau, double omega, double epsilon, int max_it, int n_print) {
    
    int threads = 256;
    int blocks = (i_max * j_max + threads - 1) / threads;
    int size = i_max * j_max;
    extern __shared__ double shared_data[];
    di_max = i_max;
    dj_max = j_max;
    // acha o máximo da matriz u e v
    while (size > 1){
        blocks = (size + threads - 1) / threads;

        max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(i_max, j_max, u, d_u);
        max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(i_max, j_max, v, d_v);
        cudaDeviceSynchronize();
    }
    
    min_and_gamma<<<1, 1>>>();
    
    cudaDeviceSynchronize();

    update_boundaries_kernel<<<blocks, threads>>>();

    cudaDeviceSynchronize();

    printf("Conditions set!\n");

    // now we calculate F and G
    calculate_F<<<blocks, threads>>>(d_F, d_u, d_v, di_max, dj_max, d_Re, dg_x, d_delta_t, d_delta_x, d_delta_y, d_gamma);
    calculate_G<<<blocks, threads>>>(d_G, d_u, d_v, di_max, dj_max, d_Re, dg_y, d_delta_t, d_delta_x, d_delta_y, d_gamma);    

    cudaDeviceSynchronize();

    printf("F, G calculated!\n");

    // now we calculate rhs
    calculate_RHS<<<blocks, threads>>>(RHS, F, G, d_u, d_v, di_max, dj_max, d_delta_t, d_delta_x, d_delta_y);

    cudaDeviceSynchronize();
    
    L2_norm<<<blocks, threads>>>(d_norm_p, d_p, di_max, dj_max);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(&norm_p, d_norm_p, sizeof(double), cudaMemcpyDeviceToHost);
    double norm = sqrt(norm_p/ ((i_max) * (j_max)));

    while(n < max_it) {
        calculate_ghost<<<blocks, threads>>>();

        cudaDeviceSynchronize();

        printf("RHS calculated!\n");
        // Now execute de SOR black and red
        red_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, di_max, dj_max, d_delta_x, d_delta_y, omega);
        
        cudaDeviceSynchronize();

        black_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, di_max, dj_max, d_delta_x, d_delta_y, omega);

        cudaDeviceSynchronize();
    
        residual_kernel<<<blocks, threads>>>(d_res, d_p, d_RHS, di_max, dj_max, d_delta_x, d_delta_y);
        
        cudaDeviceSynchronize();

        L2_norm<<<blocks, threads>>>(d_norm_res, d_res, di_max, dj_max);
        cudaDeviceSynchronize();
        cudaMemcpy(&norm_res, d_norm_res, sizeof(double), cudaMemcpyDeviceToHost);
        double temp = sqrt(norm_res / ((i_max) * (j_max)));
        if(temp <= epsilon * (norm + 0.01)) {
            return 0;
        }
    }

    printf("SOR complete!\n");
    update_velocity_kernel<<<blocks, threads>>>(d_u, d_v, d_p, di_max, dj_max, d_delta_t, d_delta_x, d_delta_y);
    cudaDeviceSynchronize();
    printf("Velocities updated!\n");
    // update the velocities

    double result[3];
    extract_value_kernel<<<1, 1>>>(d_u, d_v, d_p, i_max, j_max, result);
    cudaDeviceSynchronize();

    printf("U-CENTER: %.6f\n", result[0]);
    printf("V-CENTER: %.6f\n", result[1]);
    printf("P-CENTER: %.6f\n", result[2]);
    
}


__global__ void min_and_gamma (){
    double min = fmin(Re / 2.0 / ( 1.0 / d_delta_x / d_delta_x + 1.0 / d_delta_y / d_delta_y ), d_delta_x / fabs(du_max));
    min = fmin(min, d_delta_y / fabs(dv_max));
    min = fmin(min, 3.0);
    d_delta_t = tau * min;
    d_gamma = fmax(du_max * d_delta_t / d_delta_x, dv_max * d_delta_t / d_delta_y);
}


__global__ void max_reduce_kernel(int i_max, int j_max, double* arr, double* max_val) {
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double max_val_local = 0.0;

    for (int i = global_idx; i < i_max * j_max; i += stride) {
        if (arr[i] > max_val_local) {
            max_val_local = arr[i];
        }
    }

    shared_data[tid] = max_val_local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_data[tid + s] > shared_data[tid]) {
            shared_data[tid] = shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_val, shared_data[0]);
    }
}


BoundaryPoint* generate_boundary_indices(int i_max, int j_max, int* total_points) {
    *total_points = 2 * (i_max + j_max);
    BoundaryPoint* h_boundary_indices = (BoundaryPoint*)malloc(*total_points * sizeof(BoundaryPoint));
    int idx = 0;

    // Borda TOP (j = j_max)
    for (int i = 1; i <= i_max; i++) {
        h_boundary_indices[idx++] = (BoundaryPoint){i, j_max, 0};
    }

    // Borda BOTTOM (j = 0)
    for (int i = 1; i <= i_max; i++) {
        h_boundary_indices[idx++] = (BoundaryPoint){i, 0, 1};
    }

    // Borda LEFT (i = 0)
    for (int j = 1; j <= j_max; j++) {
        h_boundary_indices[idx++] = (BoundaryPoint){0, j, 2};
    }

    // Borda RIGHT (i = i_max + 1)
    for (int j = 1; j <= j_max; j++) {
        h_boundary_indices[idx++] = (BoundaryPoint){i_max + 1, j, 3};
    }

    return h_boundary_indices;
}

__global__ void update_boundaries_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 2 * (di_max + dj_max)) return;

    BoundaryPoint point = d_boundary_indices[tid];
    int i = point.i;
    int j = point.j;
    // O vfix e ufix são fixos pois tratam apenas do caso 1 do simulador
    switch (point.side) {
        case 0: // TOP
            d_v[i * (dj_max + 1) + j] = 0.0;
            d_u[i * (dj_max + 2) + (j + 1)] = 2 * 1.0 - d_u[i * (dj_max + 2) + j];
            break;
        case 1: // BOTTOM
            d_v[i * (dj_max + 1) + j] = 0.0;
            d_u[i * (dj_max + 2) + j] = 2 * 0.0 - d_u[i * (dj_max + 2) + (j + 1)];
            break;
        case 2: // LEFT
            d_u[i * (dj_max + 2) + j] = 0.0;
            d_v[i * (dj_max + 1) + j] = 2 * 0.0 - d_v[(i + 1) * (dj_max + 1) + j];
            break;
        case 3: // RIGHT
            d_u[i * (dj_max + 2) + j] = 0.0;
            d_v[i * (dj_max + 1) + j] = 2 * 0.0 - d_v[(i - 1) * (dj_max + 1) + j];
            break;
    }
}

// Funções diferenças finitas com índices linearizados para GPU

__device__ double du2_dx(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max) {
    // Índices lineares
    int idx = i * (j_max + 2) + j;
    int idx_i_plus = (i+1) * (j_max + 2) + j;
    int idx_i_minus = (i-1) * (j_max + 2) + j;
    
    double stencil1 = 0.5 * (u[idx] + u[idx_i_plus]);
    double stencil2 = 0.5 * (u[idx_i_minus] + u[idx]);

    double stencil3 = fabs(stencil1) * 0.5 * (u[idx] - u[idx_i_plus]);
    double stencil4 = fabs(stencil2) * 0.5 * (u[idx_i_minus] - u[idx]);

    return 1/delta_x * (stencil1*stencil1 - stencil2*stencil2) + gamma / delta_x * (stencil3 - stencil4);
}

__device__ double duv_dy(double* u, double* v, int i, int j, double delta_y, double gamma, int j_max) {
    // Índices lineares para u (com j_max + 2 colunas)
    int u_idx = i * (j_max + 2) + j;
    int u_idx_j_plus = i * (j_max + 2) + (j+1);
    int u_idx_j_minus = i * (j_max + 2) + (j-1);
    
    // Índices lineares para v (com j_max + 1 colunas) 
    int v_idx = i * (j_max + 1) + j;
    int v_idx_i_plus = (i+1) * (j_max + 1) + j;
    int v_idx_j_minus = i * (j_max + 1) + (j-1);
    int v_idx_i_plus_j_minus = (i+1) * (j_max + 1) + (j-1);

    double stencil1 = 0.5 * (v[v_idx] + v[v_idx_i_plus]);
    double stencil2 = 0.5 * (v[v_idx_j_minus] + v[v_idx_i_plus_j_minus]);

    double stencil3 = stencil1 * 0.5 * (u[u_idx] + u[u_idx_j_plus]);
    double stencil4 = stencil2 * 0.5 * (u[u_idx_j_minus] + u[u_idx]);

    double stencil5 = fabs(stencil1) * 0.5 * (u[u_idx] - u[u_idx_j_plus]);
    double stencil6 = fabs(stencil2) * 0.5 * (u[u_idx_j_minus] - u[u_idx]);

    return 1/delta_y * (stencil3 - stencil4) + gamma / delta_y * (stencil5 - stencil6);
}

__device__ double dv2_dy(double* v, double* u, int i, int j, double delta_y, double gamma, int j_max) {
    // Índices lineares para v
    int idx = i * (j_max + 1) + j;
    int idx_j_plus = i * (j_max + 1) + (j+1);
    int idx_j_minus = i * (j_max + 1) + (j-1);
    
    double stencil1 = 0.5 * (v[idx] + v[idx_j_plus]);
    double stencil2 = 0.5 * (v[idx_j_minus] + v[idx]);

    double stencil3 = fabs(stencil1) * 0.5 * (v[idx] - v[idx_j_plus]);
    double stencil4 = fabs(stencil2) * 0.5 * (v[idx_j_minus] - v[idx]);

    return 1/delta_y * (stencil1*stencil1 - stencil2*stencil2) + gamma / delta_y * (stencil3 - stencil4);
}

__device__ double duv_dx(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max) {
    // Índices lineares para u
    int u_idx = i * (j_max + 2) + j;
    int u_idx_j_plus = i * (j_max + 2) + (j+1);
    int u_idx_i_minus = (i-1) * (j_max + 2) + j;
    int u_idx_i_minus_j_plus = (i-1) * (j_max + 2) + (j+1);
    
    // Índices lineares para v
    int v_idx = i * (j_max + 1) + j;
    int v_idx_i_plus = (i+1) * (j_max + 1) + j;
    int v_idx_i_minus = (i-1) * (j_max + 1) + j;

    double stencil1 = 0.5 * (u[u_idx] + u[u_idx_j_plus]);
    double stencil2 = 0.5 * (u[u_idx_i_minus] + u[u_idx_i_minus_j_plus]);

    double stencil3 = stencil1 * 0.5 * (v[v_idx] + v[v_idx_i_plus]);
    double stencil4 = stencil2 * 0.5 * (v[v_idx_i_minus] + v[v_idx]);

    double stencil5 = fabs(stencil1) * 0.5 * (v[v_idx] - v[v_idx_i_plus]);
    double stencil6 = fabs(stencil2) * 0.5 * (v[v_idx_i_minus] - v[v_idx]);

    return 1/delta_x * (stencil3 - stencil4) + gamma / delta_x * (stencil5 - stencil6);
}

/**
 * Central differences for second derivatives.
 */

__device__ double d2u_dx2(double* u, int i, int j, double delta_x, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_i_plus = (i+1) * (j_max + 2) + j;
    int idx_i_minus = (i-1) * (j_max + 2) + j;
    
    return (u[idx_i_plus] - 2 * u[idx] + u[idx_i_minus]) / (delta_x*delta_x);
}

__device__ double d2u_dy2(double* u, int i, int j, double delta_y, int j_max) {
    int idx = i * (j_max + 2) + j;
    int idx_j_plus = i * (j_max + 2) + (j+1);
    int idx_j_minus = i * (j_max + 2) + (j-1);
    
    return (u[idx_j_plus] - 2 * u[idx] + u[idx_j_minus]) / (delta_y*delta_y);
}

__device__ double d2v_dx2(double* v, int i, int j, double delta_x, int j_max) {
    int idx = i * (j_max + 1) + j;
    int idx_i_plus = (i+1) * (j_max + 1) + j;
    int idx_i_minus = (i-1) * (j_max + 1) + j;
    
    return (v[idx_i_plus] - 2 * v[idx] + v[idx_i_minus]) / (delta_x*delta_x);
}

__device__ double d2v_dy2(double* v, int i, int j, double delta_y, int j_max) {
    int idx = i * (j_max + 1) + j;
    int idx_j_plus = i * (j_max + 1) + (j+1);
    int idx_j_minus = i * (j_max + 1) + (j-1);
    
    return (v[idx_j_plus] - 2 * v[idx] + v[idx_j_minus]) / (delta_y*delta_y);
}

__global__ void calculate_F(double* F, double* u, double* v, int i_max, int j_max, double Re,
    double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // u and v must be d_u and d_v when this function is called
    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        F[i * (dj_max + 2) + j] = u[i * (dj_max + 2) + j] + delta_t * ((1/Re) * (d2u_dx2(u, i, j, delta_x) + d2u_dy2(u, i, j, delta_y)) - du2_dx(u, v, i, j, delta_x, gamma) - duv_dy(u, v, i, j, delta_y, gamma) + g_x);
    }
}

__global__ void calculate_G(double * G, double* u, double* v, int i_max, int j_max, double Re,
    double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // u and v must be d_u and d_v when this function is called
    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        // +1 ou +2
        G[i * (dj_max + 1) + j] = v[i * (dj_max + 1) + j] + delta_t * ((1/Re) * (d2v_dx2(v, i, j, delta_x) + d2v_dy2(v, i, j, delta_y)) - duv_dx(u, v, i, j, delta_x, gamma) - dv2_dy(v, u, i, j, delta_y, gamma) + g_y);
    }
}


__global__ void calculate_RHS(double* RHS, double* F, double* G, double* u, double* v, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        RHS[i * (dj_max + 2) + j] = 1.0 / delta_t * ((F[i * (dj_max + 2) + j] - F[(i-1) * (dj_max + 2) + j]) / delta_x + (G[i * (dj_max + 1) + j] - G[i * (dj_max + 1) + (j-1)]) / delta_y);
    }
}

__global__ void red_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max,
    double delta_x, double delta_y, double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double dxdx = delta_x * delta_x;
    double dydy = delta_y * delta_y;
    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        if ((i + j) % 2 == 0) {
            p[i * (dj_max + 2) + j] = (1 - omega) * p[i * (dj_max + 2) + j] +
                omega / (2.0 * (1.0/ dxdx + 1.0 /dydy))*
                ((p[(i+1) * (dj_max + 2) + j] + p[(i-1) * (dj_max + 2) + j]) / dxdx + (p[i * (dj_max + 2) + (j+1)] + p[i * (dj_max + 2) + (j-1)]) / dydy -
                RHS[i * (dj_max + 2) + j]);
        }
    }
}


__global__ void black_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max,
    double delta_x, double delta_y, double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        if ((i + j) % 2 == 1) {
            p[i * (dj_max + 2) + j] = (1 - omega) * p[i * (dj_max + 2) + j] +
                omega / (2.0 * (1.0/ dxdx + 1.0 /dydy))*
                ((p[(i+1) * (dj_max + 2) + j] + p[(i-1) * (dj_max + 2) + j]) / dxdx + (p[i * (dj_max + 2) + (j+1)] + p[i * (dj_max + 2) + (j-1)]) / dydy -
                RHS[i * (dj_max + 2) + j]);
        }
    }
}


__global__ void calculate_ghost() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 2 * (di_max + dj_max)) return;
    
    BoundaryPoint point = d_boundary_indices[tid];
    int i = point.i;
    int j = point.j;
    
    // Tratar condições de contorno de Neumann para pressão (gradiente zero)
    switch (point.side) {
        case 0: // TOP (j = j_max)
            // p[i][j_max+1] = p[i][j_max]
            d_p[i * (dj_max + 2) + (j+1)] = d_p[i * (dj_max + 2) + j];
            break;
            
        case 1: // BOTTOM (j = 0)
            // p[i][0] = p[i][1]
            d_p[i * (dj_max + 2) + 0] = d_p[i * (dj_max + 2) + 1];
            break;
            
        case 2: // LEFT (i = 0)
            // p[0][j] = p[1][j]
            d_p[0 * (dj_max + 2) + j] = d_p[1 * (dj_max + 2) + j];
            break;
            
        case 3: // RIGHT (i = i_max+1)
            // p[i_max+1][j] = p[i_max][j]
            d_p[(i_max + 1) * (dj_max + 2) + j] = d_p[i_max * (dj_max + 2) + j];
            break;
    }
}


__global__ void L2_norm(double* norm, double* m, int i_max, int j_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= i_max * j_max) return;

    double value = m[tid];
    atomicAdd(norm, value * value);
}


__global__ void residual_kernel(double* res, double* p, double* RHS, int i_max, int j_max,
    double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        res[i * (dj_max + 2) + j] = ((p[(i+1) * (dj_max + 2) + j] - 2 * p[i * (dj_max + 2) + j] + p[(i-1) * (dj_max + 2) + j]) / (delta_x * delta_x) +
            (p[i * (dj_max + 2) + (j+1)] - 2 * p[i * (dj_max + 2) + j] + p[i * (dj_max + 2) + (j-1)]) / (delta_y * delta_y)) - RHS[i * (dj_max + 2) + j];
    }
}

__global__ void update_velocity_kernel(double* u, double* v, double* p, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= di_max && j > 0 && j <= dj_max) {
        if (i <= di_max - 1) u[i * (dj_max + 2) + j] = d_F[i * (dj_max + 2) + j] - delta_t * (p[(i+1) * (dj_max + 2) + j] - p[i * (dj_max + 2) + j]) / delta_x;
        if (j <= dj_max - 1) v[i * (dj_max + 1) + j] = d_G[i * (dj_max + 1) + j] - delta_t * (p[i * (dj_max + 2) + (j+1)] - p[i * (dj_max + 2) + j]) / delta_y;
    }
}

__global__ void extract_value_kernel(double* d_u, double* d_v, double* d_p, int i_max, j_max, double* result) {
    int idx = (i_max / 2) * (j_max + 2) + (j_max / 2);
    result[0] = d_u[idx];
    result[1] = d_v[idx];
    result[2] = d_p[idx];
}