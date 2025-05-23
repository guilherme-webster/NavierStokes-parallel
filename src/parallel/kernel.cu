#include "kernel.h"
#include "vector"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// arrays iniciadas em 0
__device__ double* d_F, *d_G;
__device__ double* d_RHS, *d_res;
__device__ double* d_u, *d_v, *d_p;

// variaveis do device
__device__ double d_delta_t, d_delta_x, d_delta_y, d_gamma;
__device__ double du_max, dv_max;

// variaveis tiradas do host (device pointers)
__device__ int* d_i_max, *d_j_max;
__device__ double* d_tau, *d_Re;
__device__ BoundaryPoint* d_boundary_indices;
__device__ double* d_gx, *d_gy;
__device__ double* d_omega, *d_epsilon;
__device__ int* d_max_it;

// variables for norms
double* d_norm_p, *d_norm_res;

void init_memory(int i_max, int j_max,  BoundaryPoint* h_boundary_indices, int total_points, double* tau, double* Re, double* g_x, double* g_y
                , double* omega, double* epsilon, int* max_it) {
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    //variaveis de valor 0 (estes são ponteiros do host, cudaMalloc está correto aqui)
    cudaMalloc((void**) &d_u , size);
    cudaMalloc((void**) &d_v , size);
    cudaMalloc((void**) &d_p , size);
    cudaMalloc((void**) &d_F , size);
    cudaMalloc((void**) &d_G , size);
    cudaMalloc((void**) &d_res , size);
    cudaMalloc((void**) &d_RHS , size);

    cudaMemset(d_u, 0, size);
    cudaMemset(d_v, 0, size);
    cudaMemset(d_p, 0, size);
    cudaMemset(d_F, 0, size);
    cudaMemset(d_G, 0, size);
    cudaMemset(d_res, 0, size);
    cudaMemset(d_RHS, 0, size);
    
    // variaveis copiadas do host (para __device__ pointers)

    // Exemplo para d_i_max (passado por valor i_max)
    int* temp_d_i_max_ptr;
    cudaMalloc((void**)&temp_d_i_max_ptr, sizeof(int));
    cudaMemcpy(temp_d_i_max_ptr, &i_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_i_max, &temp_d_i_max_ptr, sizeof(int*));

    // Exemplo para d_j_max (passado por valor j_max)
    int* temp_d_j_max_ptr;
    cudaMalloc((void**)&temp_d_j_max_ptr, sizeof(int));
    cudaMemcpy(temp_d_j_max_ptr, &j_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_j_max, &temp_d_j_max_ptr, sizeof(int*));

    // Exemplo para d_tau (passado por ponteiro host tau)
    double* temp_d_tau_ptr;
    cudaMalloc((void**)&temp_d_tau_ptr, sizeof(double));
    cudaMemcpy(temp_d_tau_ptr, tau, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_tau, &temp_d_tau_ptr, sizeof(double*));

    // Exemplo para d_Re
    double* temp_d_Re_ptr;
    cudaMalloc((void**)&temp_d_Re_ptr, sizeof(double));
    cudaMemcpy(temp_d_Re_ptr, Re, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_Re, &temp_d_Re_ptr, sizeof(double*));

    // Exemplo para d_boundary_indices
    BoundaryPoint* temp_d_boundary_indices_ptr;
    cudaMalloc((void**)&temp_d_boundary_indices_ptr, total_points * sizeof(BoundaryPoint));
    cudaMemcpy(temp_d_boundary_indices_ptr, h_boundary_indices, total_points * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_boundary_indices, &temp_d_boundary_indices_ptr, sizeof(BoundaryPoint*));

    // Exemplo para d_gx
    double* temp_d_gx_ptr;
    cudaMalloc((void**)&temp_d_gx_ptr, sizeof(double));
    cudaMemcpy(temp_d_gx_ptr, g_x, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gx, &temp_d_gx_ptr, sizeof(double*));

    // Exemplo para d_gy
    double* temp_d_gy_ptr;
    cudaMalloc((void**)&temp_d_gy_ptr, sizeof(double));
    cudaMemcpy(temp_d_gy_ptr, g_y, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_gy, &temp_d_gy_ptr, sizeof(double*));
    
    // Exemplo para d_omega
    double* temp_d_omega_ptr;
    cudaMalloc((void**)&temp_d_omega_ptr, sizeof(double));
    cudaMemcpy(temp_d_omega_ptr, omega, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_omega, &temp_d_omega_ptr, sizeof(double*));

    // Exemplo para d_epsilon
    double* temp_d_epsilon_ptr;
    cudaMalloc((void**)&temp_d_epsilon_ptr, sizeof(double));
    cudaMemcpy(temp_d_epsilon_ptr, epsilon, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_epsilon, &temp_d_epsilon_ptr, sizeof(double*));

    // Exemplo para d_max_it
    int* temp_d_max_it_ptr;
    cudaMalloc((void**)&temp_d_max_it_ptr, sizeof(int));
    cudaMemcpy(temp_d_max_it_ptr, max_it, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_max_it, &temp_d_max_it_ptr, sizeof(int*));

    // Alocação para d_norm_p e d_norm_res (estes são ponteiros do host, cudaMalloc está correto)
    // É uma boa prática inicializá-los se forem usados em reduções com adição.
    cudaMalloc((void**) &d_norm_p, sizeof(double));
    cudaMemset(d_norm_p, 0, sizeof(double)); // Inicializa para 0
    cudaMalloc((void**) &d_norm_res, sizeof(double));
    cudaMemset(d_norm_res, 0, sizeof(double)); // Inicializa para 0
}


__global__ void pick_max() {
    du_max = d_u[0];
    dv_max = d_v[0];
}



double orchestration(int i_max, int j_max) {
    
    int threads = 256;
    int blocks = (i_max * j_max + threads - 1) / threads;
    int size = i_max * j_max;

    // acha o máximo da matriz u e v
    while (size > 1){
        blocks = (size + threads - 1) / threads;
        pick_max<<<1, 1>>>();
        cudaDeviceSynchronize();
        max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(*d_i_max, *d_j_max, d_u, d_norm_p);
        max_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(*d_i_max, *d_j_max, d_v, d_norm_res);
        cudaDeviceSynchronize();
        size = size / threads;
    }
    
    min_and_gamma<<<1, 1>>>();
    
    cudaDeviceSynchronize();

    update_boundaries_kernel<<<blocks, threads>>>();

    cudaDeviceSynchronize();

    printf("Conditions set!\n");

    // now we calculate F and G
    calculate_F<<<blocks, threads>>>(d_F, d_u, d_v, *d_i_max, *d_j_max, *d_Re, *d_gx, d_delta_t, d_delta_x, d_delta_y, d_gamma);
    calculate_G<<<blocks, threads>>>(d_G, d_u, d_v, *d_i_max, *d_j_max, *d_Re, *d_gy, d_delta_t, d_delta_x, d_delta_y, d_gamma);    

    cudaDeviceSynchronize();

    printf("F, G calculated!\n");

    // now we calculate rhs
    calculate_RHS<<<blocks, threads>>>(d_RHS, d_F, d_G, d_u, d_v, *d_i_max, *d_j_max, d_delta_t, d_delta_x, d_delta_y);

    cudaDeviceSynchronize();
    
    L2_norm<<<blocks, threads>>>(d_norm_p, d_p, *d_i_max, *d_j_max);
    
    cudaDeviceSynchronize();
    
    double norm_p;
    cudaMemcpy(&norm_p, d_norm_p, sizeof(double), cudaMemcpyDeviceToHost);
    double norm = sqrt(norm_p/ ((i_max) * (j_max)));
    int it = 0;
    int max_it;
    double epsilon;
    cudaMemcpy(&max_it, d_max_it, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&epsilon, d_epsilon, sizeof(double), cudaMemcpyDeviceToHost);
    
    while(it < max_it) {
        calculate_ghost<<<blocks, threads>>>();

        cudaDeviceSynchronize();

        printf("RHS calculated!\n");
        // Now execute de SOR black and red
        red_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, *d_i_max, *d_j_max, d_delta_x, d_delta_y, *d_omega);
        
        cudaDeviceSynchronize();

        black_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, *d_i_max, *d_j_max, d_delta_x, d_delta_y, *d_omega);

        cudaDeviceSynchronize();
    
        residual_kernel<<<blocks, threads>>>(d_res, d_p, d_RHS, *d_i_max, *d_j_max, d_delta_x, d_delta_y);
        
        cudaDeviceSynchronize();

        L2_norm<<<blocks, threads>>>(d_norm_res, d_res, *d_i_max, *d_j_max);
        cudaDeviceSynchronize();
        double norm_res;
        cudaMemcpy(&norm_res, d_norm_res, sizeof(double), cudaMemcpyDeviceToHost);
        double temp = sqrt(norm_res / ((i_max) * (j_max)));
        if(temp <= epsilon * (norm + 0.01)) {
            return 0;
        }
        it++;
    }

    printf("SOR complete!\n");
    update_velocity_kernel<<<blocks, threads>>>(d_u, d_v, d_p, *d_i_max, *d_j_max, d_delta_t, d_delta_x, d_delta_y);
    cudaDeviceSynchronize();
    printf("Velocities updated!\n");
    // update the velocities

    double result[4];
    extract_value_kernel<<<1, 1>>>(d_u, d_v, d_p, i_max, j_max, result);
    cudaDeviceSynchronize();

    printf("U-CENTER: %.6f\n", result[0]);
    printf("V-CENTER: %.6f\n", result[1]);
    printf("P-CENTER: %.6f\n", result[2]);

    return result[3];
}


__global__ void min_and_gamma (){
    double min = fmin(*d_Re / 2.0 / ( 1.0 / d_delta_x / d_delta_x + 1.0 / d_delta_y / d_delta_y ), d_delta_x / fabs(du_max));
    min = fmin(min, d_delta_y / fabs(dv_max));
    min = fmin(min, 3.0);
    d_delta_t = *d_tau * min;
    d_gamma = fmax(du_max * d_delta_t / d_delta_x, dv_max * d_delta_t / d_delta_y);
}


__device__ double atomicMax(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        // Compara o valor atual (old) com o novo (val) como doubles
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void max_reduce_kernel(int i_max, int j_max, double* arr, double* max_val) {
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double max_val_local = (global_idx < i_max * j_max) ? arr[global_idx] : -1e30;

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
        atomicMax(&max_val[0], shared_data[0]);
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
    if (tid >= 2 * (*d_i_max + *d_j_max)) return;

    BoundaryPoint point = d_boundary_indices[tid];
    int i = point.i;
    int j = point.j;
    // O vfix e ufix são fixos pois tratam apenas do caso 1 do simulador
    switch (point.side) {
        case 0: // TOP
            d_v[i * (*d_j_max + 1) + j] = 0.0;
            d_u[i * (*d_j_max + 2) + (j + 1)] = 2 * 1.0 - d_u[i * (*d_j_max + 2) + j];
            break;
        case 1: // BOTTOM
            d_v[i * (*d_j_max + 1) + j] = 0.0;
            d_u[i * (*d_j_max + 2) + j] = 2 * 0.0 - d_u[i * (*d_j_max + 2) + (j + 1)];
            break;
        case 2: // LEFT
            d_u[i * (*d_j_max + 2) + j] = 0.0;
            d_v[i * (*d_j_max + 1) + j] = 2 * 0.0 - d_v[(i + 1) * (*d_j_max + 1) + j];
            break;
        case 3: // RIGHT
            d_u[i * (*d_j_max + 2) + j] = 0.0;
            d_v[i * (*d_j_max + 1) + j] = 2 * 0.0 - d_v[(i - 1) * (*d_j_max + 1) + j];
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
    double g_x, double delta_t, double delta_x, double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // u and v must be d_u and d_v when this function is called
    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        F[i * (j_max + 2) + j] = u[i * (j_max + 2) + j] + delta_t * ((1/Re) * (d2u_dx2(u, i, j, delta_x, j_max) + d2u_dy2(u, i, j, delta_y, j_max)) - du2_dx(u, v, i, j, delta_x, gamma, j_max) - duv_dy(u, v, i, j, delta_y, gamma, j_max) + g_x);
    }
}

__global__ void calculate_G(double * G, double* u, double* v, int i_max, int j_max, double Re,
    double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    // u and v must be d_u and d_v when this function is called
    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        // +1 ou +2
        G[i * (j_max + 1) + j] = v[i * (j_max + 1) + j] + delta_t * ((1/Re) * (d2v_dx2(v, i, j, delta_x, j_max) + d2v_dy2(v, i, j, delta_y, j_max)) - duv_dx(u, v, i, j, delta_x, gamma, j_max) - dv2_dy(v, u, i, j, delta_y, gamma, j_max) + g_y);
    }
}


__global__ void calculate_RHS(double* RHS, double* F, double* G, double* u, double* v, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        RHS[i * (j_max + 2) + j] = 1.0 / delta_t * ((F[i * (j_max + 2) + j] - F[(i-1) * (j_max + 2) + j]) / delta_x + (G[i * (j_max + 1) + j] - G[i * (j_max + 1) + (j-1)]) / delta_y);
    }
}

__global__ void red_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max,
    double delta_x, double delta_y, double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double dxdx = delta_x * delta_x;
    double dydy = delta_y * delta_y;
    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        if ((i + j) % 2 == 0) {
            p[i * (j_max + 2) + j] = (1 - omega) * p[i * (j_max + 2) + j] +
                omega / (2.0 * (1.0/ dxdx + 1.0 /dydy))*
                ((p[(i+1) * (j_max + 2) + j] + p[(i-1) * (j_max + 2) + j]) / dxdx + (p[i * (j_max + 2) + (j+1)] + p[i * (j_max + 2) + (j-1)]) / dydy -
                RHS[i * (j_max + 2) + j]);
        }
    }
}


__global__ void black_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max,
    double delta_x, double delta_y, double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double dxdx = delta_x * delta_x;
    double dydy = delta_y * delta_y;

    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        if ((i + j) % 2 == 1) {
            p[i * (j_max + 2) + j] = (1 - omega) * p[i * (j_max + 2) + j] +
                omega / (2.0 * (1.0/ dxdx + 1.0 /dydy))*
                ((p[(i+1) * (j_max + 2) + j] + p[(i-1) * (j_max + 2) + j]) / dxdx + (p[i * (j_max + 2) + (j+1)] + p[i * (j_max + 2) + (j-1)]) / dydy -
                RHS[i * (j_max + 2) + j]);
        }
    }
}


__global__ void calculate_ghost() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 2 * (*d_i_max + *d_j_max)) return;
    
    BoundaryPoint point = d_boundary_indices[tid];
    int i = point.i;
    int j = point.j;
    
    // Tratar condições de contorno de Neumann para pressão (gradiente zero)
    switch (point.side) {
        case 0: // TOP (j = j_max)
            // p[i][j_max+1] = p[i][j_max]
            d_p[i * (*d_j_max + 2) + (j+1)] = d_p[i * (*d_j_max + 2) + j];
            break;
            
        case 1: // BOTTOM (j = 0)
            // p[i][0] = p[i][1]
            d_p[i * (*d_j_max + 2) + 0] = d_p[i * (*d_j_max + 2) + 1];
            break;
            
        case 2: // LEFT (i = 0)
            // p[0][j] = p[1][j]
            d_p[0 * (*d_j_max + 2) + j] = d_p[1 * (*d_j_max + 2) + j];
            break;
            
        case 3: // RIGHT (i = i_max+1)
            // p[i_max+1][j] = p[i_max][j]
            d_p[(*d_i_max + 1) * (*d_j_max + 2) + j] = d_p[*d_i_max * (*d_j_max + 2) + j];
            break;
    }
}



__global__ void L2_norm(double* norm, double* m, int i_max, int j_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= i_max * j_max) return;

    double value = m[tid];
    atomicAddDouble(norm, value * value);
}


__global__ void residual_kernel(double* res, double* p, double* RHS, int i_max, int j_max,
    double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        res[i * (j_max + 2) + j] = ((p[(i+1) * (j_max + 2) + j] - 2 * p[i * (j_max + 2) + j] + p[(i-1) * (j_max + 2) + j]) / (delta_x * delta_x) +
            (p[i * (j_max + 2) + (j+1)] - 2 * p[i * (j_max + 2) + j] + p[i * (j_max + 2) + (j-1)]) / (delta_y * delta_y)) - RHS[i * (j_max + 2) + j];
    }
}

__global__ void update_velocity_kernel(double* u, double* v, double* p, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        if (i <= i_max - 1) u[i * (j_max + 2) + j] = d_F[i * (j_max + 2) + j] - delta_t * (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_x;
        if (j <= j_max - 1) v[i * (j_max + 1) + j] = d_G[i * (j_max + 1) + j] - delta_t * (p[i * (j_max + 2) + (j+1)] - p[i * (j_max + 2) + j]) / delta_y;
    }
}

__global__ void extract_value_kernel(double* d_u, double* d_v, double* d_p, int i_max,int j_max, double* result) {
    int idx = (i_max / 2) * (j_max + 2) + (j_max / 2);
    result[0] = d_u[idx];
    result[1] = d_v[idx];
    result[2] = d_p[idx];
    result[3] = d_delta_t;
}