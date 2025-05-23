#include "kernel.h"
#include "vector"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Runtime CUDA error-checking
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

// simple logging
#define LOG(msg) fprintf(stderr, "[%s:%d] %s\n", __FILE__, __LINE__, msg)
// synchronize and check errors
#define SYNC_CHECK(name) \
  do { \
    cudaError_t e = cudaDeviceSynchronize(); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "Error after %s: %s\n", name, cudaGetErrorString(e)); \
      exit(e); \
    } \
  } while(0)

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
    //variaveis de valor 0
    CUDA_CHECK(cudaMalloc((void**)&d_u, size));
    CUDA_CHECK(cudaMalloc((void**)&d_v, size));
    CUDA_CHECK(cudaMalloc((void**)&d_p, size));
    CUDA_CHECK(cudaMalloc((void**)&d_F, size));
    CUDA_CHECK(cudaMalloc((void**)&d_G, size));
    CUDA_CHECK(cudaMalloc((void**)&d_res, size));
    CUDA_CHECK(cudaMalloc((void**)&d_RHS, size));
    CUDA_CHECK(cudaMalloc((void**)&du_max, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dv_max, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_delta_t, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_delta_x, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_delta_y, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_gamma, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_norm_p, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_norm_res, sizeof(double)));
    
    CUDA_CHECK(cudaMemset(d_u, 0, size));
    CUDA_CHECK(cudaMemset(d_v, 0, size));
    CUDA_CHECK(cudaMemset(d_p, 0, size));
    CUDA_CHECK(cudaMemset(d_F, 0, size));
    CUDA_CHECK(cudaMemset(d_G, 0, size));
    CUDA_CHECK(cudaMemset(d_res, 0, size));
    CUDA_CHECK(cudaMemset(d_RHS, 0, size));

    // update device symbols for d_u and d_v so __device__ pointers are valid
    cudaMemcpyToSymbol(d_u, &d_u, sizeof(double*));
    cudaMemcpyToSymbol(d_v, &d_v, sizeof(double*));

    // variaveis copiadas do host
    CUDA_CHECK(cudaMalloc((void**)&d_tau, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Re, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_i_max, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_j_max, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_boundary_indices, total_points * sizeof(BoundaryPoint)));
    CUDA_CHECK(cudaMalloc((void**)&d_gx, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_gy, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_omega, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_epsilon, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_max_it, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_norm_p, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_norm_res, sizeof(double)));

    
    cudaMemcpy(d_max_it, max_it, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, omega, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_epsilon, epsilon, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boundary_indices, h_boundary_indices, total_points * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_max, &i_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_j_max, &j_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau, tau, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Re, Re, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gx, g_x, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gy, g_y, sizeof(double), cudaMemcpyHostToDevice);

}


void free_memory_kernel() {
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_F);
    cudaFree(d_G);
    cudaFree(d_RHS);
    cudaFree(d_res);
    cudaFree(d_boundary_indices);
    cudaFree(d_tau);
    cudaFree(d_Re);
    cudaFree(d_i_max);
    cudaFree(d_j_max);
    cudaFree(d_gx);
    cudaFree(d_gy);
    cudaFree(d_omega);
    cudaFree(d_epsilon);
    cudaFree(d_max_it);
    cudaFree(d_norm_p);
    cudaFree(d_norm_res);

}


double orchestration(int i_max, int j_max) {
    
    int threads = 256;
    int blocks = (i_max * j_max + threads - 1) / threads;
    int size = i_max * j_max;

    // acha o máximo da matriz u e v
    while (size > 1){
        double* du_max, *dv_max;
        CUDA_CHECK(cudaMalloc((void**)&du_max, sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&dv_max, sizeof(double)));
        CUDA_CHECK(cudaMemset(du_max, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(dv_max, 0, sizeof(double)));
        LOG("launch max_reduce u");
        max_reduce_kernel<<<blocks,threads>>>(*d_i_max,*d_j_max,d_u,du_max);
        KERNEL_CHECK(); SYNC_CHECK("max_reduce u"); LOG("max_reduce u complete");

        LOG("launch max_reduce v");
        max_reduce_kernel<<<blocks,threads,threads*sizeof(double)>>>(*d_i_max,*d_j_max,d_v,dv_max);
        KERNEL_CHECK(); SYNC_CHECK("max_reduce v"); LOG("max_reduce v complete");

        size /= threads;
    }


    LOG("launch min_and_gamma");
    min_and_gamma<<<1,1>>>(); KERNEL_CHECK(); SYNC_CHECK("min_and_gamma"); LOG("min_and_gamma complete");

    LOG("launch update_boundaries");
    update_boundaries_kernel<<<blocks,threads>>>(); KERNEL_CHECK(); SYNC_CHECK("update_boundaries"); LOG("update_boundaries complete");

    LOG("launch calculate_F");
    calculate_F<<<blocks,threads>>>(d_F,d_u,d_v,*d_i_max,*d_j_max,*d_Re,*d_gx,d_delta_t,d_delta_x,d_delta_y,d_gamma);
    KERNEL_CHECK(); SYNC_CHECK("calculate_F"); LOG("calculate_F complete");

    LOG("launch calculate_G");
    calculate_G<<<blocks,threads>>>(d_G,d_u,d_v,*d_i_max,*d_j_max,*d_Re,*d_gy,d_delta_t,d_delta_x,d_delta_y,d_gamma);
    KERNEL_CHECK(); SYNC_CHECK("calculate_G"); LOG("calculate_G complete");

    // now we calculate rhs
    LOG("launch calculate_RHS");
    calculate_RHS<<<blocks, threads>>>(d_RHS, d_F, d_G, d_u, d_v, *d_i_max, *d_j_max, d_delta_t, d_delta_x, d_delta_y);
    KERNEL_CHECK(); SYNC_CHECK("calculate_RHS"); LOG("calculate_RHS complete");

    cudaDeviceSynchronize();
    
    LOG("launch L2_norm for norm_p");
    L2_norm<<<blocks, threads>>>(d_norm_p, d_p, *d_i_max, *d_j_max);
    KERNEL_CHECK(); SYNC_CHECK("L2_norm for norm_p"); LOG("L2_norm for norm_p complete");

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
        LOG("launch calculate_ghost");
        calculate_ghost<<<blocks, threads>>>();
        cudaDeviceSynchronize(); LOG("calculate_ghost complete");

        printf("RHS calculated!\n");
        // Now execute de SOR black and red
        LOG("launch red_kernel");
        red_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, *d_i_max, *d_j_max, d_delta_x, d_delta_y, *d_omega);
        cudaDeviceSynchronize(); LOG("red_kernel complete");

        LOG("launch black_kernel");
        black_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, *d_i_max, *d_j_max, d_delta_x, d_delta_y, *d_omega);
        cudaDeviceSynchronize(); LOG("black_kernel complete");

        LOG("launch residual_kernel");
        residual_kernel<<<blocks, threads>>>(d_res, d_p, d_RHS, *d_i_max, *d_j_max, d_delta_x, d_delta_y);
        cudaDeviceSynchronize(); LOG("residual_kernel complete");

        LOG("launch L2_norm for norm_res");
        L2_norm<<<blocks, threads>>>(d_norm_res, d_res, *d_i_max, *d_j_max);
        cudaDeviceSynchronize(); LOG("L2_norm for norm_res complete");

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

    LOG("exit orchestration");
    return result[3];
}


__global__ void min_and_gamma (){
    double min = fmin(*d_Re / 2.0 / ( 1.0 / d_delta_x / d_delta_x + 1.0 / d_delta_y / d_delta_y ), d_delta_x / fabs(du_max));
    min = fmin(min, d_delta_y / fabs(dv_max));
    min = fmin(min, 3.0);
    d_delta_t = *d_tau * min;
    d_gamma = fmax(du_max * d_delta_t / d_delta_x, dv_max * d_delta_t / d_delta_y);
    // debug print computed values
    printf("[min_and_gamma] du_max=%f dv_max=%f d_delta_t=%f d_gamma=%f\n", du_max, dv_max, d_delta_t, d_gamma);
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
    printf("blk=%d tid=%d\n", blockIdx.x, tid);
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;

    double max_val_local = -1e5;


    for (int i = global_idx; i < i_max * j_max; i += stride) {
        if (arr[i] > max_val_local) {
            max_val_local = arr[i];
            printf("max_val_local=%f\n", max_val_local);
        }
    }

    shared_data[tid] = max_val_local;
    printf("blk=%d tid=%d local_max=%f\n", blockIdx.x, tid, max_val_local);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_data[tid + s] > shared_data[tid]) {
            shared_data[tid] = shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid==0) {
      printf("blk=%d block_max=%f -> atomicMax\n", blockIdx.x, shared_data[0]);
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