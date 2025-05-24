/**
 * @file main.c
 * @author Hollweck, Wigg
 * @date 10 April 2019
 * @brief Main file.
 *
 * Here typically goes a more extensive explanation of what the header
 * defines. Doxygens tags are words preceeded by either a backslash @\
 * or by an at symbol @@.
 * @see http://www.stack.nl/~dimitri/doxygen/docblocks.html
 * @see http://www.stack.nl/~dimitri/doxygen/commands.html
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

enum {
    TOP = 0,
    BOTTOM = 1,
    LEFT = 2,
    RIGHT = 3
};

// arrays iniciadas em 0 - ponteiros locais para device
double* d_F, *d_G;
double* d_RHS, *d_res;
double* d_u, *d_v, *d_p;

// variaveis do device - ponteiros locais para device
double* d_delta_t, *d_delta_x, *d_delta_y, *d_gamma;
double* d_du_max, *d_dv_max;

// variaveis tiradas do host (device pointers)
int* d_i_max, *d_j_max;
double* d_tau, *d_Re;
BoundaryPoint* d_boundary_indices;
double* d_gx, *d_gy;
double* d_omega, *d_epsilon;
int* d_max_it;
// variables for norms
double* d_norm_p, *d_norm_res;

double** u;     // velocity x-component
double** v;     // velocity y-component
double** p;     // pressure
double** F;     // F term
double** G;     // G term
double** res;   // SOR residuum
double** RHS;   // RHS of poisson equation
// Simulation parameters.
int i_max, j_max;                   // number of grid points in each direction
double a, b;                        // sizes of the grid
double Re;                          // reynolds number
double delta_t, delta_x, delta_y;   // step sizes
double gamma_val;                       // weight for Donor-Cell-stencil
double T;                           // max time for integration
double g_x;                         // x-component of g
double g_y;                         // y-component of g
double tau;                         // security factor for adaptive step size
double omega;                       // relaxation parameter
double epsilon;                     // relative tolerance for SOR
int max_it;                         // maximum iterations for SOR
int n_print;                        // output to file every ..th step
int problem;                        // problem type
double f; 


void init_memory(int i_max, int j_max,  BoundaryPoint* h_boundary_indices, int total_points, double* tau, double* Re, double* g_x, double* g_y
                , double* omega, double* epsilon, int* max_it);


void free_memory_kernel();

__global__ void pick_max(double* du_max, double* dv_max, double* u, double* v);

double orchestration(int i_max, int j_max);

__global__ void min_and_gamma(double* delta_t, double* gamma, double* du_max, double* dv_max, 
                              double Re, double tau, double delta_x, double delta_y);


__device__ double atomicMax(double* address, double val);

__device__ double atomicAddDouble(double* address, double val);


__global__ void max_reduce_kernel(int i_max, int j_max, double* arr, double* max_val);

BoundaryPoint* generate_boundary_indices(int i_max, int j_max, int* total_points);

__global__ void update_boundaries_kernel(double* u, double* v, BoundaryPoint* boundary_indices, 
                                        int i_max, int j_max, int total_boundary_points);

// Funções diferenças finitas com índices linearizados para GPU

__device__ double du2_dx(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max);

__device__ double duv_dy(double* u, double* v, int i, int j, double delta_y, double gamma, int j_max);

__device__ double dv2_dy(double* v, double* u, int i, int j, double delta_y, double gamma, int j_max);

__device__ double duv_dx(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max);
/**
 * Central differences for second derivatives.
 */

__device__ double d2u_dx2(double* u, int i, int j, double delta_x, int j_max) ;

__device__ double d2u_dy2(double* u, int i, int j, double delta_y, int j_max) ;

__device__ double d2v_dx2(double* v, int i, int j, double delta_x, int j_max) ;

__device__ double d2v_dy2(double* v, int i, int j, double delta_y, int j_max) ;

__global__ void calculate_F(double* F, double* u, double* v, int i_max, int j_max, double Re,
    double g_x, double delta_t, double delta_x, double delta_y, double gamma) ;

__global__ void calculate_G(double * G, double* u, double* v, int i_max, int j_max, double Re,
    double g_y, double delta_t, double delta_x, double delta_y, double gamma) ;


__global__ void calculate_RHS(double* RHS, double* F, double* G, double* u, double* v, int i_max, int j_max,
    double delta_t, double delta_x, double delta_y) ;

__global__ void red_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max,
    double delta_x, double delta_y, double omega) ;

__global__ void black_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max,
    double delta_x, double delta_y, double omega) ;


__global__ void calculate_ghost(double* p, BoundaryPoint* boundary_indices, 
                               int i_max, int j_max, int total_boundary_points);


__global__ void L2_norm(double* norm, double* m, int i_max, int j_max);


__global__ void residual_kernel(double* res, double* p, double* RHS, int i_max, int j_max,
    double delta_x, double delta_y) ;

__global__ void update_velocity_kernel(double* u, double* v, double* p, double* F, double* G, 
                                      int i_max, int j_max, double delta_t, double delta_x, double delta_y);

__global__ void extract_value_kernel(double* u, double* v, double* p, double* delta_t_device,
                                    int i_max, int j_max, double* result);


/**
* @brief Main function.
* 
* This is the main function.
* @return 0 on exit.
*/

int main(int argc, char* argv[])
{
    const char* param_file = "parameters.txt"; // file containing parameters

    // fprintf(stderr, "CUDA: Working directory test\n");
    
    // Test if we can open the file directly
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "CUDA: Could not open param_file\n");
        } else {
            // fprintf(stderr, "CUDA: Successfully opened '%s'\n", argv[1]);
            param_file = argv[1];
            fclose(fp);
        }
    }
    
    // if (argc > 1) {
    //     param_file = argv[1];
    // }
    
    // Initialize all parameters.
	init(&problem, &f, &i_max, &j_max, &a, &b, &Re, &T, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);
 
    // Set step size in space.
    delta_x = a / i_max;
    delta_y = b / j_max;

    // Allocate memory for grids.
    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);

    // Time loop.
    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();
    int total_point;
    BoundaryPoint* boundary_points = generate_boundary_indices(i_max, j_max, &total_point);

    init_memory(i_max, j_max, boundary_points, total_point, &tau, &Re, &g_x, &g_y, &omega, &epsilon, &max_it);

    while (t < T) {
        printf("%.5f / %.5f\n---------------------\n", t, T);

        t += orchestration(i_max, j_max);

        n++;
    }

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_spent);

    // Free grid memory.
    free_memory_kernel();
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    free(boundary_points);  // Liberar memória do host
    return 0;
}

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
    CUDA_CHECK(cudaMalloc((void**)&d_du_max, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_dv_max, sizeof(double)));
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
    cudaFree(d_du_max);
    cudaFree(d_dv_max);
    cudaFree(d_delta_t);
    cudaFree(d_delta_x);
    cudaFree(d_delta_y);
    cudaFree(d_gamma);
}


__global__ void pick_max(double* du_max, double* dv_max, double* u, double* v) {
    // ✅ CORREÇÃO: Inicializar com valores pequenos mas positivos
    *du_max = 1e-6;  // Valor mínimo para evitar divisão por zero
    *dv_max = 1e-6;
}



double orchestration(int i_max, int j_max) {
    
    int threads = 256;
    int blocks = (i_max * j_max + threads - 1) / threads;

    // Copiar valores para device
    CUDA_CHECK(cudaMemcpy(d_delta_x, &delta_x, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta_y, &delta_y, sizeof(double), cudaMemcpyHostToDevice));

    pick_max<<<1,1>>>(d_du_max, d_dv_max, d_u, d_v); 

    max_reduce_kernel<<<blocks,threads,threads*sizeof(double)>>>(i_max,j_max,d_u,d_norm_p);

    max_reduce_kernel<<<blocks,threads,threads*sizeof(double)>>>(i_max,j_max,d_v,d_norm_res);

    min_and_gamma<<<1,1>>>(d_delta_t, d_gamma, d_du_max, d_dv_max, Re, tau, delta_x, delta_y); 

    // Copiar gamma_val de volta para o host
    CUDA_CHECK(cudaMemcpy(&gamma_val, d_gamma, sizeof(double), cudaMemcpyDeviceToHost));

    int total_boundary_points = 2 * (i_max + j_max);
    update_boundaries_kernel<<<blocks,threads>>>(d_u, d_v, d_boundary_indices, i_max, j_max, total_boundary_points); 

    calculate_F<<<blocks,threads>>>(d_F,d_u,d_v,i_max,j_max,Re,g_x,delta_t,delta_x,delta_y,gamma_val);

    calculate_G<<<blocks,threads>>>(d_G,d_u,d_v,i_max,j_max,Re,g_y,delta_t,delta_x,delta_y,gamma_val);

    // now we calculate rhs
    calculate_RHS<<<blocks, threads>>>(d_RHS, d_F, d_G, d_u, d_v, i_max, j_max, delta_t, delta_x, delta_y);

    cudaDeviceSynchronize();
    
    // Inicializar normas
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_norm_p, &zero, sizeof(double), cudaMemcpyHostToDevice));
    
    L2_norm<<<blocks, threads>>>(d_norm_p, d_p, i_max, j_max);

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
        calculate_ghost<<<blocks, threads>>>(d_p, d_boundary_indices, i_max, j_max, total_boundary_points);
        cudaDeviceSynchronize();
        red_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, i_max, j_max, delta_x, delta_y, omega);
        cudaDeviceSynchronize();
        black_kernel<<<blocks, threads>>>(d_p, d_RHS, d_u, d_v, i_max, j_max, delta_x, delta_y, omega);
        cudaDeviceSynchronize();
        residual_kernel<<<blocks, threads>>>(d_res, d_p, d_RHS, i_max, j_max, delta_x, delta_y);
        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(d_norm_res, &zero, sizeof(double), cudaMemcpyHostToDevice));
        L2_norm<<<blocks, threads>>>(d_norm_res, d_res, i_max, j_max);
        cudaDeviceSynchronize();
        double norm_res;
        cudaMemcpy(&norm_res, d_norm_res, sizeof(double), cudaMemcpyDeviceToHost);
        double temp = sqrt(norm_res / ((i_max) * (j_max)));
        if(temp <= epsilon * (norm + 0.01)) {
            return 0;
        }
        it++;
    }

    update_velocity_kernel<<<blocks, threads>>>(d_u, d_v, d_p, d_F, d_G, i_max, j_max, delta_t, delta_x, delta_y);
    cudaDeviceSynchronize();

    double result[4];
    extract_value_kernel<<<1, 1>>>(d_u, d_v, d_p, d_delta_t, i_max, j_max, result);
    cudaDeviceSynchronize();

    printf("U-CENTER: %.6f\n", result[0]);
    printf("V-CENTER: %.6f\n", result[1]);
    printf("P-CENTER: %.6f\n", result[2]);

    return result[3];
}


__global__ void min_and_gamma(double* delta_t, double* gamma, double* du_max, double* dv_max, 
                              double Re, double tau, double delta_x, double delta_y) {
    // ✅ CORREÇÃO: Garantir valores mínimos para evitar delta_t = 0
    double du_safe = fmax(*du_max, 1e-6);
    double dv_safe = fmax(*dv_max, 1e-6);
    
    double min = fmin(Re / 2.0 / (1.0 / (delta_x * delta_x) + 1.0 / (delta_y * delta_y)), 
                      delta_x / du_safe);
    min = fmin(min, delta_y / dv_safe);
    min = fmin(min, 3.0);
    
    *delta_t = tau * min;
    
    // ✅ Garantir delta_t mínimo
    if (*delta_t < 1e-6) {
        *delta_t = 1e-6;
    }
    
    *gamma = fmax(du_safe * (*delta_t) / delta_x, dv_safe * (*delta_t) / delta_y);
    
    // Debug prints
    printf("DEBUG: du_max=%.6e, dv_max=%.6e, delta_t=%.6e\n", *du_max, *dv_max, *delta_t);
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

    // ✅ CORREÇÃO: Inicializar com o primeiro elemento válido ou 0
    double max_val_local = 0.0;  // Ou usar arr[0] se disponível

    for (int i = global_idx; i < i_max * j_max; i += stride) {
        if (fabs(arr[i]) > max_val_local) {  // ✅ Usar valor absoluto
            max_val_local = fabs(arr[i]);
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

__global__ void update_boundaries_kernel(double* u, double* v, BoundaryPoint* boundary_indices, 
                                        int i_max, int j_max, int total_boundary_points) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_boundary_points) return;

    BoundaryPoint point = boundary_indices[tid];
    int i = point.i;
    int j = point.j;
    // O vfix e ufix são fixos pois tratam apenas do caso 1 do simulador
    switch (point.side) {
        case 0: // TOP
            v[i * (j_max + 1) + j] = 0.0;
            u[i * (j_max + 2) + (j + 1)] = 2 * 1.0 - u[i * (j_max + 2) + j];
            break;
        case 1: // BOTTOM
            v[i * (j_max + 1) + j] = 0.0;
            u[i * (j_max + 2) + j] = 2 * 0.0 - u[i * (j_max + 2) + (j + 1)];
            break;
        case 2: // LEFT
            u[i * (j_max + 2) + j] = 0.0;
            v[i * (j_max + 1) + j] = 2 * 0.0 - v[(i + 1) * (j_max + 1) + j];
            break;
        case 3: // RIGHT
            u[i * (j_max + 2) + j] = 0.0;
            v[i * (j_max + 1) + j] = 2 * 0.0 - v[(i - 1) * (j_max + 1) + j];
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


__global__ void calculate_ghost(double* p, BoundaryPoint* boundary_indices, 
                               int i_max, int j_max, int total_boundary_points) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_boundary_points) return;
    
    BoundaryPoint point = boundary_indices[tid];
    int i = point.i;
    int j = point.j;
    
    // Tratar condições de contorno de Neumann para pressão (gradiente zero)
    switch (point.side) {
        case 0: // TOP (j = j_max)
            // p[i][j_max+1] = p[i][j_max]
            p[i * (j_max + 2) + (j+1)] = p[i * (j_max + 2) + j];
            break;
            
        case 1: // BOTTOM (j = 0)
            // p[i][0] = p[i][1]
            p[i * (j_max + 2) + 0] = p[i * (j_max + 2) + 1];
            break;
            
        case 2: // LEFT (i = 0)
            // p[0][j] = p[1][j]
            p[0 * (j_max + 2) + j] = p[1 * (j_max + 2) + j];
            break;
            
        case 3: // RIGHT (i = i_max+1)
            // p[i_max+1][j] = p[i_max][j]
            p[(i_max + 1) * (j_max + 2) + j] = p[i_max * (j_max + 2) + j];
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

__global__ void update_velocity_kernel(double* u, double* v, double* p, double* F, double* G, 
                                      int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= i_max && j > 0 && j <= j_max) {
        if (i <= i_max - 1) u[i * (j_max + 2) + j] = F[i * (j_max + 2) + j] - delta_t * (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_x;
        if (j <= j_max - 1) v[i * (j_max + 1) + j] = G[i * (j_max + 1) + j] - delta_t * (p[i * (j_max + 2) + (j+1)] - p[i * (j_max + 2) + j]) / delta_y;
    }
}

__global__ void extract_value_kernel(double* u, double* v, double* p, double* delta_t_device,
                                    int i_max, int j_max, double* result) {
    int idx = (i_max / 2) * (j_max + 2) + (j_max / 2);
    result[0] = u[idx];
    result[1] = v[idx];
    result[2] = p[idx];
    result[3] = *delta_t_device;
    // Print delta_t value for debugging
    printf("Delta_t: %.6f\n", *delta_t_device);
}