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
    size_t size_v = (i_max + 1) * (j_max + 1) * sizeof(double);
    
    printf("DEBUG: Allocating memory - u,p,F,G,RHS,res size: %zu bytes\n", size);
    printf("DEBUG: Allocating memory - v size: %zu bytes\n", size_v);
    
    // ✅ Verificar memória disponível
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    printf("DEBUG: GPU Memory - Free: %zu MB, Total: %zu MB\n", free_mem/(1024*1024), total_mem/(1024*1024));
    
    //variaveis de valor 0
    CUDA_CHECK(cudaMalloc((void**)&d_u, size));
    CUDA_CHECK(cudaMalloc((void**)&d_v, size_v));  // ✅ v tem tamanho diferente
    CUDA_CHECK(cudaMalloc((void**)&d_p, size));
    CUDA_CHECK(cudaMalloc((void**)&d_F, size));
    CUDA_CHECK(cudaMalloc((void**)&d_G, size_v));  // ✅ G tem tamanho diferente
    CUDA_CHECK(cudaMalloc((void**)&d_res, size));
    CUDA_CHECK(cudaMalloc((void**)&d_RHS, size));
    
    // ✅ Inicializar com zeros
    CUDA_CHECK(cudaMemset(d_u, 0, size));
    CUDA_CHECK(cudaMemset(d_v, 0, size_v));
    CUDA_CHECK(cudaMemset(d_p, 0, size));
    CUDA_CHECK(cudaMemset(d_F, 0, size));
    CUDA_CHECK(cudaMemset(d_G, 0, size_v));
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
    *du_max = -100000.0;  // Valor mínimo para evitar divisão por zero
    *dv_max = -100000.0;
}



double orchestration(int i_max, int j_max) {
    
    // ✅ CORREÇÃO: Configuração separada para kernels 1D e 2D
    int threads_1d = 256;
    int blocks_1d = (i_max * j_max + threads_1d - 1) / threads_1d;
    
    // ✅ Configuração 2D para kernels que usam i,j
    dim3 block_size_2d(16, 16);  // 16x16 = 256 threads por block
    dim3 grid_size_2d((i_max + block_size_2d.x - 1) / block_size_2d.x,
                      (j_max + block_size_2d.y - 1) / block_size_2d.y);

    pick_max<<<1,1>>>(d_du_max, d_dv_max, d_u, d_v); 

    max_reduce_kernel<<<blocks_1d,threads_1d,threads_1d*sizeof(double)>>>(i_max,j_max,d_u,d_norm_p);

    max_reduce_kernel<<<blocks_1d,threads_1d,threads_1d*sizeof(double)>>>(i_max,j_max,d_v,d_norm_res);

    min_and_gamma<<<1,1>>>(d_delta_t, d_gamma, d_du_max, d_dv_max, Re, tau, delta_x, delta_y); 
    KERNEL_CHECK(); 
    SYNC_CHECK("min_and_gamma");

    // Copiar valores para host
    CUDA_CHECK(cudaMemcpy(&gamma_val, d_gamma, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&delta_t, d_delta_t, sizeof(double), cudaMemcpyDeviceToHost));

    int total_boundary_points = 2 * (i_max + j_max);
    update_boundaries_kernel<<<blocks_1d,threads_1d>>>(d_u, d_v, d_boundary_indices, i_max, j_max, total_boundary_points); 
    KERNEL_CHECK(); SYNC_CHECK("update_boundaries");

    // ✅ CORREÇÃO: Usar configuração 2D para kernels 2D
    calculate_F<<<grid_size_2d, block_size_2d>>>(d_F,d_u,d_v,i_max,j_max,Re,g_x,delta_t,delta_x,delta_y,gamma_val);
    KERNEL_CHECK(); SYNC_CHECK("calculate_F");

    calculate_G<<<grid_size_2d, block_size_2d>>>(d_G,d_u,d_v,i_max,j_max,Re,g_y,delta_t,delta_x,delta_y,gamma_val);
    KERNEL_CHECK(); SYNC_CHECK("calculate_G");

    // ✅ CORREÇÃO: Usar configuração 2D para calculate_RHS
    calculate_RHS<<<grid_size_2d, block_size_2d>>>(d_RHS, d_F, d_G, d_u, d_v, i_max, j_max, delta_t, delta_x, delta_y);
    KERNEL_CHECK(); SYNC_CHECK("calculate_RHS");

    // Inicializar normas
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_norm_p, &zero, sizeof(double), cudaMemcpyHostToDevice));
    
    L2_norm<<<blocks_1d, threads_1d>>>(d_norm_p, d_p, i_max, j_max);
    KERNEL_CHECK(); SYNC_CHECK("L2_norm");

    double norm_p;
    cudaMemcpy(&norm_p, d_norm_p, sizeof(double), cudaMemcpyDeviceToHost);
    double norm = sqrt(norm_p/ ((i_max) * (j_max)));
    
    int it = 0;
    double epsilon_val;
    cudaMemcpy(&epsilon_val, d_epsilon, sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("Starting SOR: norm=%.6e, epsilon=%.6e, max_it=%d\n", norm, epsilon_val, max_it);
    
    while(it < max_it) {
        calculate_ghost<<<blocks_1d, threads_1d>>>(d_p, d_boundary_indices, i_max, j_max, total_boundary_points);
        KERNEL_CHECK(); SYNC_CHECK("calculate_ghost");
        
        // ✅ CORREÇÃO: Usar configuração 2D para red/black kernels
        red_kernel<<<grid_size_2d, block_size_2d>>>(d_p, d_RHS, d_u, d_v, i_max, j_max, delta_x, delta_y, omega);
        KERNEL_CHECK(); SYNC_CHECK("red_kernel");
        
        black_kernel<<<grid_size_2d, block_size_2d>>>(d_p, d_RHS, d_u, d_v, i_max, j_max, delta_x, delta_y, omega);
        KERNEL_CHECK(); SYNC_CHECK("black_kernel");
        
        residual_kernel<<<grid_size_2d, block_size_2d>>>(d_res, d_p, d_RHS, i_max, j_max, delta_x, delta_y);
        KERNEL_CHECK(); SYNC_CHECK("residual_kernel");

        CUDA_CHECK(cudaMemcpy(d_norm_res, &zero, sizeof(double), cudaMemcpyHostToDevice));
        L2_norm<<<blocks_1d, threads_1d>>>(d_norm_res, d_res, i_max, j_max);
        KERNEL_CHECK(); SYNC_CHECK("L2_norm_res");
        
        double norm_res;
        cudaMemcpy(&norm_res, d_norm_res, sizeof(double), cudaMemcpyDeviceToHost);
        double temp = sqrt(norm_res / ((i_max) * (j_max)));
        
        if(temp <= epsilon_val * (norm + 0.01)) {
            printf("SOR converged at iteration %d\n", it);
            break;
        }
        it++;
    }

    // ✅ CORREÇÃO: Usar configuração 2D para update_velocity
    update_velocity_kernel<<<grid_size_2d, block_size_2d>>>(d_u, d_v, d_p, d_F, d_G, i_max, j_max, delta_t, delta_x, delta_y);
    KERNEL_CHECK(); SYNC_CHECK("update_velocity");

    // ✅ CORREÇÃO: Não usar array result[] problemático
    extract_value_kernel<<<1, 1>>>(d_u, d_v, d_p, d_delta_t, i_max, j_max, nullptr);
    KERNEL_CHECK(); SYNC_CHECK("extract_value");

    // ✅ Retornar o delta_t que já temos no host
    return delta_t;
}


__global__ void min_and_gamma(double* delta_t, double* gamma, double* du_max, double* dv_max, 
                              double Re, double tau, double delta_x, double delta_y) {
    
    double min = fmin(Re / 2.0 / (1.0 / (delta_x * delta_x) + 1.0 / (delta_y * delta_y)), 
                      delta_x / fabs(*du_max) + delta_y / fabs(*dv_max));
    min = fmin(min, delta_y / fabs(*du_max) + delta_x / fabs(*dv_max));
    min = fmin(min, 3.0);
    
    *delta_t = tau * min;
    
    *gamma = fmax(*du_max * (*delta_t) / delta_x, *dv_max * (*delta_t) / delta_y);
    
    // Debug prints
    printf("DEBUG: du_max=%.6e, dv_max=%.6e, delta_t=%.6e\n", *du_max, *dv_max, *delta_t);
}


__device__ double atomicMax(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double current_val = __longlong_as_double(assumed);
        
        // ✅ Comparar com valor atual, permitindo que valores negativos sejam atualizados
        double new_val = fmax(val, current_val);
        
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
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

    // ✅ CORREÇÃO: Inicializar com valor muito negativo ou primeiro elemento
    double max_val_local = -1e308;  // Menor valor possível para double
    bool found_valid = false;

    // ✅ Iterar apenas sobre elementos válidos da grade
    for (int idx = global_idx; idx < i_max * j_max; idx += stride) {
        // ✅ Converter de índice linear para (i,j) considerando padding
        int i = idx / j_max + 1;  // i de 1 a i_max
        int j = idx % j_max + 1;  // j de 1 a j_max
        
        // ✅ Calcular índice linear correto com padding
        int linear_idx = i * (j_max + 2) + j;
        
        // ✅ Verificar limites para evitar acesso inválido
        if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
            double val = fabs(arr[linear_idx]);
            
            // ✅ Na primeira iteração válida, inicializar com o primeiro valor
            if (!found_valid) {
                max_val_local = val;
                found_valid = true;
            } else if (val > max_val_local) {
                max_val_local = val;
            }
        }
    }

    // ✅ Se não encontrou nenhum valor válido, manter 0
    if (!found_valid) {
        max_val_local = 0.0;
    }

    shared_data[tid] = max_val_local;
    __syncthreads();

    // ✅ Redução paralela na memória compartilhada
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared_data[tid + s] > shared_data[tid]) {
            shared_data[tid] = shared_data[tid + s];
        }
        __syncthreads();
    }

    // ✅ Thread 0 atualiza o resultado global
    if (tid == 0) {
        atomicMax(max_val, shared_data[0]);
    }
}

// ✅ ALTERNATIVA: Versão mais simples e robusta
__global__ void max_reduce_kernel_v2(int i_max, int j_max, double* arr, double* max_val) {
    extern __shared__ double shared_data[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ✅ Inicializar com primeiro elemento válido da thread
    double max_val_local = 0.0;
    bool initialized = false;

    for (int idx = global_idx; idx < i_max * j_max; idx += stride) {
        int i = idx / j_max + 1;
        int j = idx % j_max + 1;
        int linear_idx = i * (j_max + 2) + j;
        
        if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
            double val = fabs(arr[linear_idx]);
            
            if (!initialized) {
                max_val_local = val;
                initialized = true;
            } else {
                max_val_local = fmax(max_val_local, val);
            }
        }
    }

    shared_data[tid] = max_val_local;
    __syncthreads();

    // Redução
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmax(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_val, shared_data[0]);
    }
}

// ✅ CORREÇÃO: Melhorar a função orchestration
double orchestration(int i_max, int j_max) {
    
    int threads_1d = 256;
    int blocks_1d = (i_max * j_max + threads_1d - 1) / threads_1d;
    
    dim3 block_size_2d(16, 16);
    dim3 grid_size_2d((i_max + block_size_2d.x - 1) / block_size_2d.x,
                      (j_max + block_size_2d.y - 1) / block_size_2d.y);

    // ✅ CORREÇÃO: Inicializar com valor muito negativo para permitir valores negativos
    double neg_inf = -1e308;
    CUDA_CHECK(cudaMemcpy(d_norm_p, &neg_inf, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm_res, &neg_inf, sizeof(double), cudaMemcpyHostToDevice));

    printf("DEBUG: Starting max reduction for u and v arrays\n");

    // ✅ Encontrar máximo de u
    max_reduce_kernel_v2<<<blocks_1d,threads_1d,threads_1d*sizeof(double)>>>(i_max,j_max,d_u,d_norm_p);
    KERNEL_CHECK(); SYNC_CHECK("max_reduce_u");

    // ✅ Encontrar máximo de v  
    max_reduce_kernel_v2<<<blocks_1d,threads_1d,threads_1d*sizeof(double)>>>(i_max,j_max,d_v,d_norm_res);
    KERNEL_CHECK(); SYNC_CHECK("max_reduce_v");

    // ✅ Copiar os valores máximos encontrados
    double u_max_host, v_max_host;
    CUDA_CHECK(cudaMemcpy(&u_max_host, d_norm_p, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v_max_host, d_norm_res, sizeof(double), cudaMemcpyDeviceToHost));
    
    printf("DEBUG: Max values found - u_max=%.6e, v_max=%.6e\n", u_max_host, v_max_host);
    
    // ✅ CORREÇÃO: Garantir valores mínimos seguros
    if (u_max_host <= 0 || u_max_host < 1e-12) u_max_host = 1e-6;
    if (v_max_host <= 0 || v_max_host < 1e-12) v_max_host = 1e-6;
    
    // ✅ Atualizar os valores máximos no device
    CUDA_CHECK(cudaMemcpy(d_du_max, &u_max_host, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dv_max, &v_max_host, sizeof(double), cudaMemcpyHostToDevice));

    min_and_gamma<<<1,1>>>(d_delta_t, d_gamma, d_du_max, d_dv_max, Re, tau, delta_x, delta_y); 
    KERNEL_CHECK(); SYNC_CHECK("min_and_gamma");

    // Copiar valores para host
    CUDA_CHECK(cudaMemcpy(&gamma_val, d_gamma, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&delta_t, d_delta_t, sizeof(double), cudaMemcpyDeviceToHost));

    int total_boundary_points = 2 * (i_max + j_max);
    update_boundaries_kernel<<<blocks_1d,threads_1d>>>(d_u, d_v, d_boundary_indices, i_max, j_max, total_boundary_points); 
    KERNEL_CHECK(); SYNC_CHECK("update_boundaries");

    // ✅ CORREÇÃO: Usar configuração 2D para kernels 2D
    calculate_F<<<grid_size_2d, block_size_2d>>>(d_F,d_u,d_v,i_max,j_max,Re,g_x,delta_t,delta_x,delta_y,gamma_val);
    KERNEL_CHECK(); SYNC_CHECK("calculate_F");

    calculate_G<<<grid_size_2d, block_size_2d>>>(d_G,d_u,d_v,i_max,j_max,Re,g_y,delta_t,delta_x,delta_y,gamma_val);
    KERNEL_CHECK(); SYNC_CHECK("calculate_G");

    // ✅ CORREÇÃO: Usar configuração 2D para calculate_RHS
    calculate_RHS<<<grid_size_2d, block_size_2d>>>(d_RHS, d_F, d_G, d_u, d_v, i_max, j_max, delta_t, delta_x, delta_y);
    KERNEL_CHECK(); SYNC_CHECK("calculate_RHS");

    // Inicializar normas
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_norm_p, &zero, sizeof(double), cudaMemcpyHostToDevice));
    
    L2_norm<<<blocks_1d, threads_1d>>>(d_norm_p, d_p, i_max, j_max);
    KERNEL_CHECK(); SYNC_CHECK("L2_norm");

    double norm_p;
    cudaMemcpy(&norm_p, d_norm_p, sizeof(double), cudaMemcpyDeviceToHost);
    double norm = sqrt(norm_p/ ((i_max) * (j_max)));
    
    int it = 0;
    double epsilon_val;
    cudaMemcpy(&epsilon_val, d_epsilon, sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("Starting SOR: norm=%.6e, epsilon=%.6e, max_it=%d\n", norm, epsilon_val, max_it);
    
    while(it < max_it) {
        calculate_ghost<<<blocks_1d, threads_1d>>>(d_p, d_boundary_indices, i_max, j_max, total_boundary_points);
        KERNEL_CHECK(); SYNC_CHECK("calculate_ghost");
        
        // ✅ CORREÇÃO: Usar configuração 2D para red/black kernels
        red_kernel<<<grid_size_2d, block_size_2d>>>(d_p, d_RHS, d_u, d_v, i_max, j_max, delta_x, delta_y, omega);
        KERNEL_CHECK(); SYNC_CHECK("red_kernel");
        
        black_kernel<<<grid_size_2d, block_size_2d>>>(d_p, d_RHS, d_u, d_v, i_max, j_max, delta_x, delta_y, omega);
        KERNEL_CHECK(); SYNC_CHECK("black_kernel");
        
        residual_kernel<<<grid_size_2d, block_size_2d>>>(d_res, d_p, d_RHS, i_max, j_max, delta_x, delta_y);
        KERNEL_CHECK(); SYNC_CHECK("residual_kernel");

        CUDA_CHECK(cudaMemcpy(d_norm_res, &zero, sizeof(double), cudaMemcpyHostToDevice));
        L2_norm<<<blocks_1d, threads_1d>>>(d_norm_res, d_res, i_max, j_max);
        KERNEL_CHECK(); SYNC_CHECK("L2_norm_res");
        
        double norm_res;
        cudaMemcpy(&norm_res, d_norm_res, sizeof(double), cudaMemcpyDeviceToHost);
        double temp = sqrt(norm_res / ((i_max) * (j_max)));
        
        if(temp <= epsilon_val * (norm + 0.01)) {
            printf("SOR converged at iteration %d\n", it);
            break;
        }
        it++;
    }

    // ✅ CORREÇÃO: Usar configuração 2D para update_velocity
    update_velocity_kernel<<<grid_size_2d, block_size_2d>>>(d_u, d_v, d_p, d_F, d_G, i_max, j_max, delta_t, delta_x, delta_y);
    KERNEL_CHECK(); SYNC_CHECK("update_velocity");

    // ✅ CORREÇÃO: Não usar array result[] problemático
    extract_value_kernel<<<1, 1>>>(d_u, d_v, d_p, d_delta_t, i_max, j_max, nullptr);
    KERNEL_CHECK(); SYNC_CHECK("extract_value");

    // ✅ Retornar o delta_t que já temos no host
    return delta_t;
}