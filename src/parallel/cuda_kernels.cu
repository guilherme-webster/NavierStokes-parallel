#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

// Funções utilitárias para CUDA

// Debug level: 0-none, 1-errors, 2-warnings, 3-info, 4-verbose
#define DEBUG_LEVEL 0

// Macros for debug printing
#define DEBUG_ERROR(fmt, ...) if (DEBUG_LEVEL >= 1) { fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); }
#define DEBUG_WARN(fmt, ...)  if (DEBUG_LEVEL >= 2) { fprintf(stderr, "WARNING: " fmt "\n", ##__VA_ARGS__); }
#define DEBUG_INFO(fmt, ...)  if (DEBUG_LEVEL >= 3) { fprintf(stderr, "INFO: " fmt "\n", ##__VA_ARGS__); }
#define DEBUG_VERBOSE(fmt, ...) if (DEBUG_LEVEL >= 4) { fprintf(stderr, "DEBUG: " fmt "\n", ##__VA_ARGS__); }

// Enhanced CUDA error checking with context
void check_cuda(cudaError_t error, const char *filename, const int line, const char *funcname = "")
{
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA ERROR: %s:%d:%s: %s (%d: %s)\n", 
            filename, line, funcname,
            cudaGetErrorName(error), error, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

// Check for kernel launch errors
void check_kernel_launch(const char *kernel_name, const char *filename, const int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "KERNEL LAUNCH ERROR (%s): %s:%d: %s (%s)\n", 
                kernel_name, filename, line,
                cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "KERNEL EXECUTION ERROR (%s): %s:%d: %s (%s)\n", 
                kernel_name, filename, line,
                cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Memory bounds checking helper
bool check_mem_bounds(const void* ptr, size_t size, const char* ptr_name, const char* filename, int line) {
    if (ptr == NULL) {
        fprintf(stderr, "NULL POINTER: %s at %s:%d\n", ptr_name, filename, line);
        return false;
    }
    
    // We can't actually check bounds on GPU memory from host code in a portable way
    // This is just a placeholder for the NULL check
    return true;
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__, __func__)
#define KERNEL_CHECK(kernel_name) check_kernel_launch(kernel_name, __FILE__, __LINE__)
#define CHECK_POINTER(ptr, size, name) check_mem_bounds((ptr), (size), (name), __FILE__, __LINE__)

// Declaration of initBoundaryIndices (must be added before initCudaArrays)
int initBoundaryIndices(int i_max, int j_max);

// Definition for n_min
// Helper function to find the minimum of up to 4 doubles
double n_min(int n, double a, double b, double c) {
    double min_val = a; // Start with a
    if (n >= 2 && b < min_val) {
        min_val = b;
    }
    if (n >= 3 && c < min_val) {
        min_val = c;
    }
    return min_val;
}

// Variáveis globais para arrays device memory (GPU)
double *device_p = NULL;
double *device_res = NULL;
double *device_RHS = NULL;
double *device_u = NULL;
double *device_v = NULL;
double *device_F = NULL;
double *device_G = NULL;
double *device_dpdx = NULL;
double *device_dpdy = NULL;

double u_max = 0.0;
double v_max = 0.0;

// Variáveis para índices de bordas pré-calculados
int *device_left_indices = NULL;
int *device_right_indices = NULL;
int *device_top_indices = NULL;
int *device_bottom_indices = NULL;
int num_left_indices = 0;
int num_right_indices = 0;
int num_top_indices = 0;
int num_bottom_indices = 0;

double delta_t, delta_x, delta_y, gamma_factor;  // Alterado de gamma para gamma_factor
size_t cuda_array_size = 0;
int grid_i_max = 0;
int grid_j_max = 0;

__global__ void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 to skip ghost cells
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 to skip ghost cells

    // Only update red cells (i+j is even)
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

    // Only update black cells (i+j is odd)
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
        p[i * (j_max + 2) + j] = (1.0 - omega) * p[i * (j_max + 2) + j] + 
            omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) *
            ((p[(i + 1) * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j]);
    }
}

__global__ void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= i_max && j <= j_max) {
        res[i * (j_max + 2) + j] = (p[(i + 1) * (j_max + 2) + j] - 2.0 * p[i * (j_max + 2) + j] + p[(i - 1) * (j_max + 2) + j]) / dxdx + 
            (p[i * (j_max + 2) + (j + 1)] - 2.0 * p[i * (j_max + 2) + j] + p[i * (j_max + 2) + (j - 1)]) / dydy - RHS[i * (j_max + 2) + j];
    }
    
}

// Initialize CUDA arrays once
int initCudaArrays(double** p, double** u, double** v, double** res, double** RHS, int i_max, int j_max) {
    // Verificar argumentos
    if (p == NULL || u == NULL || v == NULL || res == NULL || RHS == NULL) {
        return -1;
    }
    
    if (i_max <= 0 || j_max <= 0) {
        return -1;
    }
    
    // Salvar dimensões da grade
    grid_i_max = i_max;
    grid_j_max = j_max;
    
    // Calcular tamanho necessário
    cuda_array_size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    // Verificar se o tamanho calculado não é muito grande
    size_t free_mem = 0, total_mem = 0;
    CUDACHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    size_t total_needed = 9 * cuda_array_size; // 9 arrays GPU
    if (total_needed > free_mem) {
        return -1;
    }
    
    // Alocar memória apenas no device (GPU)
    cudaError_t err;
    
    err = cudaMalloc(&device_p, cuda_array_size);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaMalloc(&device_res, cuda_array_size);
    if (err != cudaSuccess) {
        if (device_p) cudaFree(device_p);
        return -1;
    }
    
    err = cudaMalloc(&device_RHS, cuda_array_size);
    if (err != cudaSuccess) {
        if (device_p) cudaFree(device_p);
        if (device_res) cudaFree(device_res);
        return -1;
    }
    
    err = cudaMalloc(&device_u, cuda_array_size);
    if (err != cudaSuccess) {
        if (device_p) cudaFree(device_p);
        if (device_res) cudaFree(device_res);
        if (device_RHS) cudaFree(device_RHS);
        return -1;
    }
    
    err = cudaMalloc(&device_v, cuda_array_size);
    if (err != cudaSuccess) {
        if (device_p) cudaFree(device_p);
        if (device_res) cudaFree(device_res);
        if (device_RHS) cudaFree(device_RHS);
        if (device_u) cudaFree(device_u);
        return -1;
    }
    
    err = cudaMalloc(&device_F, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    err = cudaMalloc(&device_G, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    err = cudaMalloc(&device_dpdx, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    err = cudaMalloc(&device_dpdy, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    // Verificar se as alocações foram bem-sucedidas
    if (!CHECK_POINTER(device_p, cuda_array_size, "device_p") ||
        !CHECK_POINTER(device_res, cuda_array_size, "device_res") ||
        !CHECK_POINTER(device_RHS, cuda_array_size, "device_RHS") ||
        !CHECK_POINTER(device_u, cuda_array_size, "device_u") ||
        !CHECK_POINTER(device_v, cuda_array_size, "device_v") ||
        !CHECK_POINTER(device_F, cuda_array_size, "device_F") ||
        !CHECK_POINTER(device_G, cuda_array_size, "device_G") ||
        !CHECK_POINTER(device_dpdx, cuda_array_size, "device_dpdx") ||
        !CHECK_POINTER(device_dpdy, cuda_array_size, "device_dpdy")) {
        freeCudaArrays();
        return -1;
    }
    
    // Inicializar os arrays com dados da CPU e transferir para a GPU
    double *temp_host_buffer = (double*)malloc(cuda_array_size);
    if (temp_host_buffer == NULL) {
        freeCudaArrays();
        return -1;
    }
    
    // Transferir array p para GPU
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            if (p[i] == NULL) {
                free(temp_host_buffer);
                freeCudaArrays();
                return -1;
            }
            
            size_t idx = i * (j_max + 2) + j;
            if (idx >= (i_max + 2) * (j_max + 2)) {
                free(temp_host_buffer);
                freeCudaArrays();
                return -1;
            }
            
            temp_host_buffer[idx] = p[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(device_p, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    
    // Transferir array res para GPU
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            if (res[i] == NULL) {
                free(temp_host_buffer);
                freeCudaArrays();
                return -1;
            }
            
            size_t idx = i * (j_max + 2) + j;
            temp_host_buffer[idx] = res[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(device_res, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    
    // Transferir array RHS para GPU
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            if (RHS[i] == NULL) {
                free(temp_host_buffer);
                freeCudaArrays();
                return -1;
            }
            
            size_t idx = i * (j_max + 2) + j;
            temp_host_buffer[idx] = RHS[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(device_RHS, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    
    // Transferir array u para GPU (com verificações especiais)
    memset(temp_host_buffer, 0, cuda_array_size);
    for (int i = 0; i <= i_max; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            if (u[i] == NULL) {
                free(temp_host_buffer);
                freeCudaArrays();
                return -1;
            }
            
            size_t idx = i * (j_max + 2) + j;
            temp_host_buffer[idx] = u[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(device_u, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    
    // Transferir array v para GPU (com verificações especiais)
    memset(temp_host_buffer, 0, cuda_array_size);
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max; j++) {
            if (v[i] == NULL) {
                free(temp_host_buffer);
                freeCudaArrays();
                return -1;
            }
            
            size_t idx = i * (j_max + 2) + j;
            temp_host_buffer[idx] = v[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(device_v, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    
    // Inicializar F, G, dpdx e dpdy com zeros
    memset(temp_host_buffer, 0, cuda_array_size);
    CUDACHECK(cudaMemcpy(device_F, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_G, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_dpdx, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_dpdy, temp_host_buffer, cuda_array_size, cudaMemcpyHostToDevice));
    
    // Liberar o buffer temporário
    free(temp_host_buffer);
    
    // Inicializar índices de bordas pré-calculados
    if (initBoundaryIndices(i_max, j_max) != 0) {
        fprintf(stderr, "ERROR: Failed to initialize boundary indices\n");
        return -1;
    }
    
    return 0;
}


// Free CUDA arrays once at the end
void freeCudaArrays() {
    if (device_p) {
        CUDACHECK(cudaFree(device_p));
        device_p = NULL;
    }
    
    if (device_res) {
        CUDACHECK(cudaFree(device_res));
        device_res = NULL;
    }
    
    if (device_RHS) {
        CUDACHECK(cudaFree(device_RHS));
        device_RHS = NULL;
    }
    
    if (device_u) {
        CUDACHECK(cudaFree(device_u));
        device_u = NULL;
    }
    
    if (device_v) {
        CUDACHECK(cudaFree(device_v));
        device_v = NULL;
    }
    
    if (device_F) {
        CUDACHECK(cudaFree(device_F));
        device_F = NULL;
    }
    
    if (device_G) {
        CUDACHECK(cudaFree(device_G));
        device_G = NULL;
    }
    
    if (device_dpdx) {
        CUDACHECK(cudaFree(device_dpdx));
        device_dpdx = NULL;
    }
    
    if (device_dpdy) {
        CUDACHECK(cudaFree(device_dpdy));
        device_dpdy = NULL;
    }
    
    // Duplicate free calls - removing these
    if (device_F) {
        CUDACHECK(cudaFree(device_F));
        device_F = NULL;
    }
    
    if (device_G) {
        CUDACHECK(cudaFree(device_G));
        device_G = NULL;
    }
    
    if (device_dpdx) {
        CUDACHECK(cudaFree(device_dpdx));
        device_dpdx = NULL;
    }
    
    if (device_dpdy) {
        CUDACHECK(cudaFree(device_dpdy));
        device_dpdy = NULL;
    }
    
    // Liberar arrays de índices de bordas
    if (device_left_indices) {
        CUDACHECK(cudaFree(device_left_indices));
        device_left_indices = NULL;
    }
    
    if (device_right_indices) {
        CUDACHECK(cudaFree(device_right_indices));
        device_right_indices = NULL;
    }
    
    if (device_top_indices) {
        CUDACHECK(cudaFree(device_top_indices));
        device_top_indices = NULL;
    }
    
    if (device_bottom_indices) {
        CUDACHECK(cudaFree(device_bottom_indices));
        device_bottom_indices = NULL;
    }
}

// Função para inicializar índices de bordas pré-calculados
int initBoundaryIndices(int i_max, int j_max) {
    // Calcular tamanhos dos arrays de índices
    num_left_indices = j_max;
    num_right_indices = j_max;
    num_top_indices = i_max;
    num_bottom_indices = i_max;
    
    // Alocar memória para índices no host
    int *h_left_indices = (int*)malloc(num_left_indices * sizeof(int));
    int *h_right_indices = (int*)malloc(num_right_indices * sizeof(int));
    int *h_top_indices = (int*)malloc(num_top_indices * sizeof(int));
    int *h_bottom_indices = (int*)malloc(num_bottom_indices * sizeof(int));
    
    if (!h_left_indices || !h_right_indices || !h_top_indices || !h_bottom_indices) {
        printf("ERROR: Failed to allocate host memory for boundary indices\n");
        return -1;
    }
    
    // Pré-calcular índices para cada borda
    // LEFT (i=0, j=1 até j_max)
    for (int j = 1; j <= j_max; j++) {
        h_left_indices[j-1] = 0 * (j_max + 2) + j;
    }
    
    // RIGHT (i=i_max ou i_max+1, j=1 até j_max)
    for (int j = 1; j <= j_max; j++) {
        h_right_indices[j-1] = i_max * (j_max + 2) + j;
    }
    
    // TOP (i=1 até i_max, j=j_max ou j_max+1)
    for (int i = 1; i <= i_max; i++) {
        h_top_indices[i-1] = i * (j_max + 2) + j_max;
    }
    
    // BOTTOM (i=1 até i_max, j=0)
    for (int i = 1; i <= i_max; i++) {
        h_bottom_indices[i-1] = i * (j_max + 2) + 0;
    }
    
    // Alocar memória na GPU e copiar índices
    CUDACHECK(cudaMalloc(&device_left_indices, num_left_indices * sizeof(int)));
    CUDACHECK(cudaMalloc(&device_right_indices, num_right_indices * sizeof(int)));
    CUDACHECK(cudaMalloc(&device_top_indices, num_top_indices * sizeof(int)));
    CUDACHECK(cudaMalloc(&device_bottom_indices, num_bottom_indices * sizeof(int)));
    
    CUDACHECK(cudaMemcpy(device_left_indices, h_left_indices, num_left_indices * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_right_indices, h_right_indices, num_right_indices * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_top_indices, h_top_indices, num_top_indices * sizeof(int), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(device_bottom_indices, h_bottom_indices, num_bottom_indices * sizeof(int), cudaMemcpyHostToDevice));
    
    // Liberar memória do host
    free(h_left_indices);
    free(h_right_indices);
    free(h_top_indices);
    free(h_bottom_indices);
    
    return 0;
}

// Kernels otimizados para condições de contorno usando índices pré-calculados
__global__ void set_noslip_optimized_kernel(double* u, double* v, int* indices, int num_indices, int side, int i_max, int j_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_indices) return;
    
    int idx = indices[tid];
    
    if (side == LEFT) {
        u[idx] = 0.0;
        v[idx] = -v[idx + (j_max + 2)]; // -v[1][j]
    } else if (side == RIGHT) {
        u[idx] = 0.0;
        v[idx + (j_max + 2)] = -v[idx]; // v[i_max+1][j] = -v[i_max][j]
    } else if (side == TOP) {
        u[idx + 1] = -u[idx]; // u[i][j_max+1] = -u[i][j_max]
        v[idx] = 0.0;
    } else if (side == BOTTOM) {
        u[idx] = -u[idx + 1]; // u[i][0] = -u[i][1]
        v[idx] = 0.0;
    }
}

__global__ void set_inflow_optimized_kernel(double* u, double* v, int* indices, int num_indices, int side, double u_fix, double v_fix, int i_max, int j_max) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_indices) return;
    
    int idx = indices[tid];
    
    if (side == LEFT) {
        u[idx] = u_fix;
        v[idx] = 2 * v_fix - v[idx + (j_max + 2)]; // 2 * v_fix - v[1][j]
    } else if (side == RIGHT) {
        u[idx] = u_fix;
        v[idx + (j_max + 2)] = 2 * v_fix - v[idx]; // v[i_max+1][j] = 2 * v_fix - v[i_max][j]
    } else if (side == TOP) {
        u[idx + 1] = 2 * u_fix - u[idx]; // u[i][j_max+1] = 2 * u_fix - u[i][j_max]
        v[idx] = v_fix;
    } else if (side == BOTTOM) {
        u[idx] = 2 * u_fix - u[idx + 1]; // u[i][0] = 2 * u_fix - u[i][1]
        v[idx] = v_fix;
    }
}

// Kernel combinado para todas as condições de contorno
__global__ void boundary_conditions_combined_kernel(double* u, double* v, int i_max, int j_max, 
                                                  int problem, double f, double t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BORDAS HORIZONTAIS (i=1..i_max, j=0 ou j=j_max)
    if (idx < i_max) {
        int i = idx + 1;
        
        // BOTTOM (j=0)
        u[i * (j_max + 2) + 0] = -u[i * (j_max + 2) + 1];
        v[i * (j_max + 2) + 0] = 0.0;
        
        // TOP (j=j_max)
        if (problem == 1) {
            // Lid-driven cavity
            u[i * (j_max + 2) + j_max+1] = -u[i * (j_max + 2) + j_max] + 2.0; // u=1.0 no topo
            v[i * (j_max + 2) + j_max] = 0.0;
        } else if (problem == 2) {
            // Periodic boundary
            u[i * (j_max + 2) + j_max+1] = -u[i * (j_max + 2) + j_max] + 2.0 * sin(f * t);
            v[i * (j_max + 2) + j_max] = 0.0;
        }
    }
    
    // BORDAS VERTICAIS (i=0 ou i=i_max, j=1..j_max)
    if (idx < j_max) {
        int j = idx + 1;
        
        // LEFT (i=0)
        u[0 * (j_max + 2) + j] = 0.0;
        v[0 * (j_max + 2) + j] = -v[1 * (j_max + 2) + j];
        
        // RIGHT (i=i_max)
        u[i_max * (j_max + 2) + j] = 0.0;
        v[(i_max+1) * (j_max + 2) + j] = -v[i_max * (j_max + 2) + j];
    }
}

// Kernel para encontrar o valor máximo absoluto em uma matriz linearizada (versão especializada para double)
__global__ void max_mat_kernel_double(const double* mat, int i_max, int j_max, double* max_val) {
    // Verificações de parâmetros
    if (mat == NULL || max_val == NULL || i_max <= 0 || j_max <= 0) {
        // Na GPU não podemos fazer muito mais do que isso
        return;
    }
    
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (i_max) * (j_max);
    double local_max = 0.0;
    
    // Encontrar máximo local
    for (int k = idx; k < total; k += blockDim.x * gridDim.x) {
        int i = 1 + k / j_max;
        int j = 1 + k % j_max;
        
        // Verificação de limites
        if (i >= 0 && i <= i_max+1 && j >= 0 && j <= j_max+1) {
            double val = fabs(mat[i * (j_max + 2) + j]);
            if (val > local_max) local_max = val;
        }
    }
    
    // Armazenar em memória compartilhada
    sdata[tid] = local_max;
    __syncthreads();
    
    // Redução em bloco
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Apenas a thread 0 do bloco atualiza o máximo global
    if (tid == 0) {
        double old_val = *max_val;
        double my_val = sdata[0];
        
        // Usar atomicCAS para atualização segura do máximo
        while (my_val > old_val) {
            double assumed = old_val;
            
            // Usar union para conversão de tipo segura
            union { double d; unsigned long long int i; } old_union, new_union;
            old_union.d = old_val;
            new_union.d = my_val;
            
            // Operação atômica para atualizar o máximo global
            old_val = atomicCAS((unsigned long long int*)max_val, 
                                old_union.i, 
                                new_union.i);
                                
            // Se o valor não mudou desde nossa última leitura, podemos sair
            if (old_val == assumed) {
                break;
            }
        }
    }
}

// Kernel combinado para encontrar os valores máximos de u e v simultaneamente
__global__ void max_mat_combined_kernel(const double* u_mat, const double* v_mat, int i_max, int j_max, double* u_max, double* v_max) {
    // Verificações de parâmetros
    if (u_mat == NULL || v_mat == NULL || u_max == NULL || v_max == NULL || i_max <= 0 || j_max <= 0) {
        return;
    }
    
    __shared__ double u_sdata[256];
    __shared__ double v_sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (i_max) * (j_max);
    double local_u_max = 0.0;
    double local_v_max = 0.0;
    
    // Encontrar máximos locais para u e v simultaneamente
    for (int k = idx; k < total; k += blockDim.x * gridDim.x) {
        int i = 1 + k / j_max;
        int j = 1 + k % j_max;
        
        if (i >= 0 && i <= i_max+1 && j >= 0 && j <= j_max+1) {
            double u_val = fabs(u_mat[i * (j_max + 2) + j]);
            double v_val = fabs(v_mat[i * (j_max + 2) + j]);
            if (u_val > local_u_max) local_u_max = u_val;
            if (v_val > local_v_max) local_v_max = v_val;
        }
    }
    
    // Armazenar em memória compartilhada
    u_sdata[tid] = local_u_max;
    v_sdata[tid] = local_v_max;
    __syncthreads();
    
    // Redução em bloco para ambos valores
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (u_sdata[tid + s] > u_sdata[tid]) u_sdata[tid] = u_sdata[tid + s];
            if (v_sdata[tid + s] > v_sdata[tid]) v_sdata[tid] = v_sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 atualiza os máximos globais
    if (tid == 0) {
        // Usar atomicMax para atualização segura dos máximos
        atomicMax((unsigned long long int*)u_max, __double_as_longlong(u_sdata[0]));
        atomicMax((unsigned long long int*)v_max, __double_as_longlong(v_sdata[0]));
    }
}

// Kernel para extrair múltiplos valores de pontos específicos em uma única chamada
__global__ void extract_center_values_kernel(double* u, double* v, double* p, int center_idx, 
                                           double* u_center, double* v_center, double* p_center) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *u_center = u[center_idx];
        *v_center = v[center_idx];
        *p_center = p[center_idx];
    }
}
