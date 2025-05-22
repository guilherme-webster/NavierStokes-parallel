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

// Variáveis globais para arrays unificados
double *unified_p = NULL;
double *unified_res = NULL;
double *unified_RHS = NULL;
double *unified_u = NULL;
double *unified_v = NULL;
double *unified_F = NULL;
double *unified_G = NULL;
double *unified_dpdx = NULL;
double *unified_dpdy = NULL;
double u_max = 0.0;
double v_max = 0.0;
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
    
    size_t total_needed = 9 * cuda_array_size; // 9 arrays
    if (total_needed > free_mem) {
        return -1;
    }
    
    // Alocar memória unificada
    cudaError_t err;
    
    err = cudaMallocManaged(&unified_p, cuda_array_size);
    if (err != cudaSuccess) {
        return -1;
    }
    
    err = cudaMallocManaged(&unified_res, cuda_array_size);
    if (err != cudaSuccess) {
        if (unified_p) cudaFree(unified_p);
        return -1;
    }
    
    err = cudaMallocManaged(&unified_RHS, cuda_array_size);
    if (err != cudaSuccess) {
        if (unified_p) cudaFree(unified_p);
        if (unified_res) cudaFree(unified_res);
        return -1;
    }
    
    err = cudaMallocManaged(&unified_u, cuda_array_size);
    if (err != cudaSuccess) {
        if (unified_p) cudaFree(unified_p);
        if (unified_res) cudaFree(unified_res);
        if (unified_RHS) cudaFree(unified_RHS);
        return -1;
    }
    
    err = cudaMallocManaged(&unified_v, cuda_array_size);
    if (err != cudaSuccess) {
        if (unified_p) cudaFree(unified_p);
        if (unified_res) cudaFree(unified_res);
        if (unified_RHS) cudaFree(unified_RHS);
        if (unified_u) cudaFree(unified_u);
        return -1;
    }
    
    err = cudaMallocManaged(&unified_F, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    err = cudaMallocManaged(&unified_G, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    err = cudaMallocManaged(&unified_dpdx, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    err = cudaMallocManaged(&unified_dpdy, cuda_array_size);
    if (err != cudaSuccess) {
        freeCudaArrays(); // Liberar o que já foi alocado
        return -1;
    }
    
    // Verificar se as alocações foram bem-sucedidas
    if (!CHECK_POINTER(unified_p, cuda_array_size, "unified_p") ||
        !CHECK_POINTER(unified_res, cuda_array_size, "unified_res") ||
        !CHECK_POINTER(unified_RHS, cuda_array_size, "unified_RHS") ||
        !CHECK_POINTER(unified_u, cuda_array_size, "unified_u") ||
        !CHECK_POINTER(unified_v, cuda_array_size, "unified_v") ||
        !CHECK_POINTER(unified_F, cuda_array_size, "unified_F") ||
        !CHECK_POINTER(unified_G, cuda_array_size, "unified_G") ||
        !CHECK_POINTER(unified_dpdx, cuda_array_size, "unified_dpdx") ||
        !CHECK_POINTER(unified_dpdy, cuda_array_size, "unified_dpdy")) {
        freeCudaArrays();
        return -1;
    }
    
    // Inicializar os arrays com dados da CPU - ADICIONAR VERIFICAÇÃO DE PONTEIROS NULOS
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            // Verificar se os ponteiros p[i], res[i], etc são válidos
            if (p[i] == NULL || res[i] == NULL || RHS[i] == NULL) {
                return -1;
            }
            
            // Verificação especial para u que tem dimensão i_max+1
            if (i <= i_max && u[i] == NULL) {
                return -1;
            }
            
            // Verificação especial para v que tem dimensão j_max+1
            if (v[i] == NULL) {
                return -1;
            }
            
            // Verificar se o índice linearizado está dentro dos limites
            size_t idx = i * (j_max + 2) + j;
            if (idx >= (i_max + 2) * (j_max + 2)) {
                return -1;
            }
            
            try {
                unified_p[idx] = p[i][j];
                unified_res[idx] = res[i][j];
                unified_RHS[idx] = RHS[i][j];
                
                // Verificações especiais para u e v devido às dimensões diferentes
                if (i <= i_max && j <= j_max + 1) {
                    unified_u[idx] = u[i][j];
                }
                
                if (i <= i_max + 1 && j <= j_max) {
                    unified_v[idx] = v[i][j];
                }
                
                unified_F[idx] = 0.0;
                unified_G[idx] = 0.0;
                unified_dpdx[idx] = 0.0;
                unified_dpdy[idx] = 0.0;
            } catch (...) {
                return -1;
            }
        }
    }
    
    // Prefetch para GPU
    int device = -1;
    CUDACHECK(cudaGetDevice(&device));
    
    CUDACHECK(cudaMemPrefetchAsync(unified_p, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_RHS, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_res, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_u, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_v, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_F, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_G, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_dpdx, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_dpdy, cuda_array_size, device, NULL));
    
    return 0;
}

// Free CUDA arrays once at the end
void freeCudaArrays() {
    if (unified_p) {
        CUDACHECK(cudaFree(unified_p));
        unified_p = NULL;
    }
    
    if (unified_res) {
        CUDACHECK(cudaFree(unified_res));
        unified_res = NULL;
    }
    
    if (unified_RHS) {
        CUDACHECK(cudaFree(unified_RHS));
        unified_RHS = NULL;
    }
    
    if (unified_u) {
        CUDACHECK(cudaFree(unified_u));
        unified_u = NULL;
    }
    
    if (unified_v) {
        CUDACHECK(cudaFree(unified_v));
        unified_v = NULL;
    }
    
    if (unified_F) {
        CUDACHECK(cudaFree(unified_F));
        unified_F = NULL;
    }
    
    if (unified_G) {
        CUDACHECK(cudaFree(unified_G));
        unified_G = NULL;
    }
    
    if (unified_dpdx) {
        CUDACHECK(cudaFree(unified_dpdx));
        unified_dpdx = NULL;
    }
    
    if (unified_dpdy) {
        CUDACHECK(cudaFree(unified_dpdy));
        unified_dpdy = NULL;
    }
}

int cudaSOR(double** p,double** u,double** v, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it, double** F, double** G, double tau, double Re,
            int problem, double f, double* t, int* n_out, double g_x, double g_y) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = 0.0;
    
    // Calcular norma inicial de pressão
    for (int i = 1; i <= i_max; i++) {
        for(int j = 1; j <= j_max; j++) {
            norm_p += unified_p[i * (j_max + 2) + j] * unified_p[i * (j_max + 2) + j];
        }
    }
    norm_p = sqrt(norm_p / i_max / j_max);
    
    // 1. Calcular u_max e v_max usando kernel max_mat_kernel
    double *d_umax, *d_vmax;
    CUDACHECK(cudaMallocManaged(&d_umax, sizeof(double)));
    CUDACHECK(cudaMallocManaged(&d_vmax, sizeof(double)));
    
    if (!CHECK_POINTER(d_umax, sizeof(double), "d_umax") || 
        !CHECK_POINTER(d_vmax, sizeof(double), "d_vmax")) {
        return -1;
    }
    
    *d_umax = 0.0;
    *d_vmax = 0.0;
    int max_blocks = 32; // Ajuste conforme o tamanho do domínio
    int max_threads = 256;
    
    max_mat_kernel_double<<<max_blocks, max_threads>>>(unified_u, i_max, j_max, d_umax);
    KERNEL_CHECK("max_mat_kernel_double (u)");
    
    max_mat_kernel_double<<<max_blocks, max_threads>>>(unified_v, i_max, j_max, d_vmax);
    KERNEL_CHECK("max_mat_kernel_double (v)");
    
    u_max = *d_umax;
    v_max = *d_vmax;
    
    
    CUDACHECK(cudaFree(d_umax));
    CUDACHECK(cudaFree(d_vmax));
    
    // Calcular delta_t e gamma_factor
    delta_t = tau * n_min(4, 3.0, Re / 2.0 / ( 1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y ), delta_x / fabs(u_max), delta_y / fabs(v_max));
    gamma_factor = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);
    
    
    // 1. Boundary conditions (GPU)
    dim3 block1D_j((j_max + 127) / 128); // for sides with j_max
    dim3 block1D_i((i_max + 127) / 128); // for sides with i_max
    int threads1D = 128;
    if (problem == 1){
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, unified_u, unified_v, LEFT);
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, unified_u, unified_v, RIGHT);
        set_noslip_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, unified_u, unified_v, BOTTOM);
        set_inflow_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, unified_u, unified_v, TOP, 1.0, 0.0);
    }
    else if (problem == 2){
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, unified_u, unified_v, LEFT);
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, unified_u, unified_v, RIGHT);
        set_noslip_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, unified_u, unified_v, BOTTOM);
        set_inflow_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, unified_u, unified_v, TOP, sin(f*(*t)), 0.0);           
    }
    else {
        return -1;
    }
    CUDACHECK(cudaDeviceSynchronize());
    printf("Conditions set!\n");

    // 2. FG calculation (GPU)
    dim3 block2D(16, 16);
    dim3 grid2D((i_max+block2D.x-1)/block2D.x, (j_max+block2D.y-1)/block2D.y);
    FG_linear_kernel<<<grid2D, block2D>>>(unified_u, unified_v, unified_F, unified_G, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma_factor);
    CUDACHECK(cudaDeviceSynchronize());
    printf("F, G calculated!\n");

    // 3. RHS calculation (GPU)
    RHS_kernel<<<grid2D, block2D>>>(unified_F, unified_G, unified_RHS, i_max, j_max, delta_t, delta_x, delta_y);
    CUDACHECK(cudaDeviceSynchronize());
    printf("RHS calculated!\n");

    // 4. Copy p, RHS, res to unified arrays (GPU, if needed)
    // (If all arrays are already on GPU, this step can be skipped)

    // 5. SOR loop (already uses GPU kernels)
    dim3 blockSOR(16, 16);
    dim3 gridSOR((i_max+blockSOR.x-1)/blockSOR.x, (j_max+blockSOR.y-1)/blockSOR.y);
    while (it < max_it) {
        // Atualizar condições de contorno de pressão (ghost cells) na GPU
        // Kernel para copiar bordas (pode ser otimizado, mas aqui faz em CPU para simplicidade)
        for (int i = 1; i <= i_max; i++) {
            unified_p[i * (j_max + 2) + 0] = unified_p[i * (j_max + 2) + 1];
            unified_p[i * (j_max + 2) + (j_max + 1)] = unified_p[i * (j_max + 2) + j_max];
        }
        for (int j = 1; j <= j_max; j++) {
            unified_p[0 * (j_max + 2) + j] = unified_p[1 * (j_max + 2) + j];
            unified_p[(i_max + 1) * (j_max + 2) + j] = unified_p[i_max * (j_max + 2) + j];
        }
        CUDACHECK(cudaDeviceSynchronize());
        // Red points
        RedSORKernel<<<gridSOR, blockSOR>>>(unified_p, unified_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        // Black points
        BlackSORKernel<<<gridSOR, blockSOR>>>(unified_p, unified_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        // Calcular resíduos
        CalculateResidualKernel<<<gridSOR, blockSOR>>>(unified_p, unified_res, unified_RHS, i_max, j_max, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        // Verificar convergência (CPU reduction)
        double res_norm = 0.0;
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                res_norm += unified_res[i * (j_max + 2) + j] * unified_res[i * (j_max + 2) + j];
            }
        }
        res_norm = sqrt(res_norm / (i_max * j_max));
        if (res_norm <= eps * (norm_p + 0.01)) {
            break; // Convergência atingida
        }
        it++;
    }
    // Copiar resultados de volta para os arrays 2D originais
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            p[i][j] = unified_p[i * (j_max + 2) + j];
        }
    }
    printf("SOR complete!\n");
    // 4. Atualizar u e v usando kernel update_uv_kernel
    update_uv_kernel<<<grid2D, block2D>>>(unified_u, unified_v, unified_F, unified_G, unified_p, i_max, j_max, delta_t, delta_x, delta_y);
    KERNEL_CHECK("update_uv_kernel");
    CUDACHECK(cudaDeviceSynchronize());
    
    // Não há mais cudaMalloc/cudaFree aqui!
    // Corrigir acesso incorreto para os valores centrais
    int center_i = i_max/2;
    int center_j = j_max/2;
    
    // Verificar limites antes de acessar valores centrais
    if (center_i < 0 || center_i > i_max+1 || center_j < 0 || center_j > j_max+1) {
        return -1;
    }
    
    int center_idx = center_i * (j_max + 2) + center_j;
    
    // Output central values for debugging
    printf("TIMESTEP: %d TIME: %.6f\n", (*n_out), *t);
    printf("U-CENTER: %.6f\n", unified_u[center_idx]);
    printf("V-CENTER: %.6f\n", unified_v[center_idx]);
    printf("P-CENTER: %.6f\n", unified_p[center_idx]);
    
    (*n_out)++;  // Incrementa o valor apontado pelo ponteiro
    *t += delta_t;  // Atualiza o tempo

    return (it < max_it) ? 0 : -1;
}

// IMPORTANTE: Versão template removida para evitar conflito com __double_as_int
// Usamos apenas a versão especializada para double abaixo

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

// Kernel para calcular dp_dx e dp_dy em arrays de saída
__global__ void dp_dx_kernel(const double* p, double* out, int i_max, int j_max, double delta_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= i_max && j <= j_max) {
        out[i * (j_max + 2) + j] = (p[i * (j_max + 2) + j+1] - p[i * (j_max + 2) + j]) / delta_x;
    }
}
__global__ void dp_dy_kernel(const double* p, double* out, int i_max, int j_max, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= i_max && j <= j_max) {
        out[i * (j_max + 2) + j] = (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_y;
    }
}

// Kernel para set_noslip_linear
__global__ void set_noslip_linear_kernel(int i_max, int j_max, double* u, double* v, int side) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (side == LEFT && idx <= j_max) {
        u[0 * (j_max + 2) + idx] = 0.0;
        v[0 * (j_max + 2) + idx] = -v[1 * (j_max + 2) + idx];
    } else if (side == RIGHT && idx <= j_max) {
        u[i_max * (j_max + 2) + idx] = 0.0;
        v[(i_max+1) * (j_max + 2) + idx] = -v[i_max * (j_max + 2) + idx];
    } else if (side == TOP && idx <= i_max) {
        u[idx * (j_max + 2) + j_max+1] = -u[idx * (j_max + 2) + j_max];
        v[idx * (j_max + 2) + j_max] = 0.0;
    } else if (side == BOTTOM && idx <= i_max) {
        u[idx * (j_max + 2) + 0] = -u[idx * (j_max + 2) + 1];
        v[idx * (j_max + 2) + 0] = 0.0;
    }
}

// Kernel para set_inflow_linear
__global__ void set_inflow_linear_kernel(int i_max, int j_max, double* u, double* v, int side, double u_fix, double v_fix) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (side == TOP && idx <= i_max) {
        u[idx * (j_max + 2) + j_max+1] = 2 * u_fix - u[idx * (j_max + 2) + j_max];
        v[idx * (j_max + 2) + j_max] = v_fix;
    } else if (side == BOTTOM && idx <= i_max) {
        u[idx * (j_max + 2) + 0] = 2 * u_fix - u[idx * (j_max + 2) + 1];
        v[idx * (j_max + 2) + 0] = v_fix;
    } else if (side == LEFT && idx <= j_max) {
        u[0 * (j_max + 2) + idx] = u_fix;
        v[0 * (j_max + 2) + idx] = 2 * v_fix - v[1 * (j_max + 2) + idx];
    } else if (side == RIGHT && idx <= j_max) {
        u[i_max * (j_max + 2) + idx] = u_fix;
        v[(i_max+1) * (j_max + 2) + idx] = 2 * v_fix - v[i_max * (j_max + 2) + idx];
    }
}

// Kernel para calcular F e G (Navier-Stokes)
__global__ void FG_linear_kernel(double* u, double* v, double* F, double* G, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma_factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= i_max - 1 && j <= j_max) {
        // F (u preditor)
        double du2dx = ((u[(i+1)*(j_max+2)+j] + u[i*(j_max+2)+j])*(u[(i+1)*(j_max+2)+j] + u[i*(j_max+2)+j])
                        - (u[i*(j_max+2)+j] + u[(i-1)*(j_max+2)+j])*(u[i*(j_max+2)+j] + u[(i-1)*(j_max+2)+j])) / (4.0*delta_x);
        double duvdy = ((v[i*(j_max+2)+j] + v[(i+1)*(j_max+2)+j])*(u[i*(j_max+2)+j+1] + u[i*(j_max+2)+j])
                        - (v[i*(j_max+2)+j-1] + v[(i+1)*(j_max+2)+j-1])*(u[i*(j_max+2)+j] + u[i*(j_max+2)+j-1])) / (4.0*delta_y);
        double laplu = (u[(i+1)*(j_max+2)+j] - 2.0*u[i*(j_max+2)+j] + u[(i-1)*(j_max+2)+j]) / (delta_x*delta_x)
                        + (u[i*(j_max+2)+j+1] - 2.0*u[i*(j_max+2)+j] + u[i*(j_max+2)+j-1]) / (delta_y*delta_y);
        F[i*(j_max+2)+j] = u[i*(j_max+2)+j] + delta_t * ((laplu/Re) - du2dx - duvdy + g_x);
    }
    if (i <= i_max && j <= j_max - 1) {
        // G (v preditor)
        double dv2dy = ((v[i*(j_max+2)+j+1] + v[i*(j_max+2)+j])*(v[i*(j_max+2)+j+1] + v[i*(j_max+2)+j])
                        - (v[i*(j_max+2)+j] + v[i*(j_max+2)+j-1])*(v[i*(j_max+2)+j] + v[i*(j_max+2)+j-1])) / (4.0*delta_y);
        double duvdx = ((u[i*(j_max+2)+j] + u[i*(j_max+2)+j+1])*(v[(i+1)*(j_max+2)+j] + v[i*(j_max+2)+j])
                        - (u[(i-1)*(j_max+2)+j] + u[(i-1)*(j_max+2)+j+1])*(v[i*(j_max+2)+j] + v[(i-1)*(j_max+2)+j])) / (4.0*delta_x);
        double laplv = (v[(i+1)*(j_max+2)+j] - 2.0*v[i*(j_max+2)+j] + v[(i-1)*(j_max+2)+j]) / (delta_x*delta_x)
                        + (v[i*(j_max+2)+j+1] - 2.0*v[i*(j_max+2)+j] + v[i*(j_max+2)+j-1]) / (delta_y*delta_y);
        G[i*(j_max+2)+j] = v[i*(j_max+2)+j] + delta_t * ((laplv/Re) - dv2dy - duvdx + g_y);
    }
}

// Kernel para calcular RHS
__global__ void RHS_kernel(double* F, double* G, double* RHS, int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= i_max && j <= j_max) {
        RHS[i*(j_max+2)+j] = ((F[i*(j_max+2)+j] - F[(i-1)*(j_max+2)+j]) / delta_x
                            + (G[i*(j_max+2)+j] - G[i*(j_max+2)+j-1]) / delta_y) / delta_t;
    }
}

// Kernel para atualizar u e v após SOR
__global__ void update_uv_kernel(double* u, double* v, double* F, double* G, double* p, int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= i_max - 1 && j <= j_max) {
        u[i*(j_max+2)+j] = F[i*(j_max+2)+j] - delta_t * (p[(i+1)*(j_max+2)+j] - p[i*(j_max+2)+j]) / delta_x;
    }
    if (i <= i_max && j <= j_max - 1) {
        v[i*(j_max+2)+j] = G[i*(j_max+2)+j] - delta_t * (p[i*(j_max+2)+j+1] - p[i*(j_max+2)+j]) / delta_y;
    }
}

// Função utilitária para mínimo de até 4 valores double
__host__ __device__ double n_min(int n, double a, double b, double c, double d) {
    double minval = a;
    if (b < minval) minval = b;
    if (c < minval) minval = c;
    if (d < minval) minval = d;
    return minval;
}

// Exemplo de chamada de kernel para max_mat:
// double* d_max;
// cudaMallocManaged(&d_max, sizeof(double));
// *d_max = 0.0;
// max_mat_kernel<<<numBlocks, blockSize>>>(unified_u, i_max, j_max, d_max);
// cudaDeviceSynchronize();
// double max_val = *d_max;
// cudaFree(d_max);
//
// Para os outros kernels, use grid/block adequados conforme o tamanho do domínio.
