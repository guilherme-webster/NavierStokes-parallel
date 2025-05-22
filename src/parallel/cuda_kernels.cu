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
    
    return 0;
}
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
}

int cudaSOR(double** p,double** u,double** v, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it, double** F, double** G, double tau, double Re,
            int problem, double f, double* t, int* n_out, double g_x, double g_y) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = 0.0;
    
    // Allocate device memory for reduction and temporary values
    double *d_norm_p, *d_res_norm;
    double *d_center_values, h_center_values[3]; // For u, v, p values at center
    
    CUDACHECK(cudaMalloc(&d_norm_p, sizeof(double)));
    CUDACHECK(cudaMalloc(&d_res_norm, sizeof(double)));
    CUDACHECK(cudaMalloc(&d_center_values, 3 * sizeof(double)));
    
    if (!CHECK_POINTER(d_norm_p, sizeof(double), "d_norm_p") || 
        !CHECK_POINTER(d_res_norm, sizeof(double), "d_res_norm") ||
        !CHECK_POINTER(d_center_values, 3 * sizeof(double), "d_center_values")) {
        return -1;
    }
    
    // Calculate initial pressure norm using GPU
    CUDACHECK(cudaMemset(d_norm_p, 0, sizeof(double)));
    int num_threads = 256;
    int num_blocks = (i_max + num_threads - 1) / num_threads;
    calculate_pressure_norm_kernel<<<num_blocks, num_threads, num_threads * sizeof(double)>>>(
        device_p, d_norm_p, i_max, j_max);
    KERNEL_CHECK("calculate_pressure_norm_kernel");
    
    // Get result back to CPU
    double h_norm_p;
    CUDACHECK(cudaMemcpy(&h_norm_p, d_norm_p, sizeof(double), cudaMemcpyDeviceToHost));
    norm_p = sqrt(h_norm_p / (i_max * j_max));
    
    // 1. Calculate u_max and v_max using kernel max_mat_kernel
    double *d_umax, *d_vmax, h_umax = 0.0, h_vmax = 0.0;
    CUDACHECK(cudaMalloc(&d_umax, sizeof(double)));
    CUDACHECK(cudaMalloc(&d_vmax, sizeof(double)));
    
    if (!CHECK_POINTER(d_umax, sizeof(double), "d_umax") || 
        !CHECK_POINTER(d_vmax, sizeof(double), "d_vmax")) {
        CUDACHECK(cudaFree(d_norm_p));
        CUDACHECK(cudaFree(d_res_norm));
        CUDACHECK(cudaFree(d_center_values));
        return -1;
    }
    
    // Initialize with zero
    CUDACHECK(cudaMemset(d_umax, 0, sizeof(double)));
    CUDACHECK(cudaMemset(d_vmax, 0, sizeof(double)));
    
    int max_blocks = 32; // Adjust according to the domain size
    int max_threads = 256;
    
    max_mat_kernel_double<<<max_blocks, max_threads>>>(device_u, i_max, j_max, d_umax);
    KERNEL_CHECK("max_mat_kernel_double (u)");
    
    max_mat_kernel_double<<<max_blocks, max_threads>>>(device_v, i_max, j_max, d_vmax);
    KERNEL_CHECK("max_mat_kernel_double (v)");
    
    // Copy results back to host
    CUDACHECK(cudaMemcpy(&h_umax, d_umax, sizeof(double), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(&h_vmax, d_vmax, sizeof(double), cudaMemcpyDeviceToHost));
    
    u_max = h_umax;
    v_max = h_vmax;
    
    CUDACHECK(cudaFree(d_umax));
    CUDACHECK(cudaFree(d_vmax));
    
    // Calculate delta_t and gamma_factor
    delta_t = tau * n_min(4, 3.0, Re / 2.0 / ( 1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y ), delta_x / fabs(u_max), delta_y / fabs(v_max));
    gamma_factor = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);
    
    
    // 1. Boundary conditions (GPU)
    dim3 block1D_j((j_max + 127) / 128); // for sides with j_max
    dim3 block1D_i((i_max + 127) / 128); // for sides with i_max
    int threads1D = 128;
    if (problem == 1){
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, device_u, device_v, LEFT);
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, device_u, device_v, RIGHT);
        set_noslip_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, device_u, device_v, BOTTOM);
        set_inflow_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, device_u, device_v, TOP, 1.0, 0.0);
    }
    else if (problem == 2){
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, device_u, device_v, LEFT);
        set_noslip_linear_kernel<<<block1D_j, threads1D>>>(i_max, j_max, device_u, device_v, RIGHT);
        set_noslip_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, device_u, device_v, BOTTOM);
        set_inflow_linear_kernel<<<block1D_i, threads1D>>>(i_max, j_max, device_u, device_v, TOP, sin(f*(*t)), 0.0);           
    }
    else {
        CUDACHECK(cudaFree(d_norm_p));
        CUDACHECK(cudaFree(d_res_norm));
        CUDACHECK(cudaFree(d_center_values));
        return -1;
    }
    CUDACHECK(cudaDeviceSynchronize());
    printf("Conditions set!\n");

    // 2. FG calculation (GPU)
    dim3 block2D(16, 16);
    dim3 grid2D((i_max+block2D.x-1)/block2D.x, (j_max+block2D.y-1)/block2D.y);
    FG_linear_kernel<<<grid2D, block2D>>>(device_u, device_v, device_F, device_G, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma_factor);
    CUDACHECK(cudaDeviceSynchronize());
    printf("F, G calculated!\n");

    // 3. RHS calculation (GPU)
    RHS_kernel<<<grid2D, block2D>>>(device_F, device_G, device_RHS, i_max, j_max, delta_t, delta_x, delta_y);
    CUDACHECK(cudaDeviceSynchronize());
    printf("RHS calculated!\n");

    // 4. SOR loop (GPU calculations)
    dim3 blockSOR(16, 16);
    dim3 gridSOR((i_max+blockSOR.x-1)/blockSOR.x, (j_max+blockSOR.y-1)/blockSOR.y);
    while (it < max_it) {
        // Update pressure boundary conditions (ghost cells) on GPU
        update_pressure_bounds_kernel<<<block1D_i, threads1D>>>(device_p, i_max, j_max);
        CUDACHECK(cudaDeviceSynchronize());
        
        // Red points
        RedSORKernel<<<gridSOR, blockSOR>>>(device_p, device_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Black points
        BlackSORKernel<<<gridSOR, blockSOR>>>(device_p, device_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Calculate residuals
        CalculateResidualKernel<<<gridSOR, blockSOR>>>(device_p, device_res, device_RHS, i_max, j_max, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Check convergence using GPU reduction
        CUDACHECK(cudaMemset(d_res_norm, 0, sizeof(double)));
        calculate_residual_norm_kernel<<<num_blocks, num_threads, num_threads * sizeof(double)>>>(
            device_res, d_res_norm, i_max, j_max);
        KERNEL_CHECK("calculate_residual_norm_kernel");
        
        // Get result back to CPU
        double h_res_norm;
        CUDACHECK(cudaMemcpy(&h_res_norm, d_res_norm, sizeof(double), cudaMemcpyDeviceToHost));
        double res_norm = sqrt(h_res_norm / (i_max * j_max));
        
        if (res_norm <= eps * (norm_p + 0.01)) {
            break; // Convergence achieved
        }
        it++;
    }
    printf("SOR complete!\n");
    
    // 4. Update u and v using update_uv_kernel
    update_uv_kernel<<<grid2D, block2D>>>(device_u, device_v, device_F, device_G, device_p, i_max, j_max, delta_t, delta_x, delta_y);
    KERNEL_CHECK("update_uv_kernel");
    CUDACHECK(cudaDeviceSynchronize());
    
    // Calculate center index
    int center_i = i_max/2;
    int center_j = j_max/2;
    
    // Check limits before accessing central values
    if (center_i < 0 || center_i > i_max+1 || center_j < 0 || center_j > j_max+1) {
        CUDACHECK(cudaFree(d_norm_p));
        CUDACHECK(cudaFree(d_res_norm));
        CUDACHECK(cudaFree(d_center_values));
        return -1;
    }
    
    int center_idx = center_i * (j_max + 2) + center_j;
    
    // Extract central values directly on GPU
    extract_value_kernel<<<1, 1>>>(device_u, center_idx, &d_center_values[0]);
    extract_value_kernel<<<1, 1>>>(device_v, center_idx, &d_center_values[1]);
    extract_value_kernel<<<1, 1>>>(device_p, center_idx, &d_center_values[2]);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDACHECK(cudaMemcpy(h_center_values, d_center_values, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Output central values for debugging
    printf("TIMESTEP: %d TIME: %.6f\n", (*n_out), *t);
    printf("U-CENTER: %.6f\n", h_center_values[0]);
    printf("V-CENTER: %.6f\n", h_center_values[1]);
    printf("P-CENTER: %.6f\n", h_center_values[2]);
    
    // Free temporary device memory
    CUDACHECK(cudaFree(d_norm_p));
    CUDACHECK(cudaFree(d_res_norm));
    CUDACHECK(cudaFree(d_center_values));
    
    (*n_out)++;  // Increment the value pointed to by the pointer
    *t += delta_t;  // Update time

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

// Kernel para atualizar bordas/ghost cells de pressão
__global__ void update_pressure_bounds_kernel(double* p, int i_max, int j_max) {
    // Thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    // Atualiza bordas em i
    if (idx <= i_max) {
        // Bottom boundary
        p[idx * (j_max + 2) + 0] = p[idx * (j_max + 2) + 1];
        // Top boundary
        p[idx * (j_max + 2) + (j_max + 1)] = p[idx * (j_max + 2) + j_max];
    }
    
    // Atualiza bordas em j
    if (idx <= j_max) {
        // Left boundary
        p[0 * (j_max + 2) + idx] = p[1 * (j_max + 2) + idx];
        // Right boundary
        p[(i_max + 1) * (j_max + 2) + idx] = p[i_max * (j_max + 2) + idx];
    }
}

// Kernel para redução de soma (para calcular norma)
__global__ void reduce_sum_kernel(double* input, double* output, int size) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Carregar dados da memória global para compartilhada
    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();
    
    // Realizar redução na memória compartilhada
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 escreve o resultado para a saída
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel para extrair valor de um ponto específico
__global__ void extract_value_kernel(double* array, int idx, double* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = array[idx];
    }
}

// Kernel para calcular norma de resíduo
__global__ void calculate_residual_norm_kernel(double* res, double* norm, int i_max, int j_max) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    double sum = 0.0;
    
    // Cada thread soma vários elementos
    if (i <= i_max) {
        for (int j = 1; j <= j_max; j++) {
            double val = res[i * (j_max + 2) + j];
            sum += val * val;
        }
    }
    
    // Carregar na memória compartilhada
    sdata[tid] = sum;
    __syncthreads();
    
    // Redução dentro do bloco
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 escreve o resultado parcial
    if (tid == 0) {
        atomicAdd(norm, sdata[0]);
    }
}

// Kernel para calcular a norma de pressão inicial
__global__ void calculate_pressure_norm_kernel(double* p, double* norm, int i_max, int j_max) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    double sum = 0.0;
    
    // Cada thread soma vários elementos
    if (i <= i_max) {
        for (int j = 1; j <= j_max; j++) {
            double val = p[i * (j_max + 2) + j];
            sum += val * val;
        }
    }
    
    // Carregar na memória compartilhada
    sdata[tid] = sum;
    __syncthreads();
    
    // Redução dentro do bloco
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 escreve o resultado parcial
    if (tid == 0) {
        atomicAdd(norm, sdata[0]);
    }
}
