#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>

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

// Variáveis globais para arrays unificados
double *unified_p = NULL;
double *unified_res = NULL;
double *unified_RHS = NULL;
double *unified_u = NULL;
double *unified_v = NULL;
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
int initCudaArrays(int i_max, int j_max) {
    // Salvar dimensões da grade
    grid_i_max = i_max;
    grid_j_max = j_max;
    
    // Calcular tamanho necessário
    cuda_array_size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    // Alocar memória unificada
    CUDACHECK(cudaMallocManaged(&unified_p, cuda_array_size));
    CUDACHECK(cudaMallocManaged(&unified_res, cuda_array_size));
    CUDACHECK(cudaMallocManaged(&unified_RHS, cuda_array_size));
    CUDACHECK(cudaMallocManaged(&unified_u, cuda_array_size));
    CUDACHECK(cudaMallocManaged(&unified_v, cuda_array_size));
    // Inicializar com zeros
    CUDACHECK(cudaMemset(unified_p, 0, cuda_array_size));
    CUDACHECK(cudaMemset(unified_res, 0, cuda_array_size));
    CUDACHECK(cudaMemset(unified_RHS, 0, cuda_array_size));
    CUDACHECK(cudaMemset(unified_u, 0, cuda_array_size));
    CUDACHECK(cudaMemset(unified_v, 0, cuda_array_size));
    // Prefetch para GPU
    int device = -1;
    CUDACHECK(cudaGetDevice(&device));
    CUDACHECK(cudaMemPrefetchAsync(unified_p, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_RHS, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_res, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_u, cuda_array_size, device, NULL));
    CUDACHECK(cudaMemPrefetchAsync(unified_v, cuda_array_size, device, NULL));
    return 0;
}

// Free CUDA arrays once at the end
void freeCudaArrays() {
    if (unified_p) CUDACHECK(cudaFree(unified_p));
    if (unified_res) CUDACHECK(cudaFree(unified_res));
    if (unified_RHS) CUDACHECK(cudaFree(unified_RHS));
    if (unified_u) CUDACHECK(cudaFree(unified_u));
    if (unified_v) CUDACHECK(cudaFree(unified_v));
    unified_p = NULL;
    unified_res = NULL;
    unified_RHS = NULL;
}

int cudaSOR(double** p,double** u,double** v, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it, double** F, double** G, double tau, double Re,
            int problem, double f, double* t, int* n_out, double g_x, double g_y) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = 0.0;
    for (int i = 1; i <= i_max; i++) {
        for(int j = 1; j <= j_max; j++) {
            norm_p += unified_p[i * (j_max + 2) + j] * unified_p[i * (j_max + 2) + j];
        }
    }
    norm_p = sqrt(norm_p / i_max / j_max);
    

    u_max = max_mat(i_max, j_max, unified_u);
    v_max = max_mat(i_max, j_max, unified_v);
    delta_t = tau * n_min(3, Re / 2.0 / (1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y), delta_x / fabs(u_max), delta_y / fabs(v_max));
    gamma_factor = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);
    
    if (problem == 1){
        set_noslip_linear(i_max, j_max, unified_u, unified_v, LEFT);
        set_noslip_linear(i_max, j_max, unified_u, unified_v, RIGHT);
        set_noslip_linear(i_max, j_max, unified_u, unified_v, BOTTOM);
        set_inflow_linear(i_max, j_max, unified_u, unified_v, TOP, 1.0, 0.0);
    }
    else if (problem == 2){
        set_noslip_linear(i_max, j_max, unified_u, unified_v, LEFT);
        set_noslip_linear(i_max, j_max, unified_u, unified_v, RIGHT);
        set_noslip_linear(i_max, j_max, unified_u, unified_v, BOTTOM);
        set_inflow_linear(i_max, j_max, unified_u, unified_v, TOP, sin(f*(*t)), 0.0);           
    }
    else {
        printf("Unknown problem type (see parameters.txt).\n");
        exit(EXIT_FAILURE);
    }

    printf("Conditions set!\n");

    FG_linear(F, G, unified_u, unified_v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma_factor);
    
    printf("F, G calculated!\n");
    for (int i = 1; i <= i_max; i++ ) {
        for (int j = 1; j <= j_max; j++) {
            RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + (G[i][j] - G[i][j-1])/delta_y);
        }
    }
    printf("RHS calculated!\n");
    // Copiar dados dos arrays 2D para os arrays unificados
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            unified_p[i * (j_max + 2) + j] = p[i][j];
            unified_RHS[i * (j_max + 2) + j] = RHS[i][j];
            unified_res[i * (j_max + 2) + j] = 0.0;
        }
    }
    
    // Grid e block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Executar iterações Red-Black SOR
    while (it < max_it) {
        // Atualizar condições de contorno
        for (int i = 1; i <= i_max; i++) {
            unified_p[i * (j_max + 2) + 0] = unified_p[i * (j_max + 2) + 1];
            unified_p[i * (j_max + 2) + (j_max + 1)] = unified_p[i * (j_max + 2) + j_max];
        }
        for (int j = 1; j <= j_max; j++) {
            unified_p[0 * (j_max + 2) + j] = unified_p[1 * (j_max + 2) + j];
            unified_p[(i_max + 1) * (j_max + 2) + j] = unified_p[i_max * (j_max + 2) + j];
        }
        
        // Sincronizar antes de computação
        CUDACHECK(cudaDeviceSynchronize());
        
        // Red points
        RedSORKernel<<<gridSize, blockSize>>>(unified_p, unified_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Black points
        BlackSORKernel<<<gridSize, blockSize>>>(unified_p, unified_RHS, i_max, j_max, omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Calcular resíduos
        CalculateResidualKernel<<<gridSize, blockSize>>>(unified_p, unified_res, unified_RHS, i_max, j_max, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());
        
        // Verificar convergência
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
    for(int i = 1; i <= i_max; i++ ) {
        for (int j = 1; j <= j_max; j++) {
            if (i <= i_max - 1) {
                // Acesso correto ao array 1D linearizado
                unified_u[i * (j_max + 2) + j] = F[i][j] - delta_t * dp_dx(unified_p, i, j,j_max, delta_x);
            }
            if (j <= j_max - 1) {
                // Acesso correto ao array 1D linearizado
                unified_v[i * (j_max + 2) + j] = G[i][j] - delta_t * dp_dy(unified_p, i, j, j_max, delta_y);
            }
        }
    }

    // Corrigir acesso incorreto para os valores centrais
    printf("TIMESTEP: %d TIME: %.6f\n", (*n_out), *t);
    printf("U-CENTER: %.6f\n", unified_u[(i_max/2) * (j_max + 2) + (j_max/2)]);
    printf("V-CENTER: %.6f\n", unified_v[(i_max/2) * (j_max + 2) + (j_max/2)]);
    printf("P-CENTER: %.6f\n", unified_p[(i_max/2) * (j_max + 2) + (j_max/2)]);
    (*n_out)++;  // Incrementa o valor apontado pelo ponteiro
    *t += delta_t;  // Atualiza o tempo

    return (it < max_it) ? 0 : -1;
}


double max_mat(int i_max, int j_max, double* mat) {
    double max_val = 0.0;
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            if (fabs(mat[i * (j_max + 2) + j]) > max_val) {
                max_val = fabs(mat[i * (j_max + 2) + j]);
            }
        }
    }
    return max_val;
}

double n_min (int count, ...) {
    va_list args;
    va_start(args, count);
    
    double min_val = va_arg(args, double);
    
    for (int i = 1; i < count; i++) {
        double val = va_arg(args, double);
        if (val < min_val) {
            min_val = val;
        }
    }
    
    va_end(args);
    return min_val;
}
double dp_dx(double* p, int i, int j, int j_max, double delta_x) {
    return (p[i * (j_max + 2) + j+1] - p[i * (j_max + 2) + j]) / delta_x;
}

double dp_dy(double* p, int i, int j, int j_max, double delta_y) {
    return (p[(i+1) * (j_max + 2) + j] - p[i * (j_max + 2) + j]) / delta_y;
}
// Adicione novas funções que lidam com arrays linearizados
int set_noslip_linear(int i_max, int j_max, double* u, double* v, int side) {
    return set_inflow_linear(i_max, j_max, u, v, side, 0.0, 0.0);
}

int set_inflow_linear(int i_max, int j_max, double* u, double* v, int side, double u_fix, double v_fix) {
    int i, j;
    switch(side) {
        case TOP:            
            for (i = 1; i <= i_max; i++) {
                v[i * (j_max + 2) + j_max] = v_fix; // Set fixed values on border
                u[i * (j_max + 2) + (j_max + 1)] = 2 * u_fix - u[i * (j_max + 2) + j_max]; // Set values by averaging
            }
            break;
        case BOTTOM:           
            for (i = 1; i <= i_max; i++) {
                v[i * (j_max + 2) + 0] = v_fix;
                u[i * (j_max + 2) + 0] = 2 * u_fix - u[i * (j_max + 2) + 1];
            }
            break;
        case LEFT:        
            for (j = 1; j <= j_max; j++) {
                u[0 * (j_max + 2) + j] = u_fix;
                v[0 * (j_max + 2) + j] = 2 * v_fix - v[1 * (j_max + 2) + j];
            }
            break;
        case RIGHT:            
            for (j = 1; j <= j_max; j++) {
                u[i_max * (j_max + 2) + j] = u_fix;
                v[(i_max+1) * (j_max + 2) + j] = 2 * v_fix - v[i_max * (j_max + 2) + j];
            }
            break;
        default: 
            return -1;
    }

    return 0;
}


// Implementação unificada de FG que trabalha diretamente com arrays lineares
void FG_linear(double** F, double** G, double* u, double* v, int i_max, int j_max, 
       double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    
    // Calcular F e G diretamente acessando os arrays linearizados
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            // Calcular F em (i,j)
            if (i <= i_max-1) {
                // Termo convectivo em x (donor-cell)
                double du2_dx = 0.0;
                if (u[i * (j_max + 2) + j] > 0) {
                    du2_dx = ((u[i * (j_max + 2) + j] * u[i * (j_max + 2) + j] - 
                              u[(i-1) * (j_max + 2) + j] * u[(i-1) * (j_max + 2) + j]) / delta_x) *
                             (1 - gamma) + 
                             (u[i * (j_max + 2) + j] * (u[i * (j_max + 2) + j] - u[(i-1) * (j_max + 2) + j]) / delta_x) *
                             gamma;
                } else {
                    du2_dx = ((u[(i+1) * (j_max + 2) + j] * u[(i+1) * (j_max + 2) + j] - 
                              u[i * (j_max + 2) + j] * u[i * (j_max + 2) + j]) / delta_x) *
                             (1 - gamma) + 
                             (u[i * (j_max + 2) + j] * (u[(i+1) * (j_max + 2) + j] - u[i * (j_max + 2) + j]) / delta_x) *
                             gamma;
                }
                
                // Termo convectivo em y
                double duv_dy = 0.0;
                if ((v[(i) * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) / 2.0 > 0) {
                    duv_dy = ((v[i * (j_max + 2) + j] * u[i * (j_max + 2) + j] - 
                              v[i * (j_max + 2) + (j-1)] * u[i * (j_max + 2) + (j-1)]) / delta_y) *
                             (1 - gamma) + 
                             ((v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) / 2.0) * 
                             (u[i * (j_max + 2) + j] - u[i * (j_max + 2) + (j-1)]) / delta_y * 
                             gamma;
                } else {
                    duv_dy = ((v[i * (j_max + 2) + (j+1)] * u[i * (j_max + 2) + (j+1)] - 
                              v[i * (j_max + 2) + j] * u[i * (j_max + 2) + j]) / delta_y) *
                             (1 - gamma) + 
                             ((v[i * (j_max + 2) + j] + v[(i+1) * (j_max + 2) + j]) / 2.0) * 
                             (u[i * (j_max + 2) + (j+1)] - u[i * (j_max + 2) + j]) / delta_y * 
                             gamma;
                }
                
                // Termos difusivos
                double d2u_dx2 = (u[(i+1) * (j_max + 2) + j] - 2.0 * u[i * (j_max + 2) + j] + u[(i-1) * (j_max + 2) + j]) / (delta_x * delta_x);
                double d2u_dy2 = (u[i * (j_max + 2) + (j+1)] - 2.0 * u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j-1)]) / (delta_y * delta_y);
                
                // F[i][j] no formato 2D normal
                F[i][j] = u[i * (j_max + 2) + j] + delta_t * (
                              1.0/Re * (d2u_dx2 + d2u_dy2) - du2_dx - duv_dy + g_x
                          );
            }
            
            // Calcular G em (i,j)
            if (j <= j_max-1) {
                // Implementação similar para G
                // Termo convectivo em x
                double duv_dx = 0.0;
                if ((u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) / 2.0 > 0) {
                    duv_dx = ((u[i * (j_max + 2) + j] * v[i * (j_max + 2) + j] - 
                              u[(i-1) * (j_max + 2) + j] * v[(i-1) * (j_max + 2) + j]) / delta_x) *
                             (1 - gamma) + 
                             ((u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) / 2.0) * 
                             (v[i * (j_max + 2) + j] - v[(i-1) * (j_max + 2) + j]) / delta_x * 
                             gamma;
                } else {
                    duv_dx = ((u[(i+1) * (j_max + 2) + j] * v[(i+1) * (j_max + 2) + j] - 
                              u[i * (j_max + 2) + j] * v[i * (j_max + 2) + j]) / delta_x) *
                             (1 - gamma) + 
                             ((u[i * (j_max + 2) + j] + u[i * (j_max + 2) + (j+1)]) / 2.0) * 
                             (v[(i+1) * (j_max + 2) + j] - v[i * (j_max + 2) + j]) / delta_x * 
                             gamma;
                }
                
                // Termo convectivo em y
                double dv2_dy = 0.0;
                if (v[i * (j_max + 2) + j] > 0) {
                    dv2_dy = ((v[i * (j_max + 2) + j] * v[i * (j_max + 2) + j] - 
                              v[i * (j_max + 2) + (j-1)] * v[i * (j_max + 2) + (j-1)]) / delta_y) *
                             (1 - gamma) + 
                             (v[i * (j_max + 2) + j] * (v[i * (j_max + 2) + j] - v[i * (j_max + 2) + (j-1)]) / delta_y) *
                             gamma;
                } else {
                    dv2_dy = ((v[i * (j_max + 2) + (j+1)] * v[i * (j_max + 2) + (j+1)] - 
                              v[i * (j_max + 2) + j] * v[i * (j_max + 2) + j]) / delta_y) *
                             (1 - gamma) + 
                             (v[i * (j_max + 2) + j] * (v[i * (j_max + 2) + (j+1)] - v[i * (j_max + 2) + j]) / delta_y) *
                             gamma;
                }
                
                // Termos difusivos
                double d2v_dx2 = (v[(i+1) * (j_max + 2) + j] - 2.0 * v[i * (j_max + 2) + j] + v[(i-1) * (j_max + 2) + j]) / (delta_x * delta_x);
                double d2v_dy2 = (v[i * (j_max + 2) + (j+1)] - 2.0 * v[i * (j_max + 2) + j] + v[i * (j_max + 2) + (j-1)]) / (delta_y * delta_y);
                
                // G[i][j] no formato 2D normal
                G[i][j] = v[i * (j_max + 2) + j] + delta_t * (
                              1.0/Re * (d2v_dx2 + d2v_dy2) - duv_dx - dv2_dy + g_y
                          );
            }
        }
    }
}
