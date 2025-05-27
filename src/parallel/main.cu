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
#include "integration.h"
#include "boundaries.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cmath>


typedef struct{
    int i;
    int j;
    int position;
} BoundaryPoint;


// Macro para verificar erros CUDA
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Função para alocar memória usando UVA (Unified Virtual Addressing)
// Adicionar num_border_points como parâmetro
int allocate_unified_memory(double ***u, double ***v, double ***p, double ***res, double ***RHS, double ***F, double ***G, int i_max, int j_max, BoundaryPoint **borders, int num_border_points) {
    int rows = i_max + 2;
    int cols = j_max + 2;
    
    // Alocar arrays de ponteiros para linhas
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)u, rows * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)v, rows * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)p, rows * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)res, rows * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)RHS, rows * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)F, rows * sizeof(double*)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)G, rows * sizeof(double*)));
    // Alocar o número exato de pontos de borda usando num_border_points
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)borders, num_border_points * sizeof(BoundaryPoint)));

    double *u_data, *v_data, *p_data, *res_data, *RHS_data, *F_data, *G_data;
    
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&u_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&v_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&p_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&res_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&RHS_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&F_data, rows * cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&G_data, rows * cols * sizeof(double)));
    // Configurar ponteiros de linha para apontar para a memória contígua
    for (int i = 0; i < rows; i++) {
        (*u)[i] = &u_data[i * cols];
        (*v)[i] = &v_data[i * cols];
        (*p)[i] = &p_data[i * cols];
        (*res)[i] = &res_data[i * cols];
        (*RHS)[i] = &RHS_data[i * cols];
        (*F)[i] = &F_data[i * cols];
        (*G)[i] = &G_data[i * cols];
    }
    
    // Inicializar com zeros
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*u)[i][j] = 0.0;
            (*v)[i][j] = 0.0;
            (*p)[i][j] = 0.0;
            (*res)[i][j] = 0.0;
            (*RHS)[i][j] = 0.0;
            (*F)[i][j] = 0.0;
            (*G)[i][j] = 0.0;
        }
    }
    
    return 0;
}

// Função para liberar memória UVA
void free_unified_memory(double **u, double **v, double **p, double **res, double **RHS, double **F, double **G, 
                         BoundaryPoint *borders) {
    if (u) {
        cudaFree(u[0]); // Libera dados contíguos
        cudaFree(u);    // Libera array de ponteiros
    }
    if (v) {
        cudaFree(v[0]);
        cudaFree(v);
    }
    if (p) {
        cudaFree(p[0]);
        cudaFree(p);
    }
    if (res) {
        cudaFree(res[0]);
        cudaFree(res);
    }
    if (RHS) {
        cudaFree(RHS[0]);
        cudaFree(RHS);
    }
    if (F) {
        cudaFree(F[0]);
        cudaFree(F);
    }
    if (G) {
        cudaFree(G[0]);
        cudaFree(G);
    }
    if (borders) {
        cudaFree(borders);
    }
}


void precalculate_borders(int i_max, int j_max, BoundaryPoint *borders_ptr) { // Nome do parâmetro alterado para clareza local
    int index = 0;
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            if (i == 0 || i == i_max + 1 || j == 0 || j == j_max + 1) {
                borders_ptr[index].i = i;
                borders_ptr[index].j = j;
                borders_ptr[index].position = (i == 0) ? LEFT : (i == i_max + 1) ? RIGHT : (j == 0) ? BOTTOM : TOP;
                index++;
            }
        }
    }
}


// Kernels CUDA que podem acessar diretamente as matrizes 2D
__global__ void calculate_RHS_kernel(double **RHS, double **F, double **G, 
                                   int i_max, int j_max, double delta_t, 
                                   double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + 
                                     (G[i][j] - G[i][j-1])/delta_y);
    }
}

__global__ void update_velocities_kernel(double **u, double **v, double **F, double **G, double **p,
                                        int i_max, int j_max, double delta_t, 
                                        double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        if (i <= i_max - 1) {
            u[i][j] = F[i][j] - delta_t * (p[i+1][j] - p[i][j]) / delta_x;
        }
        if (j <= j_max - 1) {
            v[i][j] = G[i][j] - delta_t * (p[i][j+1] - p[i][j]) / delta_y;
        }
    }
}

// SOR kernel usando UVA - versão Red-Black simplificada
__global__ void sor_red_kernel_uva(double **p, double **RHS, 
                                  int i_max, int j_max, double delta_x, double delta_y, 
                                  double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max && (i + j) % 2 == 0) {
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        double coeff = 2.0 * (1.0/dx2 + 1.0/dy2);
        
        double p_old = p[i][j];
        p[i][j] = (1.0 - omega) * p_old + 
                  omega / coeff * 
                  ((p[i+1][j] + p[i-1][j]) / dx2 +
                   (p[i][j+1] + p[i][j-1]) / dy2 -
                   RHS[i][j]);
    }
}

__global__ void sor_black_kernel_uva(double **p, double **RHS, 
                                    int i_max, int j_max, double delta_x, double delta_y, 
                                    double omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        double coeff = 2.0 * (1.0/dx2 + 1.0/dy2);
        
        double p_old = p[i][j];
        p[i][j] = (1.0 - omega) * p_old + 
                  omega / coeff * 
                  ((p[i+1][j] + p[i-1][j]) / dx2 +
                   (p[i][j+1] + p[i][j-1]) / dy2 -
                   RHS[i][j]);
    }
}

// Kernel otimizado para atualizar bordas usando pontos pré-calculados
__global__ void update_boundaries_with_precalc_kernel(double **p, BoundaryPoint *borders, int border_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < border_count) {
        int i = borders[idx].i;
        int j = borders[idx].j;
        int position = borders[idx].position;
        
        // Aplicar condição de Neumann apropriada baseada na posição
        switch (position) {
            case LEFT:
                p[i][j] = p[i+1][j];  // Copia do vizinho à direita
                break;
            case RIGHT:
                p[i][j] = p[i-1][j];  // Copia do vizinho à esquerda
                break;
            case BOTTOM:
                p[i][j] = p[i][j+1];  // Copia do vizinho acima
                break;
            case TOP:
                p[i][j] = p[i][j-1];  // Copia do vizinho abaixo
                break;
        }
    }
}


// Kernel para calcular o resíduo da equação de Poisson: L(p) - RHS
__global__ void calculate_poisson_residual_kernel(double **p, double **RHS, double **res,
                                                int i_max, int j_max, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i <= i_max && j <= j_max) {
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;

        res[i][j] = (p[i+1][j] - 2.0 * p[i][j] + p[i-1][j]) / dx2 +
                    (p[i][j+1] - 2.0 * p[i][j] + p[i][j-1]) / dy2 -
                    RHS[i][j];
    }
}

// Função auxiliar no host para calcular a norma L2 de uma matriz UVA
double calculate_L2_norm_host_uva(double **matrix, int i_max, int j_max) {
    double norm_sq_sum = 0.0;
    if (i_max == 0 || j_max == 0) return 0.0;

    for (int r = 1; r <= i_max; r++) {
        for (int c = 1; c <= j_max; c++) {
            norm_sq_sum += matrix[r][c] * matrix[r][c];
        }
    }
    return sqrt(norm_sq_sum / (i_max * j_max));
}

// Device versions of differential functions
__device__ double du2_dx_device(double** u, double** v, int i, int j, double delta_x, double gamma) {
    double stencil1 = 0.5 * (u[i][j] + u[i+1][j]);
    double stencil2 = 0.5 * (u[i-1][j] + u[i][j]);

    double stencil3 = fabs(stencil1) * 0.5 * (u[i][j] - u[i+1][j]);
    double stencil4 = fabs(stencil2) * 0.5 * (u[i-1][j] - u[i][j]);

    return (1.0/delta_x) * (stencil1*stencil1 - stencil2*stencil2) + (gamma / delta_x) * (stencil3 - stencil4);
}

__device__ double duv_dy_device(double** u, double** v, int i, int j, double delta_y, double gamma) {
    double stencil1 = 0.5 * (v[i][j] + v[i+1][j]);
    double stencil2 = 0.5 * (v[i][j-1] + v[i+1][j-1]);

    double stencil3 = stencil1 * 0.5 * (u[i][j] + u[i][j+1]);
    double stencil4 = stencil2 * 0.5 * (u[i][j-1] + u[i][j]);

    double stencil5 = fabs(stencil1) * 0.5 * (u[i][j] - u[i][j+1]);
    double stencil6 = fabs(stencil2) * 0.5 * (u[i][j-1] - u[i][j]);

    return (1.0/delta_y) * (stencil3 - stencil4) + (gamma / delta_y) * (stencil5 - stencil6);
}

__device__ double dv2_dy_device(double** u, double** v, int i, int j, double delta_y, double gamma) {
    double stencil1 = 0.5 * (v[i][j] + v[i][j+1]);
    double stencil2 = 0.5 * (v[i][j-1] + v[i][j]);

    double stencil3 = fabs(stencil1) * 0.5 * (v[i][j] - v[i][j+1]);
    double stencil4 = fabs(stencil2) * 0.5 * (v[i][j-1] - v[i][j]);

    return (1.0/delta_y) * (stencil1*stencil1 - stencil2*stencil2) + (gamma / delta_y) * (stencil3 - stencil4);
}

__device__ double duv_dx_device(double** u, double** v, int i, int j, double delta_x, double gamma) {
    double stencil1 = 0.5 * (u[i][j] + u[i][j+1]);
    double stencil2 = 0.5 * (u[i-1][j] + u[i-1][j+1]);

    double stencil3 = stencil1 * 0.5 * (v[i][j] + v[i+1][j]);
    double stencil4 = stencil2 * 0.5 * (v[i-1][j] + v[i][j]);

    double stencil5 = fabs(stencil1) * 0.5 * (v[i][j] - v[i+1][j]);
    double stencil6 = fabs(stencil2) * 0.5 * (v[i-1][j] - v[i][j]);

    return (1.0/delta_x) * (stencil3 - stencil4) + (gamma / delta_x) * (stencil5 - stencil6);
}

// Central differences for second derivatives
__device__ double d2u_dx2_device(double** u, int i, int j, double delta_x) {
    return (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / (delta_x * delta_x);
}

__device__ double d2u_dy2_device(double** u, int i, int j, double delta_y) {
    return (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / (delta_y * delta_y);
}

__device__ double d2v_dx2_device(double** v, int i, int j, double delta_x) {
    return (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) / (delta_x * delta_x);
}

__device__ double d2v_dy2_device(double** v, int i, int j, double delta_y) {
    return (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) / (delta_y * delta_y);
}

__global__ void calculate_F_kernel(double **F, double **u, double **v, int i_max, int j_max, 
                                  double Re, double g_x, double delta_t, double delta_x, 
                                  double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max-1 && j <= j_max) {  // F bounds: i from 1 to i_max-1
        F[i][j] = u[i][j] + delta_t * (
            (1.0/Re) * (d2u_dx2_device(u, i, j, delta_x) + d2u_dy2_device(u, i, j, delta_y)) 
            - du2_dx_device(u, v, i, j, delta_x, gamma) 
            - duv_dy_device(u, v, i, j, delta_y, gamma) 
            + g_x
        );
    }
}

__global__ void calculate_G_kernel(double **G, double **u, double **v, int i_max, int j_max, 
                                  double Re, double g_y, double delta_t, double delta_x, 
                                  double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max-1) {  // G bounds: j from 1 to j_max-1
        G[i][j] = v[i][j] + delta_t * (
            (1.0/Re) * (d2v_dx2_device(v, i, j, delta_x) + d2v_dy2_device(v, i, j, delta_y)) 
            - duv_dx_device(u, v, i, j, delta_x, gamma) 
            - dv2_dy_device(u, v, i, j, delta_y, gamma) 
            + g_y
        );
    }
}

__global__ void sor_shared_memory_kernel(double **p, double **RHS, 
                                        int i_max, int j_max, double delta_x, double delta_y, 
                                        double omega, int color) {
    const int BLOCK_SIZE = 16; // Assuming BLOCK_SIZE is 16 as used elsewhere
    const int SHARED_MEM_DIM_SIZE = BLOCK_SIZE + 2; // e.g., 18 for BLOCK_SIZE 16
    
    // p_shared needs to hold a (BLOCK_SIZE+2) x (BLOCK_SIZE+2) tile
    // RHS_shared only needs the BLOCK_SIZE x BLOCK_SIZE interior part
    __shared__ double p_shared[SHARED_MEM_DIM_SIZE][SHARED_MEM_DIM_SIZE + 1]; // +1 for padding against bank conflicts
    __shared__ double RHS_shared[BLOCK_SIZE][BLOCK_SIZE]; // Only interior RHS values are needed by the stencil

    int tx = threadIdx.x; // 0 to BLOCK_SIZE-1
    int ty = threadIdx.y; // 0 to BLOCK_SIZE-1

    // Global indices for the current thread's primary responsibility (center of its 3x3 stencil in shared mem)
    int current_i = blockIdx.x * BLOCK_SIZE + tx + 1; // 1 to i_max
    int current_j = blockIdx.y * BLOCK_SIZE + ty + 1; // 1 to j_max

    // --- Stage 1: Load data into shared memory ---

    // Each thread loads its corresponding p[current_i][current_j] into the center of its shared memory view
    // p_shared[tx+1][ty+1] corresponds to p[current_i][current_j]
    if (current_i >= 1 && current_i <= i_max && current_j >= 1 && current_j <= j_max) {
        p_shared[tx + 1][ty + 1] = p[current_i][current_j];
        RHS_shared[tx][ty] = RHS[current_i][current_j]; // RHS_shared is indexed 0..BLOCK_SIZE-1
    }

    // Load halo regions into p_shared
    // Global indices for p array are 0 to i_max+1 and 0 to j_max+1

    // Left halo: p_shared[0][ty+1]
    if (tx == 0) {
        int gi = blockIdx.x * BLOCK_SIZE; // Global i for the halo element p[gi][gj]
        int gj = blockIdx.y * BLOCK_SIZE + ty + 1;
        if (gi >= 0 && gi <= i_max + 1 && gj >= 1 && gj <= j_max) { // Check gj bounds carefully
            p_shared[0][ty + 1] = p[gi][gj];
        }
    }
    // Right halo: p_shared[BLOCK_SIZE+1][ty+1]
    if (tx == BLOCK_SIZE - 1) {
        int gi = blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + 1;
        int gj = blockIdx.y * BLOCK_SIZE + ty + 1;
        if (gi >= 0 && gi <= i_max + 1 && gj >= 1 && gj <= j_max) {
            p_shared[BLOCK_SIZE + 1][ty + 1] = p[gi][gj];
        }
    }
    // Top halo: p_shared[tx+1][0]
    if (ty == 0) {
        int gi = blockIdx.x * BLOCK_SIZE + tx + 1;
        int gj = blockIdx.y * BLOCK_SIZE;
        if (gi >= 1 && gi <= i_max && gj >= 0 && gj <= j_max + 1) { // Check gi bounds carefully
             p_shared[tx + 1][0] = p[gi][gj];
        }
    }
    // Bottom halo: p_shared[tx+1][BLOCK_SIZE+1]
    if (ty == BLOCK_SIZE - 1) {
        int gi = blockIdx.x * BLOCK_SIZE + tx + 1;
        int gj = blockIdx.y * BLOCK_SIZE + BLOCK_SIZE + 1;
        if (gi >= 1 && gi <= i_max && gj >= 0 && gj <= j_max + 1) {
            p_shared[tx + 1][BLOCK_SIZE + 1] = p[gi][gj];
        }
    }

    // Corner halos for p_shared
    // Top-Left: p_shared[0][0]
    if (tx == 0 && ty == 0) {
        int gi = blockIdx.x * BLOCK_SIZE;
        int gj = blockIdx.y * BLOCK_SIZE;
        if (gi >= 0 && gi <= i_max + 1 && gj >= 0 && gj <= j_max + 1) {
            p_shared[0][0] = p[gi][gj];
        }
    }
    // Top-Right: p_shared[BLOCK_SIZE+1][0]
    if (tx == BLOCK_SIZE - 1 && ty == 0) {
        int gi = blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + 1;
        int gj = blockIdx.y * BLOCK_SIZE;
        if (gi >= 0 && gi <= i_max + 1 && gj >= 0 && gj <= j_max + 1) {
            p_shared[BLOCK_SIZE + 1][0] = p[gi][gj];
        }
    }
    // Bottom-Left: p_shared[0][BLOCK_SIZE+1]
    if (tx == 0 && ty == BLOCK_SIZE - 1) {
        int gi = blockIdx.x * BLOCK_SIZE;
        int gj = blockIdx.y * BLOCK_SIZE + BLOCK_SIZE + 1;
        if (gi >= 0 && gi <= i_max + 1 && gj >= 0 && gj <= j_max + 1) {
            p_shared[0][BLOCK_SIZE + 1] = p[gi][gj];
        }
    }
    // Bottom-Right: p_shared[BLOCK_SIZE+1][BLOCK_SIZE+1]
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1) {
        int gi = blockIdx.x * BLOCK_SIZE + BLOCK_SIZE + 1;
        int gj = blockIdx.y * BLOCK_SIZE + BLOCK_SIZE + 1;
        if (gi >= 0 && gi <= i_max + 1 && gj >= 0 && gj <= j_max + 1) {
            p_shared[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = p[gi][gj];
        }
    }

    __syncthreads();

    // --- Stage 2: Perform SOR update using shared memory ---
    // Computation is for p[current_i][current_j]
    if (current_i >= 1 && current_i <= i_max && current_j >= 1 && current_j <= j_max && (current_i + current_j) % 2 == color) {
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        double coeff = 2.0 * (1.0/dx2 + 1.0/dy2);
        
        // Access p_shared using indices relative to the current thread's (tx,ty)
        // Center: p_shared[tx+1][ty+1]
        // Left:   p_shared[tx  ][ty+1]
        // Right:  p_shared[tx+2][ty+1]
        // Top:    p_shared[tx+1][ty  ]
        // Bottom: p_shared[tx+1][ty+2]
        double p_old = p_shared[tx + 1][ty + 1];
        double p_new = (1.0 - omega) * p_old + 
                        omega / coeff * 
                        ((p_shared[tx + 2][ty + 1] + p_shared[tx][ty + 1]) / dx2 +
                        (p_shared[tx + 1][ty + 2] + p_shared[tx + 1][ty]) / dy2 -
                        RHS_shared[tx][ty]); // Use the loaded interior RHS_shared

            // we are writing to the global mem, may affect performance
            p[current_i][current_j] = p_new;
        }
}

// Após o kernel sor_shared_memory_kernel

__global__ void calculate_residual_and_norm_kernel(double **p, double **RHS, 
                                                 int i_max, int j_max, 
                                                 double delta_x, double delta_y,
                                                 double *block_norms) {
    const int BLOCK_SIZE = 16;
    
    // Compartilhar valores do resíduo para blocos - usar para redução local
    __shared__ double res_shared[BLOCK_SIZE][BLOCK_SIZE+1]; // +1 para evitar conflitos de bancos
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Índices globais
    int i = blockIdx.x * BLOCK_SIZE + tx + 1;
    int j = blockIdx.y * BLOCK_SIZE + ty + 1;
    
    double dx2 = delta_x * delta_x;
    double dy2 = delta_y * delta_y;
    double res_squared = 0.0;
    
    // Calcular resíduo para cada ponto e armazenar em memória compartilhada
    if (i <= i_max && j <= j_max) {
        double residual = (p[i+1][j] - 2.0 * p[i][j] + p[i-1][j]) / dx2 +
                          (p[i][j+1] - 2.0 * p[i][j] + p[i][j-1]) / dy2 -
                          RHS[i][j];
        
        // Armazenar o quadrado do resíduo para posterior redução
        res_squared = residual * residual;
        res_shared[tx][ty] = res_squared;
    } else {
        res_shared[tx][ty] = 0.0;
    }
    
    __syncthreads();
    
    // Redução paralela dentro do bloco - somando todos os valores da norma ao quadrado
    // Redução em x
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            res_shared[tx][ty] += res_shared[tx + stride][ty];
        }
        __syncthreads();
    }
    
    // Redução em y - apenas threads com tx==0 participam
    if (tx == 0) {
        for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
            if (ty < stride) {
                res_shared[0][ty] += res_shared[0][ty + stride];
            }
            __syncthreads();
        }
        
        // Thread (0,0) salva o resultado final do bloco
        if (ty == 0) {
            block_norms[blockIdx.y * gridDim.x + blockIdx.x] = res_shared[0][0];
        }
    }
}

// Kernel para redução final de normas de blocos para um único valor
__global__ void reduce_block_norms_kernel(double *block_norms, int num_blocks, double *final_norm, int i_max, int j_max) {
    __shared__ double shared_data[256];
    
    int tid = threadIdx.x;
    
    // Carregar dados para memória compartilhada
    double sum = 0.0;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += block_norms[i];
    }
    shared_data[tid] = sum;
    
    __syncthreads();
    
    // Redução paralela
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 escreve o resultado final
    if (tid == 0) {
        *final_norm = sqrt(shared_data[0] / (i_max * j_max)); // Agora i_max e j_max são parâmetros
    }
}
int SOR_UVA_with_shared_memory(double **p, int i_max, int j_max, double delta_x, double delta_y,
                               double **res, double **RHS, double omega, double epsilon, int max_it,
                               BoundaryPoint *borders, int border_count) {
    
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((i_max + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (j_max + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    dim3 boundaryBlockDim(256); 
    dim3 boundaryGridDim((border_count + 255) / 256);
    
    // Número total de blocos para o cálculo da norma
    int total_blocks = gridDim.x * gridDim.y;
    
    // Alocação de memória para normas de blocos e norma final
    double *d_block_norms, *d_final_norm;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_norms, total_blocks * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_final_norm, sizeof(double)));
    
    double norm_p_initial = calculate_L2_norm_host_uva(p, i_max, j_max);
    double current_L2_res_norm;
    
    for (int it = 0; it < max_it; it++) {
        // Atualizar bordas
        update_boundaries_with_precalc_kernel<<<boundaryGridDim, boundaryBlockDim>>>(p, borders, border_count);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Pontos vermelhos com memória compartilhada
        sor_shared_memory_kernel<<<gridDim, blockDim>>>(p, RHS, i_max, j_max, delta_x, delta_y, omega, 0);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Atualizar bordas novamente
        update_boundaries_with_precalc_kernel<<<boundaryGridDim, boundaryBlockDim>>>(p, borders, border_count);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Pontos pretos com memória compartilhada
        sor_shared_memory_kernel<<<gridDim, blockDim>>>(p, RHS, i_max, j_max, delta_x, delta_y, omega, 1);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Verificação de convergência usando o novo método otimizado
        calculate_residual_and_norm_kernel<<<gridDim, blockDim>>>(p, RHS, i_max, j_max, delta_x, delta_y, d_block_norms);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Redução final das normas de blocos em um único valor
        reduce_block_norms_kernel<<<1, 256>>>(d_block_norms, total_blocks, d_final_norm, i_max, j_max);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Transferir apenas o valor final da norma para o host
        CHECK_CUDA_ERROR(cudaMemcpy(&current_L2_res_norm, d_final_norm, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Verificar convergência
        if (current_L2_res_norm <= epsilon * (norm_p_initial + 1.5)) {
            // Liberar memória e retornar
            cudaFree(d_block_norms);
            cudaFree(d_final_norm);
            return it + 1;
        }
    }
    
    // Liberar memória
    cudaFree(d_block_norms);
    cudaFree(d_final_norm);
    
    return -1; // Não convergiu
}


__global__ void find_max_kernel(double **matrix, int i_max, int j_max, double *block_max) {
    const int BLOCK_SIZE = 16;
    
    // Compartilhar valores máximos para blocos - usar para redução local
    __shared__ double max_shared[BLOCK_SIZE][BLOCK_SIZE+1]; // +1 para evitar conflitos de bancos
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Índices globais
    int i = blockIdx.x * BLOCK_SIZE + tx + 1;
    int j = blockIdx.y * BLOCK_SIZE + ty + 1;
    
    // Inicializar com valor mínimo
    double local_max = -1e30; // Valor negativo grande
    
    // Pegar o valor da matriz se estiver dentro dos limites
    if (i <= i_max && j <= j_max) {
        local_max = matrix[i][j];
    }
    
    // Armazenar valor local em memória compartilhada
    max_shared[tx][ty] = local_max;
    
    __syncthreads();
    
    // Redução paralela dentro do bloco - encontrando o máximo
    // Redução em x
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tx < stride) {
            max_shared[tx][ty] = fmax(max_shared[tx][ty], max_shared[tx + stride][ty]);
        }
        __syncthreads();
    }
    
    // Redução em y - apenas threads com tx==0 participam
    if (tx == 0) {
        for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
            if (ty < stride) {
                max_shared[0][ty] = fmax(max_shared[0][ty], max_shared[0][ty + stride]);
            }
            __syncthreads();
        }
        
        // Thread (0,0) salva o resultado final do bloco
        if (ty == 0) {
            block_max[blockIdx.y * gridDim.x + blockIdx.x] = max_shared[0][0];
        }
    }
}

// Kernel para redução final dos máximos de blocos para um único valor
__global__ void reduce_block_max_kernel(double *block_max, int num_blocks, double *final_max) {
    __shared__ double shared_data[256];
    
    int tid = threadIdx.x;
    
    // Inicializar com valor mínimo
    shared_data[tid] = -1e30;
    
    // Carregar dados para memória compartilhada
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        shared_data[tid] = fmax(shared_data[tid], block_max[i]);
    }
    
    __syncthreads();
    
    // Redução paralela para encontrar o máximo global
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmax(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    // Thread 0 escreve o resultado final
    if (tid == 0) {
        *final_max = shared_data[0];
    }
}

// Função host para encontrar o máximo de uma matriz usando CUDA
double max_mat_cuda(int i_max, int j_max, double **matrix) {
    const int BLOCK_SIZE = 16;
    
    // Configurar dimensões da grade e blocos
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((i_max + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (j_max + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Número total de blocos
    int total_blocks = gridDim.x * gridDim.y;
    
    // Alocar memória para máximos de blocos e máximo final
    double *d_block_max, *d_final_max;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_max, total_blocks * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_final_max, sizeof(double)));
    
    // Encontrar máximos locais em cada bloco
    find_max_kernel<<<gridDim, blockDim>>>(matrix, i_max, j_max, d_block_max);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Reduzir máximos locais para um máximo global
    reduce_block_max_kernel<<<1, 256>>>(d_block_max, total_blocks, d_final_max);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Transferir o máximo final para o host
    double max_value;
    CHECK_CUDA_ERROR(cudaMemcpy(&max_value, d_final_max, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Liberar memória
    cudaFree(d_block_max);
    cudaFree(d_final_max);
    
    return max_value;
}

/**
 * @brief Main function.
 * 
 * This is the main function.
 * @return 0 on exit.
 */

int main(int argc, char* argv[])
{
    // Grid pointers - agora serão alocados com UVA
    double** u;     // velocity x-component
    double** v;     // velocity y-component
    double** p;     // pressure
    double** F;     // F term
    double** G;     // G term
    double** res;   // SOR residuum
    double** RHS;   // RHS of poisson equation
    BoundaryPoint* borders; // Array to store border points
    // Simulation parameters.
    int i_max, j_max;                   // number of grid points in each direction
    double a, b;                        // sizes of the grid
    double Re;                          // reynolds number
    double delta_t, delta_x, delta_y;   // step sizes
    double gamma;                       // weight for Donor-Cell-stencil
    double T;                           // max time for integration
    double g_x;                         // x-component of g
    double g_y;                         // y-component of g
    double tau;                         // security factor for adaptive step size
    double omega;                       // relaxation parameter
    double epsilon;                     // relative tolerance for SOR
    int max_it;                         // maximum iterations for SOR
    int n_print;                        // output to file every ..th step
    int problem;                        // problem type
    double f;                           // frequency of periodic boundary conditions (if problem == 2)

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

    // Calcular o número exato de pontos de borda
    // A fórmula é 2 * ( (i_max+2) + (j_max+2) - 2 ) = 2 * (i_max + j_max + 2)
    // (soma dos comprimentos das bordas, subtraindo os 4 cantos contados duas vezes, mas cada célula de canto é um ponto)
    // Ou mais simples: (i_max+2)*2 para bordas superior/inferior + j_max*2 para bordas laterais (excluindo cantos já contados)
    // = 2*i_max + 4 + 2*j_max = 2 * (i_max + j_max + 2)
    int num_actual_border_points = 2 * (i_max + j_max + 2);

    // Passar num_actual_border_points para allocate_unified_memory
    allocate_unified_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max, &borders, num_actual_border_points);
    
    // precalculate_borders preenche o array 'borders'.
    // Ele não precisa mais do count como parâmetro se a memória já está dimensionada corretamente.
    precalculate_borders(i_max, j_max, borders);
    
    // Allocate memory using UVA instead of regular allocation

    // Time loop.
    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();

    while (t < T) {
        printf("%.5f / %.5f\n---------------------\n", t, T);

        // Adaptive stepsize and weight factor for Donor-Cell
        double u_max = max_mat_cuda(i_max, j_max, u);
        double v_max = max_mat_cuda(i_max, j_max, v);
        delta_t = tau * n_min(3, Re / 2.0 / ( 1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y ), delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);

        // Set boundary conditions (permanecem na CPU)
        if (problem == 1) {
            set_noslip(i_max, j_max, u, v, LEFT);
            set_noslip(i_max, j_max, u, v, RIGHT);
            set_noslip(i_max, j_max, u, v, BOTTOM);
            set_inflow(i_max, j_max, u, v, TOP, 1.0, 0.0);
        } else if (problem == 2) {
            set_noslip(i_max, j_max, u, v, LEFT);
            set_noslip(i_max, j_max, u, v, RIGHT);
            set_noslip(i_max, j_max, u, v, BOTTOM);
            set_inflow(i_max, j_max, u, v, TOP, sin(f*t), 0.0);           
        }

        dim3 blockDim(16, 16);
        dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x,
                     (j_max + blockDim.y - 1) / blockDim.y);

        // Calculate F and G (pode ser mantido na CPU ou implementado em CUDA)
        calculate_F_kernel<<<gridDim, blockDim>>>(F, u, v, i_max, j_max, Re, g_x, delta_t, delta_x, delta_y, gamma);
        CHECK_CUDA_ERROR(cudaGetLastError());

        calculate_G_kernel<<<gridDim, blockDim>>>(G, u, v, i_max, j_max, Re, g_y, delta_t, delta_x, delta_y, gamma);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // RHS of Poisson equation - now using CUDA kernel
        calculate_RHS_kernel<<<gridDim, blockDim>>>(RHS, F, G, i_max, j_max, delta_t, delta_x, delta_y);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        //clock_t start_sor = clock();
        // Execute SOR step using UVA
        // Passar num_actual_border_points para SOR_UVA
        //SOR_UVA(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it, borders, num_actual_border_points);
        SOR_UVA_with_shared_memory(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it, borders, num_actual_border_points);
        CHECK_CUDA_ERROR(cudaGetLastError());
        //clock_t end_sor = clock();
        //double sor_time = (double)(end_sor - start_sor) / CLOCKS_PER_SEC;
        //fprintf(stderr, "SOR time: %.6f\n", sor_time);

        // Update velocities using CUDA kernel
        update_velocities_kernel<<<gridDim, blockDim>>>(u, v, F, G, p, i_max, j_max, delta_t, delta_x, delta_y);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Print values (acessível diretamente da CPU devido ao UVA)
        printf("TIMESTEP: %d TIME: %.6f\n", n_out, t);
        printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
        printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);


        t += delta_t;
        n++;
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    fprintf(stderr, "%.6f", time_spent);

    // Free unified memory
    free_unified_memory(u, v, p, res, RHS, F, G, borders);
    return 0;
}

