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
int allocate_unified_memory(double ***u, double ***v, double ***p, double ***res, double ***RHS, double ***F, double ***G, int i_max, int j_max, BoundaryPoint **borders) {
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
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&borders, 2 * (i_max + j_max + 4) * sizeof(BoundaryPoint)));

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


void precalculate_borders(int i_max, int j_max, BoundaryPoint *borders) {
    int index = 0;
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            if (i == 0 || i == i_max + 1 || j == 0 || j == j_max + 1) {
                borders[index].i = i;
                borders[index].j = j;
                borders[index].position = (i == 0) ? LEFT : (i == i_max + 1) ? RIGHT : (j == 0) ? BOTTOM : TOP;
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

// Função SOR usando UVA com critério de convergência similar ao serial
// Função SOR atualizada para usar o kernel otimizado de bordas
int SOR_UVA(double **p, int i_max, int j_max, double delta_x, double delta_y,
           double **res, double **RHS, double omega, double epsilon, int max_it,
           BoundaryPoint *borders) {
    
    dim3 blockDim(16, 16);
    dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x,
                 (j_max + blockDim.y - 1) / blockDim.y);
    
    // Configuração para o novo kernel de bordas (1D)
    int border_count = 2 * (i_max + j_max + 4); // Total de pontos de borda
    dim3 boundaryBlockDim(256); // Usar 256 threads por bloco para kernel 1D
    dim3 boundaryGridDim((border_count + boundaryBlockDim.x - 1) / boundaryBlockDim.x);
    
    // Calcular norma inicial de p
    double norm_p_initial = calculate_L2_norm_host_uva(p, i_max, j_max);
    
    for (int it = 0; it < max_it; it++) {
        // Prefetch para GPU se necessário (opcional)
        cudaMemPrefetchAsync(p[0], (i_max + 2) * (j_max + 2) * sizeof(double), 0);
        cudaMemPrefetchAsync(RHS[0], (i_max + 2) * (j_max + 2) * sizeof(double), 0);
        cudaMemPrefetchAsync(borders, border_count * sizeof(BoundaryPoint), 0);
        
        // 1. Atualizar bordas usando kernel otimizado com pontos pré-calculados
        update_boundaries_with_precalc_kernel<<<boundaryGridDim, boundaryBlockDim>>>(p, borders, border_count);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // 2. Red points (sem alteração)
        sor_red_kernel_uva<<<gridDim, blockDim>>>(p, RHS, i_max, j_max, 
                                                 delta_x, delta_y, omega);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // 3. Black points (sem alteração)
        sor_black_kernel_uva<<<gridDim, blockDim>>>(p, RHS, i_max, j_max, 
                                                   delta_x, delta_y, omega);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // 4. Calcular resíduo (sem alteração)
        calculate_poisson_residual_kernel<<<gridDim, blockDim>>>(p, RHS, res, i_max, j_max, delta_x, delta_y);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Calcular norma L2 do resíduo
        double current_L2_res_norm = calculate_L2_norm_host_uva(res, i_max, j_max);
        if (current_L2_res_norm <= epsilon * (norm_p_initial + 1.5)) {
            return it + 1;
        }
    }
    
    return -1;
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
    allocate_unified_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max, &borders);
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
        double u_max = max_mat(i_max, j_max, u);
        double v_max = max_mat(i_max, j_max, v);
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


        // Calculate F and G (pode ser mantido na CPU ou implementado em CUDA)
        FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);

        // RHS of Poisson equation - now using CUDA kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x,
                     (j_max + blockDim.y - 1) / blockDim.y);
        
        calculate_RHS_kernel<<<gridDim, blockDim>>>(RHS, F, G, i_max, j_max, delta_t, delta_x, delta_y);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        //clock_t start_sor = clock();
        // Execute SOR step using UVA
        SOR_UVA(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it, borders);
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

