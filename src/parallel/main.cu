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

#include <time.h>
#include <math.h>
#include <stdio.h>

#ifndef BOUNDARY_TYPES_H
#define BOUNDARY_TYPES_H

#define LEFT 0
#define RIGHT 1
#define BOTTOM 2
#define TOP 3

typedef struct {
    int i;
    int j;
    int direction; // 0: LEFT, 1: RIGHT, 2: BOTTOM, 3: TOP
} BoundaryPoint;

#endif // BOUNDARY_TYPES_H

// Arrays para armazenar pontos de borda pré-calculados
BoundaryPoint *h_noslip_left_points;
BoundaryPoint *h_noslip_right_points;
BoundaryPoint *h_noslip_bottom_points;
BoundaryPoint *h_inflow_top_points;

BoundaryPoint *d_noslip_left_points;
BoundaryPoint *d_noslip_right_points;
BoundaryPoint *d_noslip_bottom_points;
BoundaryPoint *d_inflow_top_points;

int left_count, right_count, bottom_count, top_count;

// Kernels CUDA para aplicar condições de contorno em paralelo
__global__ void apply_noslip_left_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max);

__global__ void apply_noslip_right_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max);


__global__ void apply_noslip_bottom_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max);

__global__ void apply_inflow_top_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max, double u_in, double v_in);
// Função para pré-calcular pontos de borda
void precompute_boundary_points(int i_max, int j_max);

// Função para aplicar condições de contorno em paralelo
void apply_boundary_conditions_parallel(double *d_u, double *d_v, int i_max, int j_max, int problem, double t, double f);

// Função para liberar memória dos pontos de borda
void free_boundary_points();


__global__ void calculate_FG_kernel(double *d_F, double *d_G, double *d_u, double *d_v, 
                                   int i_max, int j_max, double Re, double g_x, double g_y,
                                   double delta_t, double delta_x, double delta_y, double gamma);

// Função wrapper para calcular F e G em paralelo
void calculate_FG_parallel(double *d_F, double *d_G, double *d_u, double *d_v, 
                          int i_max, int j_max, double Re, double g_x, double g_y,
                          double delta_t, double delta_x, double delta_y, double gamma);


__global__ void calculate_RHS_kernel(double *d_RHS, double *d_F, double *d_G, 
                                   int i_max, int j_max, double delta_t, 
                                   double delta_x, double delta_y);


void calculate_RHS_parallel(double *d_RHS, double *d_F, double *d_G, 
                          int i_max, int j_max, double delta_t, 
                          double delta_x, double delta_y);

__global__ void find_max_kernel(double *data, double *result, int size);

double find_max_parallel(double *d_data, int size);

int sor_parallel(double *d_p, int i_max, int j_max, double delta_x, double delta_y, 
                double *d_res, double *d_RHS, double omega, double epsilon, int max_it);

__global__ void sor_red_black_kernel(double *d_p, double *d_RHS, int i_max, int j_max, 
                                   double omega, double delta_x, double delta_y, int red_or_black);

__global__ void calculate_residual_kernel(double *d_p, double *d_RHS, double *d_res, 
                                        int i_max, int j_max, double delta_x, double delta_y);

int sor_parallel(double *d_p, int i_max, int j_max, double delta_x, double delta_y, 
                double *d_res, double *d_RHS, double omega, double epsilon, int max_it);

// Kernel para atualizar velocidades
__global__ void update_velocities_kernel(double *d_u, double *d_v, double *d_F, double *d_G, 
                                      double *d_p, int i_max, int j_max, 
                                      double delta_t, double delta_x, double delta_y);

void update_velocities_parallel(double *d_u, double *d_v, double *d_F, double *d_G, 
                             double *d_p, int i_max, int j_max, 
                             double delta_t, double delta_x, double delta_y);

void extract_key_values(double *d_u, double *d_v, double *d_p, int i_max, int j_max, double *u_center, double *v_center, double *p_center);
/**
 * @brief Main function.
 * 
 * This is the main function.
 * @return 0 on exit.
 */

int main(int argc, char* argv[])
{
    // Simulation parameters.
    int i_max, j_max;                   // number of grid points in each direction
    double a, b;                        // sizes of the grid
    double Re;                          // reynolds number
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

    precompute_boundary_points(i_max, j_max);

    // Set step size in space.
    double delta_x = a / i_max;
    double delta_y = b / j_max;
    double delta_t = 0.0; // Initialize delta_t

    double *d_u, *d_v, *d_p;
    double *d_F, *d_G, *d_res, *d_RHS;
    // Allocate memory for device variables
    int size = (i_max +2) * (j_max + 2) * sizeof(double);
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_p, size);
    cudaMalloc((void**)&d_F, size);
    cudaMalloc((void**)&d_G, size);
    cudaMalloc((void**)&d_res, size);
    cudaMalloc((void**)&d_RHS, size);
    
    cudaMemset(d_u, 0, size);
    cudaMemset(d_v, 0, size);
    cudaMemset(d_p, 0, size);
    cudaMemset(d_F, 0, size);
    cudaMemset(d_G, 0, size);
    cudaMemset(d_res, 0, size);
    cudaMemset(d_RHS, 0, size);
    

        // Time loop.
    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();

    while (t < T) {
        // Garantir que d_u e d_v estejam atualizados na GPU
        // (Este código deve ser executado em outro lugar - antes do loop ou na iteração anterior)
        
        // Calcular u_max e v_max diretamente na GPU
        double u_max = find_max_parallel(d_u, (i_max+2) * (j_max+2));
        double v_max = find_max_parallel(d_v, (i_max+2) * (j_max+2));
        printf("u_max: %.6f, v_max: %.6f\n", u_max, v_max);
        
        // Calcular delta_t e gamma na CPU
        printf("delta_x: %.6f, delta_y: %.6f\n", delta_x, delta_y);
        delta_t = tau * n_min(3, Re / 2.0 / (1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y), 
                          delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);
        
        // Aplicar condições de contorno
        apply_boundary_conditions_parallel(d_u, d_v, i_max, j_max, problem, t, f);

        // Calculate F and G.
        // FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);
        // Calcular F e G em paralelo na GPU
        calculate_FG_parallel(d_F, d_G, d_u, d_v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);

        // Calcular RHS diretamente na GPU
        calculate_RHS_parallel(d_RHS, d_F, d_G, i_max, j_max, delta_t, delta_x, delta_y);

        // Executar SOR paralelo
        int sor_iterations = sor_parallel(d_p, i_max, j_max, delta_x, delta_y, d_res, d_RHS, omega, epsilon, max_it);

        // Atualizar velocidades em paralelo na GPU
        update_velocities_parallel(d_u, d_v, d_F, d_G, d_p, i_max, j_max, delta_t, delta_x, delta_y);


        if (n % n_print == 0) {
            // Extrair apenas os valores necessários para impressão
            double u_center, v_center, p_center;
            //extract_key_values(d_u, d_v, d_p, i_max, j_max, &u_center, &v_center, &p_center);
            //
            //// Imprimir os dados
            //printf("TIMESTEP: %d TIME: %.6f\n", n_out, t);
            //printf("U-CENTER: %.6f\n", u_center);
            //printf("V-CENTER: %.6f\n", v_center);
            //printf("P-CENTER: %.6f\n", p_center);
            
            n_out++;
        }

        t += delta_t;
        n++;
    }

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_spent);
  

    return 0;
}


// Kernels CUDA para aplicar condições de contorno em paralelo
__global__ void apply_noslip_left_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        BoundaryPoint bp = points[idx];
        int i = bp.i;
        int j = bp.j;
        
        // Índice linear para matriz 1D representando uma matriz 2D
        int idx_u = j * (i_max + 2) + i;
        
        // LEFT boundary: no-slip
        u[idx_u] = 0.0;
        v[idx_u] = -v[idx_u + 1];
    }
}

__global__ void apply_noslip_right_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        BoundaryPoint bp = points[idx];
        int i = bp.i;
        int j = bp.j;
        
        int idx_u = j * (i_max + 2) + i;
        
        // RIGHT boundary: no-slip
        u[idx_u - 1] = 0.0;
        v[idx_u] = -v[idx_u - 1];
    }
}

__global__ void apply_noslip_bottom_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        BoundaryPoint bp = points[idx];
        int i = bp.i;
        int j = bp.j;
        
        int idx_u = j * (i_max + 2) + i;
        
        // BOTTOM boundary: no-slip
        u[idx_u] = -u[idx_u + (i_max + 2)];
        v[idx_u] = 0.0;
    }
}

__global__ void apply_inflow_top_kernel(double *u, double *v, BoundaryPoint *points, int count, int i_max, int j_max, double u_in, double v_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        BoundaryPoint bp = points[idx];
        int i = bp.i;
        int j = bp.j;
        
        int idx_u = j * (i_max + 2) + i;
        int idx_below = (j-1) * (i_max + 2) + i;
        
        // TOP boundary: inflow
        u[idx_below] = 2.0 * u_in - u[idx_u];
        v[idx_below] = v_in;
    }
}

// Função para pré-calcular pontos de borda
void precompute_boundary_points(int i_max, int j_max) {
    printf("Pré-calculando pontos de borda...\n");
    
    // Calcular tamanhos dos arrays
    left_count = j_max + 2;
    right_count = j_max + 2;
    bottom_count = i_max + 2;
    top_count = i_max + 2;
    
    // Alocar memória no host
    h_noslip_left_points = (BoundaryPoint*)malloc(left_count * sizeof(BoundaryPoint));
    h_noslip_right_points = (BoundaryPoint*)malloc(right_count * sizeof(BoundaryPoint));
    h_noslip_bottom_points = (BoundaryPoint*)malloc(bottom_count * sizeof(BoundaryPoint));
    h_inflow_top_points = (BoundaryPoint*)malloc(top_count * sizeof(BoundaryPoint));
    
    // Preencher arrays de pontos
    for (int j = 0; j <= j_max+1; j++) {
        h_noslip_left_points[j].i = 0;
        h_noslip_left_points[j].j = j;
        h_noslip_left_points[j].direction = LEFT;
        
        h_noslip_right_points[j].i = i_max+1;
        h_noslip_right_points[j].j = j;
        h_noslip_right_points[j].direction = RIGHT;
    }
    
    for (int i = 0; i <= i_max+1; i++) {
        h_noslip_bottom_points[i].i = i;
        h_noslip_bottom_points[i].j = 0;
        h_noslip_bottom_points[i].direction = BOTTOM;
        
        h_inflow_top_points[i].i = i;
        h_inflow_top_points[i].j = j_max+1;
        h_inflow_top_points[i].direction = TOP;
    }
    
    // Alocar memória no device
    cudaMalloc((void**)&d_noslip_left_points, left_count * sizeof(BoundaryPoint));
    cudaMalloc((void**)&d_noslip_right_points, right_count * sizeof(BoundaryPoint));
    cudaMalloc((void**)&d_noslip_bottom_points, bottom_count * sizeof(BoundaryPoint));
    cudaMalloc((void**)&d_inflow_top_points, top_count * sizeof(BoundaryPoint));
    
    // Copiar dados para o device
    cudaMemcpy(d_noslip_left_points, h_noslip_left_points, left_count * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_noslip_right_points, h_noslip_right_points, right_count * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_noslip_bottom_points, h_noslip_bottom_points, bottom_count * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inflow_top_points, h_inflow_top_points, top_count * sizeof(BoundaryPoint), cudaMemcpyHostToDevice);
    
    printf("Pré-cálculo concluído: %d pontos LEFT, %d pontos RIGHT, %d pontos BOTTOM, %d pontos TOP\n", 
           left_count, right_count, bottom_count, top_count);
}

// Função para aplicar condições de contorno em paralelo
void apply_boundary_conditions_parallel(double *d_u, double *d_v, int i_max, int j_max, int problem, double t, double f) {
    int blockSize = 256;
    int gridSize;
    
    // Aplicar condições no-slip nas bordas LEFT, RIGHT e BOTTOM
    gridSize = (left_count + blockSize - 1) / blockSize;
    apply_noslip_left_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_noslip_left_points, left_count, i_max, j_max);
    
    gridSize = (right_count + blockSize - 1) / blockSize;
    apply_noslip_right_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_noslip_right_points, right_count, i_max, j_max);
    
    gridSize = (bottom_count + blockSize - 1) / blockSize;
    apply_noslip_bottom_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_noslip_bottom_points, bottom_count, i_max, j_max);
    
    // Aplicar condição inflow na borda TOP
    double u_in = (problem == 1) ? 1.0 : sin(f * t);
    double v_in = 0.0;
    
    gridSize = (top_count + blockSize - 1) / blockSize;
    apply_inflow_top_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_inflow_top_points, top_count, i_max, j_max, u_in, v_in);
    
    // Verificar erros
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro CUDA: %s\n", cudaGetErrorString(err));
    }
}

// Função para liberar memória dos pontos de borda
void free_boundary_points() {
    free(h_noslip_left_points);
    free(h_noslip_right_points);
    free(h_noslip_bottom_points);
    free(h_inflow_top_points);
    
    cudaFree(d_noslip_left_points);
    cudaFree(d_noslip_right_points);
    cudaFree(d_noslip_bottom_points);
    cudaFree(d_inflow_top_points);
}

// Kernel de redução paralela para encontrar o valor máximo absoluto
__global__ void find_max_kernel(double *data, double *result, int size) {
    // Definindo tamanho fixo da memória compartilhada - 256 é um bom valor (múltiplo de 32)
    __shared__ double sdata[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Inicialização
    double threadMax = 0.0;
    if (i < size) {
        threadMax = fabs(data[i]);
    }
    
    // Carregar na memória compartilhada
    sdata[tid] = threadMax;
    __syncthreads();
    
    // Redução na memória compartilhada
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Thread 0 escreve o resultado para memória global
    if (tid == 0) {
        atomicMax((unsigned long long int*)&result[0], 
                  __double_as_longlong(sdata[0]));
    }
}

// Função para calcular o máximo em paralelo
double find_max_parallel(double *d_data, int size) {
    int blockSize = 256; // Mesmo tamanho da memória compartilhada
    int numBlocks = (size + blockSize - 1) / blockSize;
    
    // Alocar e inicializar resultado na GPU
    double *d_max;
    cudaMalloc((void**)&d_max, sizeof(double));
    cudaMemset(d_max, 0, sizeof(double));
    
    // Lançar kernel de redução
    find_max_kernel<<<numBlocks, blockSize>>>(d_data, d_max, size);
    
    // Copiar resultado para CPU
    double h_max;
    cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    
    // Liberar memória
    cudaFree(d_max);
    
    return h_max;
}

// Kernel para calcular F e G em paralelo
__global__ void calculate_FG_kernel(double *d_F, double *d_G, double *d_u, double *d_v, 
                                   int i_max, int j_max, double Re, double g_x, double g_y,
                                   double delta_t, double delta_x, double delta_y, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        int idx = j * (i_max + 2) + i;
        
        // Índices para acessar elementos vizinhos
        int idx_im1 = j * (i_max + 2) + (i-1);        // (i-1,j)
        int idx_ip1 = j * (i_max + 2) + (i+1);        // (i+1,j)
        int idx_jm1 = (j-1) * (i_max + 2) + i;        // (i,j-1)
        int idx_jp1 = (j+1) * (i_max + 2) + i;        // (i,j+1)
        
        // Índices diagonais para termos de convecção
        int idx_im1_jm1 = (j-1) * (i_max + 2) + (i-1); // (i-1,j-1)
        int idx_ip1_jm1 = (j-1) * (i_max + 2) + (i+1); // (i+1,j-1)
        int idx_im1_jp1 = (j+1) * (i_max + 2) + (i-1); // (i-1,j+1)
        int idx_ip1_jp1 = (j+1) * (i_max + 2) + (i+1); // (i+1,j+1)
        
        // Cálculo de F[i][j]
        if (i <= i_max - 1) {
            // Termos de difusão para u
            double laplace_u = (d_u[idx_ip1] - 2.0 * d_u[idx] + d_u[idx_im1]) / (delta_x * delta_x) + 
                              (d_u[idx_jp1] - 2.0 * d_u[idx] + d_u[idx_jm1]) / (delta_y * delta_y);
            
            // Termos de convecção para u
            double du2_dx = 1.0 / delta_x * (
                ((d_u[idx] + d_u[idx_ip1]) * (d_u[idx] + d_u[idx_ip1]) / 4.0) - 
                ((d_u[idx_im1] + d_u[idx]) * (d_u[idx_im1] + d_u[idx]) / 4.0)
            );
            
            double duv_dy = 1.0 / delta_y * (
                ((d_v[idx] + d_v[idx_ip1]) * (d_u[idx] + d_u[idx_jp1]) / 4.0) - 
                ((d_v[idx_jm1] + d_v[idx_jm1 + 1]) * (d_u[idx_jm1] + d_u[idx]) / 4.0)
            );
            
            // Termos de convecção com Donor-Cell
            if (gamma > 0.0) {
                double u_donor = gamma * (
                    fabs(d_u[idx] + d_u[idx_ip1]) * (d_u[idx] - d_u[idx_ip1]) / 4.0 +
                    fabs(d_u[idx_im1] + d_u[idx]) * (d_u[idx_im1] - d_u[idx]) / 4.0
                );
                
                double v_donor = gamma * (
                    fabs(d_v[idx] + d_v[idx_ip1]) * (d_u[idx] - d_u[idx_jp1]) / 4.0 +
                    fabs(d_v[idx_jm1] + d_v[idx_jm1 + 1]) * (d_u[idx_jm1] - d_u[idx]) / 4.0
                );
                
                du2_dx += u_donor;
                duv_dy += v_donor;
            }
            
            // Combinando todos os termos para F
            d_F[idx] = d_u[idx] + delta_t * (
                (1.0 / Re) * laplace_u - du2_dx - duv_dy + g_x
            );
        }
        
        // Cálculo de G[i][j]
        if (j <= j_max - 1) {
            // Termos de difusão para v
            double laplace_v = (d_v[idx_ip1] - 2.0 * d_v[idx] + d_v[idx_im1]) / (delta_x * delta_x) + 
                              (d_v[idx_jp1] - 2.0 * d_v[idx] + d_v[idx_jm1]) / (delta_y * delta_y);
            
            // Termos de convecção para v
            double duv_dx = 1.0 / delta_x * (
                ((d_u[idx] + d_u[idx_jp1]) * (d_v[idx] + d_v[idx_ip1]) / 4.0) - 
                ((d_u[idx_im1] + d_u[idx_im1 + (i_max + 2)]) * (d_v[idx_im1] + d_v[idx]) / 4.0)
            );
            
            double dv2_dy = 1.0 / delta_y * (
                ((d_v[idx] + d_v[idx_jp1]) * (d_v[idx] + d_v[idx_jp1]) / 4.0) - 
                ((d_v[idx_jm1] + d_v[idx]) * (d_v[idx_jm1] + d_v[idx]) / 4.0)
            );
            
            // Termos de convecção com Donor-Cell
            if (gamma > 0.0) {
                double u_donor = gamma * (
                    fabs(d_u[idx] + d_u[idx_jp1]) * (d_v[idx] - d_v[idx_ip1]) / 4.0 +
                    fabs(d_u[idx_im1] + d_u[idx_im1 + (i_max + 2)]) * (d_v[idx_im1] - d_v[idx]) / 4.0
                );
                
                double v_donor = gamma * (
                    fabs(d_v[idx] + d_v[idx_jp1]) * (d_v[idx] - d_v[idx_jp1]) / 4.0 +
                    fabs(d_v[idx_jm1] + d_v[idx]) * (d_v[idx_jm1] - d_v[idx]) / 4.0
                );
                
                duv_dx += u_donor;
                dv2_dy += v_donor;
            }
            
            // Combinando todos os termos para G
            d_G[idx] = d_v[idx] + delta_t * (
                (1.0 / Re) * laplace_v - duv_dx - dv2_dy + g_y
            );
        }
    }
}

// Função wrapper para calcular F e G em paralelo
void calculate_FG_parallel(double *d_F, double *d_G, double *d_u, double *d_v, 
                          int i_max, int j_max, double Re, double g_x, double g_y,
                          double delta_t, double delta_x, double delta_y, double gamma) {
    // Definir dimensões da grade e dos blocos
    dim3 blockDim(16, 16);  // 16x16 = 256 threads por bloco
    dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x, 
                 (j_max + blockDim.y - 1) / blockDim.y);
    
    // Lançar kernel
    calculate_FG_kernel<<<gridDim, blockDim>>>(
        d_F, d_G, d_u, d_v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma
    );
    
    // Verificar erros
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro CUDA no cálculo de F e G: %s\n", cudaGetErrorString(err));
    }
    
    // Sincronizar para garantir que o cálculo foi concluído
    cudaDeviceSynchronize();
}

// Kernel para calcular RHS da equação de Poisson em paralelo
__global__ void calculate_RHS_kernel(double *d_RHS, double *d_F, double *d_G, 
                                   int i_max, int j_max, double delta_t, 
                                   double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        // Calcular índices lineares para acesso aos arrays
        int idx = j * (i_max + 2) + i;
        int idx_im1 = j * (i_max + 2) + (i-1);    // F[i-1][j]
        int idx_jm1 = (j-1) * (i_max + 2) + i;    // G[i][j-1]
        
        // Calcular RHS
        d_RHS[idx] = 1.0 / delta_t * ((d_F[idx] - d_F[idx_im1]) / delta_x + 
                                     (d_G[idx] - d_G[idx_jm1]) / delta_y);
    }
}

// Função wrapper para calcular RHS em paralelo
void calculate_RHS_parallel(double *d_RHS, double *d_F, double *d_G, 
                          int i_max, int j_max, double delta_t, 
                          double delta_x, double delta_y) {
    // Definir dimensões da grade e dos blocos
    dim3 blockDim(16, 16);  // 16x16 = 256 threads por bloco
    dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x, 
                (j_max + blockDim.y - 1) / blockDim.y);
    
    // Lançar kernel
    calculate_RHS_kernel<<<gridDim, blockDim>>>(
        d_RHS, d_F, d_G, i_max, j_max, delta_t, delta_x, delta_y
    );
    
    // Verificar erros
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro CUDA no cálculo de RHS: %s\n", cudaGetErrorString(err));
    }
    
    // Sincronizar para garantir que o cálculo foi concluído
    cudaDeviceSynchronize();
}

// Kernel para calcular uma iteração Red-Black SOR
__global__ void sor_red_black_kernel(double *d_p, double *d_RHS, int i_max, int j_max, 
                                   double omega, double delta_x, double delta_y, int red_or_black) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // red_or_black: 0 para pontos vermelhos (i+j é par), 1 para pontos pretos (i+j é ímpar)
    if (i <= i_max && j <= j_max && (i + j) % 2 == red_or_black) {
        // Calcular índice linear para acessar arrays
        int idx = j * (i_max + 2) + i;
        int idx_ip1 = idx + 1;          // p[i+1][j]
        int idx_im1 = idx - 1;          // p[i-1][j]
        int idx_jp1 = idx + (i_max + 2); // p[i][j+1]
        int idx_jm1 = idx - (i_max + 2); // p[i][j-1]
        
        // Calcular coeficientes
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        
        // Aplicar uma iteração do SOR
        d_p[idx] = (1.0 - omega) * d_p[idx] + 
                   omega * ((d_p[idx_ip1] + d_p[idx_im1]) / dx2 + 
                           (d_p[idx_jp1] + d_p[idx_jm1]) / dy2 - 
                           d_RHS[idx]) / (2.0 / dx2 + 2.0 / dy2);
    }
}

// Kernel para calcular o resíduo
__global__ void calculate_residual_kernel(double *d_p, double *d_RHS, double *d_res, 
                                        int i_max, int j_max, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        // Calcular índice linear para acessar arrays
        int idx = j * (i_max + 2) + i;
        int idx_ip1 = idx + 1;          // p[i+1][j]
        int idx_im1 = idx - 1;          // p[i-1][j]
        int idx_jp1 = idx + (i_max + 2); // p[i][j+1]
        int idx_jm1 = idx - (i_max + 2); // p[i][j-1]
        
        // Calcular coeficientes
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        
        // Calcular resíduo
        d_res[idx] = fabs(
            ((d_p[idx_ip1] - 2.0 * d_p[idx] + d_p[idx_im1]) / dx2 + 
             (d_p[idx_jp1] - 2.0 * d_p[idx] + d_p[idx_jm1]) / dy2) - 
            d_RHS[idx]
        );
    }
}

// Função wrapper para resolver a equação de Poisson usando SOR paralelo
int sor_parallel(double *d_p, int i_max, int j_max, double delta_x, double delta_y, 
                double *d_res, double *d_RHS, double omega, double epsilon, int max_it) {
    // Configurações de grade e bloco para kernels
    dim3 blockDim(16, 16);
    dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x, 
                 (j_max + blockDim.y - 1) / blockDim.y);
    
    // Loop SOR
    int it;
    double max_res = 0.0;
    
    for (it = 0; it < max_it; it++) {
        // Aplicar Red-Black SOR
        sor_red_black_kernel<<<gridDim, blockDim>>>(d_p, d_RHS, i_max, j_max, omega, delta_x, delta_y, 0); // Red
        cudaDeviceSynchronize();
        
        sor_red_black_kernel<<<gridDim, blockDim>>>(d_p, d_RHS, i_max, j_max, omega, delta_x, delta_y, 1); // Black
        cudaDeviceSynchronize();
        
        // A cada 50 iterações (ou na última), verificar convergência para economizar tempo
        if (it % 50 == 0 || it == max_it - 1) {
            // Calcular resíduo
            calculate_residual_kernel<<<gridDim, blockDim>>>(d_p, d_RHS, d_res, i_max, j_max, delta_x, delta_y);
            cudaDeviceSynchronize();
            
            // Encontrar o resíduo máximo usando a função já existente
            max_res = find_max_parallel(d_res, (i_max+2) * (j_max+2));
            
            // Verificar convergência
            if (max_res < epsilon) break;
        }
    }
    
    // Retornar -1 se excedeu o número máximo de iterações, ou o número de iterações
    return (it == max_it) ? -1 : it;
}

// Kernel para atualizar velocidades
__global__ void update_velocities_kernel(double *d_u, double *d_v, double *d_F, double *d_G, 
                                      double *d_p, int i_max, int j_max, 
                                      double delta_t, double delta_x, double delta_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        // Calcular índice linear para acessar arrays
        int idx = j * (i_max + 2) + i;
        int idx_ip1 = idx + 1;          // p[i+1][j]
        int idx_im1 = idx - 1;          // p[i-1][j]
        int idx_jp1 = idx + (i_max + 2); // p[i][j+1]
        int idx_jm1 = idx - (i_max + 2); // p[i][j-1]
        
        // Atualizar u
        if (i <= i_max - 1) {
            d_u[idx] = d_F[idx] - delta_t * (d_p[idx_ip1] - d_p[idx]) / delta_x;
        }
        
        // Atualizar v
        if (j <= j_max - 1) {
            d_v[idx] = d_G[idx] - delta_t * (d_p[idx_jp1] - d_p[idx]) / delta_y;
        }
    }
}

// Função wrapper para atualizar velocidades
void update_velocities_parallel(double *d_u, double *d_v, double *d_F, double *d_G, 
                             double *d_p, int i_max, int j_max, 
                             double delta_t, double delta_x, double delta_y) {
    // Configurações de grade e bloco
    dim3 blockDim(16, 16);
    dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x, 
                 (j_max + blockDim.y - 1) / blockDim.y);
    
    // Lançar kernel
    update_velocities_kernel<<<gridDim, blockDim>>>(
        d_u, d_v, d_F, d_G, d_p, i_max, j_max, delta_t, delta_x, delta_y
    );
    
    // Verificar erros
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Erro CUDA na atualização de velocidades: %s\n", cudaGetErrorString(err));
    }
    
    // Sincronizar
    cudaDeviceSynchronize();
}

// Adicionar esta nova função para extrair valores específicos
void extract_key_values(double *d_u, double *d_v, double *d_p, int i_max, int j_max,
                       double *u_center, double *v_center, double *p_center) {
    // Calcular índice do centro
    int center_idx = (j_max/2) * (i_max + 2) + (i_max/2);
    
    // Transferir apenas os valores específicos
    cudaMemcpy(u_center, &d_u[center_idx], sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_center, &d_v[center_idx], sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p_center, &d_p[center_idx], sizeof(double), cudaMemcpyDeviceToHost);
}