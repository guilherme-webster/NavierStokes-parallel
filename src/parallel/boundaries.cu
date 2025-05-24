#include "boundaries.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Arrays para armazenar pontos de borda pré-calculados
BoundaryPoint *h_noslip_points;  // Host array para pontos no-slip
BoundaryPoint *d_noslip_points;  // Device array para pontos no-slip
BoundaryPoint *h_inflow_points;  // Host array para pontos de entrada
BoundaryPoint *d_inflow_points;  // Device array para pontos de entrada
int noslip_count = 0;
int inflow_count = 0;

// Pré-calcula os pontos de borda e aloca na GPU
void precompute_boundary_points(int i_max, int j_max) {
    // Calcular número total de pontos para cada tipo de borda
    int left_right_count = 2 * (j_max + 2);  // LEFT + RIGHT (incluindo cantos)
    int bottom_count = i_max + 2;            // BOTTOM
    int top_count = i_max + 2;               // TOP
    
    // Alocar arrays de host para pontos de borda
    noslip_count = left_right_count + bottom_count;
    inflow_count = top_count;
    
    h_noslip_points = (BoundaryPoint*)malloc(noslip_count * sizeof(BoundaryPoint));
    h_inflow_points = (BoundaryPoint*)malloc(inflow_count * sizeof(BoundaryPoint));
    
    if (h_noslip_points == NULL || h_inflow_points == NULL) {
        fprintf(stderr, "Falha na alocação de memória para pontos de borda\n");
        exit(EXIT_FAILURE);
    }
    
    int idx = 0;
    
    // Calcular pontos LEFT (no-slip)
    for (int j = 0; j <= j_max+1; j++) {
        h_noslip_points[idx].i = 0;
        h_noslip_points[idx].j = j;
        h_noslip_points[idx].direction = LEFT;
        idx++;
    }
    
    // Calcular pontos RIGHT (no-slip)
    for (int j = 0; j <= j_max+1; j++) {
        h_noslip_points[idx].i = i_max+1;
        h_noslip_points[idx].j = j;
        h_noslip_points[idx].direction = RIGHT;
        idx++;
    }
    
    // Calcular pontos BOTTOM (no-slip)
    for (int i = 0; i <= i_max+1; i++) {
        h_noslip_points[idx].i = i;
        h_noslip_points[idx].j = 0;
        h_noslip_points[idx].direction = BOTTOM;
        idx++;
    }
    
    // Calcular pontos TOP (inflow)
    idx = 0;
    for (int i = 0; i <= i_max+1; i++) {
        h_inflow_points[idx].i = i;
        h_inflow_points[idx].j = j_max+1;
        h_inflow_points[idx].direction = TOP;
        idx++;
    }
    
    // Alocar e copiar arrays para device
    cudaMalloc((void**)&d_noslip_points, noslip_count * sizeof(BoundaryPoint));
    cudaMalloc((void**)&d_inflow_points, inflow_count * sizeof(BoundaryPoint));
    
    cudaMemcpy(d_noslip_points, h_noslip_points, noslip_count * sizeof(BoundaryPoint), 
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_inflow_points, h_inflow_points, inflow_count * sizeof(BoundaryPoint), 
              cudaMemcpyHostToDevice);
    
    printf("Pré-cálculo de bordas concluído: %d pontos no-slip, %d pontos inflow\n", 
           noslip_count, inflow_count);
}

// Kernel CUDA para aplicar condições no-slip
__global__ void apply_noslip_kernel(double *u, double *v, BoundaryPoint *points, 
                                   int count, int i_max, int j_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        BoundaryPoint bp = points[idx];
        int i = bp.i;
        int j = bp.j;
        int direction = bp.direction;
        
        // Índice linear para matrizes 2D
        int idx_u = j * (i_max + 2) + i;
        
        if (direction == LEFT) {
            u[idx_u] = 0.0;
            v[idx_u] = -v[idx_u + 1];
        } else if (direction == RIGHT) {
            u[idx_u - 1] = 0.0;
            v[idx_u] = -v[idx_u - 1];
        } else if (direction == BOTTOM) {
            u[idx_u] = -u[idx_u + (i_max + 2)];
            v[idx_u] = 0.0;
        }
    }
}

// Kernel CUDA para aplicar condições inflow
__global__ void apply_inflow_kernel(double *u, double *v, BoundaryPoint *points, 
                                   int count, int i_max, int j_max, double u_in, double v_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        BoundaryPoint bp = points[idx];
        int i = bp.i;
        int j = bp.j;
        
        // Índice linear para matrizes 2D
        int idx_u = j * (i_max + 2) + i;
        
        // TOP boundary
        u[idx_u - (i_max + 2)] = 2.0 * u_in - u[idx_u];
        v[idx_u - (i_max + 2)] = v_in;
    }
}

// Wrapper para chamar kernels de borda
void apply_boundary_conditions(double *d_u, double *d_v, int i_max, int j_max, 
                             int problem, double t, double f) {
    int blockSize = 256;
    int gridSize;
    
    // Aplicar condições no-slip
    gridSize = (noslip_count + blockSize - 1) / blockSize;
    apply_noslip_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_noslip_points, 
                                               noslip_count, i_max, j_max);
    
    // Aplicar condições inflow
    gridSize = (inflow_count + blockSize - 1) / blockSize;
    double u_in = (problem == 1) ? 1.0 : sin(f * t);
    double v_in = 0.0;
    
    apply_inflow_kernel<<<gridSize, blockSize>>>(d_u, d_v, d_inflow_points, 
                                               inflow_count, i_max, j_max, u_in, v_in);
}

// Liberar memória alocada
void free_boundary_points() {
    free(h_noslip_points);
    free(h_inflow_points);
    cudaFree(d_noslip_points);
    cudaFree(d_inflow_points);
}