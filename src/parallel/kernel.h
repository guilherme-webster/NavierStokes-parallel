#ifndef KERNEL_H
#define KERNEL_H

enum {
    TOP = 0,
    BOTTOM = 1,
    LEFT = 2,
    RIGHT = 3
};

typedef struct{
    int i;
    int j;
    int side;
} BoundaryPoint;

// Função host
void init_memory(int i_max, int j_max, BoundaryPoint* h_boundary_indices, int total_points, double* tau, double* Re, double* g_x, double* g_y, double* omega, double* epsilon, int* max_it);

double orchestration(int i_max, int j_max);

BoundaryPoint* generate_boundary_indices(int i_max, int j_max, int* total_points);

#ifdef __CUDACC__
// Kernels CUDA
__global__ void pick_max();

__global__ void min_and_gamma();

__device__ double atomicMax(double* address, double val);

__global__ void max_reduce_kernel(int i_max, int j_max, double* arr, double* max_val);

__global__ void update_boundaries_kernel();

__device__ double du2_dx(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max);

__device__ double duv_dy(double* u, double* v, int i, int j, double delta_y, double gamma, int j_max);

__device__ double dv2_dy(double* v, double* u, int i, int j, double delta_y, double gamma, int j_max);

__device__ double duv_dx(double* u, double* v, int i, int j, double delta_x, double gamma, int j_max);

__device__ double d2u_dx2(double* u, int i, int j, double delta_x, int j_max);

__device__ double d2u_dy2(double* u, int i, int j, double delta_y, int j_max);

__device__ double d2v_dx2(double* v, int i, int j, double delta_x, int j_max);

__device__ double d2v_dy2(double* v, int i, int j, double delta_y, int j_max);

__global__ void calculate_F(double* F, double* u, double* v, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma);

__global__ void calculate_G(double * G, double* u, double* v, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma);

__global__ void calculate_RHS(double* RHS, double* F, double* G, double* u, double* v, int i_max, int j_max, double delta_t, double delta_x, double delta_y);

__global__ void red_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max, double delta_x, double delta_y, double omega);

__global__ void black_kernel(double* p, double* RHS, double* u, double* v, int i_max, int j_max, double delta_x, double delta_y, double omega);

__global__ void calculate_ghost();

__global__ void L2_norm(double* norm, double* m, int i_max, int j_max);

__global__ void residual_kernel(double* res, double* p, double* RHS, int i_max, int j_max, double delta_x, double delta_y);

__global__ void update_velocity_kernel(double* u, double* v, double* p, int i_max, int j_max, double delta_t, double delta_x, double delta_y);

__global__ void extract_value_kernel(double* d_u, double* d_v, double* d_p, int i_max,int j_max, double* result);

#endif // __CUDACC__

#endif // KERNEL_H

