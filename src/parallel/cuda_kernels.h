#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3

// Inicializa os arrays CUDA em memória de device
int initCudaArrays(double** p, double** u, double** v, double** res, double** RHS, int i_max, int j_max);

// Libera os arrays CUDA em memória de device
void freeCudaArrays();

// SOR principal
int cudaSOR(double** p, double** u, double** v, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it, double** F, double** G, double tau, double Re,
            int problem, double f, double* t, int* n_out, double g_x, double g_y);

// Função utilitária para mínimo
double n_min(int n, double a, double b, double c, double d);

// KERNELS CUDA (definidos em cuda_kernels.cu)
#ifdef __CUDACC__
// Template precisa estar fora do extern "C"
template <typename T>
__global__ void max_mat_kernel(const T* mat, int i_max, int j_max, T* max_val);

// CUDA kernel launches em extern "C" (sem template)
extern "C" {
    __global__ void max_mat_kernel_double(const double* mat, int i_max, int j_max, double* max_val);
    __global__ void dp_dx_kernel(const double* p, double* out, int i_max, int j_max, double delta_x);
    __global__ void dp_dy_kernel(const double* p, double* out, int i_max, int j_max, double delta_y);
    __global__ void set_noslip_linear_kernel(int i_max, int j_max, double* u, double* v, int side);
    __global__ void set_inflow_linear_kernel(int i_max, int j_max, double* u, double* v, int side, double u_fix, double v_fix);
    __global__ void FG_linear_kernel(double* u, double* v, double* F, double* G, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma_factor);
    __global__ void RHS_kernel(double* F, double* G, double* RHS, int i_max, int j_max, double delta_t, double delta_x, double delta_y);
    __global__ void update_uv_kernel(double* u, double* v, double* F, double* G, double* p, int i_max, int j_max, double delta_t, double delta_x, double delta_y);
    __global__ void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
    __global__ void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
    __global__ void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy);
    __global__ void update_pressure_bounds_kernel(double* p, int i_max, int j_max);
    __global__ void reduce_sum_kernel(double* input, double* output, int size);
    __global__ void extract_value_kernel(double* array, int idx, double* result);
    __global__ void calculate_residual_norm_kernel(double* res, double* norm, int i_max, int j_max);
    __global__ void calculate_pressure_norm_kernel(double* p, double* norm, int i_max, int j_max);
}
#else
// Para compilação C/C++ normal, declarar apenas os protótipos das funções host
template <typename T>
void max_mat_kernel(const T* mat, int i_max, int j_max, T* max_val);
void max_mat_kernel_double(const double* mat, int i_max, int j_max, double* max_val);
void dp_dx_kernel(const double* p, double* out, int i_max, int j_max, double delta_x);
void dp_dy_kernel(const double* p, double* out, int i_max, int j_max, double delta_y);
void set_noslip_linear_kernel(int i_max, int j_max, double* u, double* v, int side);
void set_inflow_linear_kernel(int i_max, int j_max, double* u, double* v, int side, double u_fix, double v_fix);
void FG_linear_kernel(double* u, double* v, double* F, double* G, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma_factor);
void RHS_kernel(double* F, double* G, double* RHS, int i_max, int j_max, double delta_t, double delta_x, double delta_y);
void update_uv_kernel(double* u, double* v, double* F, double* G, double* p, int i_max, int j_max, double delta_t, double delta_x, double delta_y);
void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy);
void update_pressure_bounds_kernel(double* p, int i_max, int j_max);
void reduce_sum_kernel(double* input, double* output, int size);
void extract_value_kernel(double* array, int idx, double* result);
void calculate_residual_norm_kernel(double* res, double* norm, int i_max, int j_max);
void calculate_pressure_norm_kernel(double* p, double* norm, int i_max, int j_max);
void RHS_kernel(double* F, double* G, double* RHS, int i_max, int j_max, double delta_t, double delta_x, double delta_y);
void update_uv_kernel(double* u, double* v, double* F, double* G, double* p, int i_max, int j_max, double delta_t, double delta_x, double delta_y);
void RedSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
void BlackSORKernel(double* p, double* RHS, int i_max, int j_max, double omega, double dxdx, double dydy);
void CalculateResidualKernel(double* p, double* res, double* RHS, int i_max, int j_max, double dxdx, double dydy);
#endif

#endif // CUDA_KERNELS_H