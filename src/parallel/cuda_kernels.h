#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3

// Inicializa os arrays CUDA unificados
int initCudaArrays(double** p, double** u, double** v, double** res, double** RHS, int i_max, int j_max);

// Libera os arrays CUDA unificados
void freeCudaArrays();

// SOR principal
int cudaSOR(double** p, double** u, double** v, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it, double** F, double** G, double tau, double Re,
            int problem, double f, double* t, int* n_out, double g_x, double g_y);

// Função utilitária para mínimo
double n_min(int n, double a, double b, double c, double d);

// KERNELS CUDA (definidos em cuda_kernels.cu)
#ifdef __CUDACC__
// CUDA kernel launches
extern "C" {
    template <typename T>
    __global__ void max_mat_kernel(const T* mat, int i_max, int j_max, T* max_val);
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
}
#else
// Para compilação C/C++ normal, declarar apenas os protótipos das funções host
template <typename T>
void max_mat_kernel(const T* mat, int i_max, int j_max, T* max_val);
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
#endif

#endif // CUDA_KERNELS_H