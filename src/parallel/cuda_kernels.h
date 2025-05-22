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

// KERNELS CUDA (definidos em cuda_kernels.cu)
// Encontrar valor máximo absoluto em matriz linearizada
template <typename T>
void max_mat_kernel(const T* mat, int i_max, int j_max, T* max_val);

// Derivadas espaciais
void dp_dx_kernel(const double* p, double* out, int i_max, int j_max, double delta_x);
void dp_dy_kernel(const double* p, double* out, int i_max, int j_max, double delta_y);

// Condições de contorno
void set_noslip_linear_kernel(int i_max, int j_max, double* u, double* v, int side);
void set_inflow_linear_kernel(int i_max, int j_max, double* u, double* v, int side, double u_fix, double v_fix);

// Função para calcular F e G (pode ser kernel futuramente)
void FG_linear(double** F, double** G, double* u, double* v, int i_max, int j_max,
        double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma);

// Kernel para cálculo de FG (a ser implementado em cuda_kernels.cu)
void FG_linear_kernel(double* u, double* v, double* F, double* G, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma_factor);

// Kernel para cálculo de RHS (a ser implementado em cuda_kernels.cu)
void RHS_kernel(double* F, double* G, double* RHS, int i_max, int j_max, double delta_t, double delta_x, double delta_y);

// Kernel para atualização de u, v após SOR (a ser implementado em cuda_kernels.cu)
void update_uv_kernel(double* u, double* v, double* F, double* G, double* p, int i_max, int j_max, double delta_t, double delta_x, double delta_y);

#endif // CUDA_KERNELS_H