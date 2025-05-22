#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3
// Inicializa os arrays CUDA unificados
int initCudaArrays(int i_max, int j_max);

// Libera os arrays CUDA unificados
void freeCudaArrays();

// Função SOR modificada que usa os arrays globais
int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it);



// Utility functions
double max_mat(int i_max, int j_max, double** matrix);
double n_min(int count, ...);
double dp_dx(double** p, int i, int j, double delta_x);
double dp_dy(double** p, int i, int j, double delta_y);

// Função para calcular o erro L2
double L2(double** m, int i_max, int j_max);
void FG(double** F, double** G, double** u, double** v, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma);

// Boundary conditions
int set_noslip(int i_max, int j_max, double** u, double** v, int side);
int set_inflow(int i_max, int j_max, double** u, double** v, int side, double u_fix, double v_fix);


#endif // CUDA_KERNELS_H