#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// Inicializa os arrays CUDA unificados
int initCudaArrays(int i_max, int j_max);

// Libera os arrays CUDA unificados
void freeCudaArrays();

// Função SOR modificada que usa os arrays globais
int cudaSOR(double** p, int i_max, int j_max, double delta_x, double delta_y, 
            double** res, double** RHS, double omega, double eps, int max_it);

#endif // CUDA_KERNELS_H