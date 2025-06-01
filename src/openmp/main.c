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
#include <stdarg.h>
#include <stdlib.h>
#include <omp.h>

// Helper functions for GPU computation
#pragma omp declare target
inline double du2_dx_gpu(double* u, double* v, int i, int j, int j_max, double delta_x, double gamma) {
    double stencil1 = 0.5 * (u[i*(j_max+2) + j] + u[(i+1)*(j_max+2) + j]);
    double stencil2 = 0.5 * (u[(i-1)*(j_max+2) + j] + u[i*(j_max+2) + j]);

    double stencil3 = fabs(stencil1) * 0.5 * (u[i*(j_max+2) + j] - u[(i+1)*(j_max+2) + j]);
    double stencil4 = fabs(stencil2) * 0.5 * (u[(i-1)*(j_max+2) + j] - u[i*(j_max+2) + j]);

    return 1/delta_x * (stencil1*stencil1 - stencil2*stencil2) + gamma / delta_x * (stencil3 - stencil4);
}

inline double duv_dy_gpu(double* u, double* v, int i, int j, int j_max, double delta_y, double gamma) {
    double stencil1 = 0.5 * (v[i*(j_max+2) + j] + v[(i+1)*(j_max+2) + j]);
    double stencil2 = 0.5 * (v[i*(j_max+2) + (j-1)] + v[(i+1)*(j_max+2) + (j-1)]);

    double stencil3 = stencil1 * 0.5 * (u[i*(j_max+2) + j] + u[i*(j_max+2) + (j+1)]);
    double stencil4 = stencil2 * 0.5 * (u[i*(j_max+2) + (j-1)] + u[i*(j_max+2) + j]);

    double stencil5 = fabs(stencil1) * 0.5 * (u[i*(j_max+2) + j] - u[i*(j_max+2) + (j+1)]);
    double stencil6 = fabs(stencil2) * 0.5 * (u[i*(j_max+2) + (j-1)] - u[i*(j_max+2) + j]);

    return 1/delta_y * (stencil3 - stencil4) + gamma / delta_y * (stencil5 - stencil6);
}

inline double dv2_dy_gpu(double* u, double* v, int i, int j, int j_max, double delta_y, double gamma) {
    double stencil1 = 0.5 * (v[i*(j_max+2) + j] + v[i*(j_max+2) + (j+1)]);
    double stencil2 = 0.5 * (v[i*(j_max+2) + (j-1)] + v[i*(j_max+2) + j]);

    double stencil3 = fabs(stencil1) * 0.5 * (v[i*(j_max+2) + j] - v[i*(j_max+2) + (j+1)]);
    double stencil4 = fabs(stencil2) * 0.5 * (v[i*(j_max+2) + (j-1)] - v[i*(j_max+2) + j]);

    return 1/delta_y * (stencil1*stencil1 - stencil2*stencil2) + gamma / delta_y * (stencil3 - stencil4);
}

inline double duv_dx_gpu(double* u, double* v, int i, int j, int j_max, double delta_x, double gamma) {
    double stencil1 = 0.5 * (u[i*(j_max+2) + j] + u[i*(j_max+2) + (j+1)]);
    double stencil2 = 0.5 * (u[(i-1)*(j_max+2) + j] + u[(i-1)*(j_max+2) + (j+1)]);

    double stencil3 = stencil1 * 0.5 * (v[i*(j_max+2) + j] + v[(i+1)*(j_max+2) + j]);
    double stencil4 = stencil2 * 0.5 * (v[(i-1)*(j_max+2) + j] + v[i*(j_max+2) + j]);

    double stencil5 = fabs(stencil1) * 0.5 * (v[i*(j_max+2) + j] - v[(i+1)*(j_max+2) + j]);
    double stencil6 = fabs(stencil2) * 0.5 * (v[(i-1)*(j_max+2) + j] - v[i*(j_max+2) + j]);

    return 1/delta_x * (stencil3 - stencil4) + gamma / delta_x * (stencil5 - stencil6);
}

inline double d2u_dx2_gpu(double* u, int i, int j, int j_max, double delta_x) {
    return (u[(i+1)*(j_max+2) + j] - 2 * u[i*(j_max+2) + j] + u[(i-1)*(j_max+2) + j]) / (delta_x*delta_x);
}

inline double d2u_dy2_gpu(double* u, int i, int j, int j_max, double delta_y) {
    return (u[i*(j_max+2) + (j+1)] - 2 * u[i*(j_max+2) + j] + u[i*(j_max+2) + (j-1)]) / (delta_y*delta_y);
}

inline double d2v_dx2_gpu(double* v, int i, int j, int j_max, double delta_x) {
    return (v[(i+1)*(j_max+2) + j] - 2 * v[i*(j_max+2) + j] + v[(i-1)*(j_max+2) + j]) / (delta_x*delta_x);
}

inline double d2v_dy2_gpu(double* v, int i, int j, int j_max, double delta_y) {
    return (v[i*(j_max+2) + (j+1)] - 2 * v[i*(j_max+2) + j] + v[i*(j_max+2) + (j-1)]) / (delta_y*delta_y);
}

inline double dp_dx_gpu(double* p, int i, int j, int j_max, double delta_x) {
    return (p[(i+1)*(j_max+2) + j] - p[(i-1)*(j_max+2) + j]) / (2.0 * delta_x);
}

inline double dp_dy_gpu(double* p, int i, int j, int j_max, double delta_y) {
    return (p[i*(j_max+2) + (j+1)] - p[i*(j_max+2) + (j-1)]) / (2.0 * delta_y);
}
#pragma omp end declare target

int SOR_GPU(double** p, int i_max, int j_max, double delta_x, double delta_y, double** res, double** RHS, 
            double omega, double eps, int max_it) {
    // Constantes úteis
    const double dydy = delta_y * delta_y;
    const double dxdx = delta_x * delta_x;
    const double coeff = omega / (2.0 * (1.0 / dxdx + 1.0 / dydy));
    int it = 0;
    int result = -1;  // Valor padrão se exceder máximo de iterações
    
    // Alocar arrays contíguos para melhor desempenho na GPU
    double* p_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* res_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* RHS_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    
    // Copiar dados para arrays contíguos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            p_flat[i*(j_max+2) + j] = p[i][j];
            if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
                res_flat[i*(j_max+2) + j] = res[i][j];
                RHS_flat[i*(j_max+2) + j] = RHS[i][j];
            }
        }
    }
    
    // Calcular norma L2 inicial
    double norm_p = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:norm_p)
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            norm_p += p_flat[i*(j_max+2) + j] * p_flat[i*(j_max+2) + j];
        }
    }
    norm_p = sqrt(norm_p / i_max / j_max);
    
    // Offload computação para GPU
    #pragma omp target data map(tofrom:p_flat[0:(i_max+2)*(j_max+2)], res_flat[0:(i_max+2)*(j_max+2)], result) \
                         map(to:RHS_flat[0:(i_max+2)*(j_max+2)], norm_p, coeff, i_max, j_max, dxdx, dydy, omega, eps)
    {
        while (it < max_it && result == -1) {
            // Atualizar células fantasma
            #pragma omp target teams distribute parallel for
            for (int j = 1; j <= j_max; j++) {
                p_flat[0*(j_max+2) + j] = p_flat[1*(j_max+2) + j];
                p_flat[(i_max+1)*(j_max+2) + j] = p_flat[i_max*(j_max+2) + j];
            }
            
            #pragma omp target teams distribute parallel for
            for (int i = 1; i <= i_max; i++) {
                p_flat[i*(j_max+2) + 0] = p_flat[i*(j_max+2) + 1];
                p_flat[i*(j_max+2) + (j_max+1)] = p_flat[i*(j_max+2) + j_max];
            }
            
            // Atualizar pontos vermelhos (Red)
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= i_max; i++) {
                for (int j = 1; j <= j_max; j++) {
                    if ((i + j) % 2 == 0) {  // Pontos vermelhos (i+j é par)
                        p_flat[i*(j_max+2) + j] = (1.0 - omega) * p_flat[i*(j_max+2) + j] + 
                            coeff * ((p_flat[(i+1)*(j_max+2) + j] + p_flat[(i-1)*(j_max+2) + j]) / dxdx + 
                                    (p_flat[i*(j_max+2) + (j+1)] + p_flat[i*(j_max+2) + (j-1)]) / dydy - 
                                    RHS_flat[i*(j_max+2) + j]);
                    }
                }
            }
            
            // Garantir que os pontos vermelhos estão atualizados antes dos pretos
            #pragma omp target
            { }  // Kernel vazio para sincronização
            
            // Atualizar pontos pretos (Black)
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= i_max; i++) {
                for (int j = 1; j <= j_max; j++) {
                    if ((i + j) % 2 == 1) {  // Pontos pretos (i+j é ímpar)
                        p_flat[i*(j_max+2) + j] = (1.0 - omega) * p_flat[i*(j_max+2) + j] + 
                            coeff * ((p_flat[(i+1)*(j_max+2) + j] + p_flat[(i-1)*(j_max+2) + j]) / dxdx + 
                                    (p_flat[i*(j_max+2) + (j+1)] + p_flat[i*(j_max+2) + (j-1)]) / dydy - 
                                    RHS_flat[i*(j_max+2) + j]);
                    }
                }
            }
            
            // Calcular resíduos
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= i_max; i++) {
                for (int j = 1; j <= j_max; j++) {
                    res_flat[i*(j_max+2) + j] = 
                        (p_flat[(i+1)*(j_max+2) + j] - 2.0*p_flat[i*(j_max+2) + j] + p_flat[(i-1)*(j_max+2) + j]) / dxdx + 
                        (p_flat[i*(j_max+2) + (j+1)] - 2.0*p_flat[i*(j_max+2) + j] + p_flat[i*(j_max+2) + (j-1)]) / dydy - 
                        RHS_flat[i*(j_max+2) + j];
                }
            }
            
            // Calcular norma L2 dos resíduos
            double res_norm = 0.0;
            #pragma omp target teams distribute parallel for collapse(2) reduction(+:res_norm)
            for (int i = 1; i <= i_max; i++) {
                for (int j = 1; j <= j_max; j++) {
                    res_norm += res_flat[i*(j_max+2) + j] * res_flat[i*(j_max+2) + j];
                }
            }
            
            #pragma omp target
            {
                res_norm = sqrt(res_norm / i_max / j_max);
                if (res_norm <= eps * (norm_p + 1.5)) {
                    result = 0;  // Convergiu
                }
            }
            
            it++;
        }
    }
    
    // Copiar dados de volta para arrays 2D
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            p[i][j] = p_flat[i*(j_max+2) + j];
            if (i >= 1 && i <= i_max && j >= 1 && j <= j_max) {
                res[i][j] = res_flat[i*(j_max+2) + j];
            }
        }
    }
    
    // Liberar arrays temporários
    free(p_flat);
    free(res_flat);
    free(RHS_flat);
    
    return result;
}


double max_mat_gpu(int i_max, int j_max, double** m) {
    // Alocar array contíguo para GPU
    double* m_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    
    // Copiar dados para array contíguo
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            m_flat[i*(j_max+2) + j] = m[i][j];
        }
    }
    
    double max_val = 0.0;
    
    // Computação na GPU
    #pragma omp target data map(to: m_flat[0:(i_max+2)*(j_max+2)]) map(tofrom: max_val)
    {
        #pragma omp target teams distribute parallel for collapse(2) reduction(max:max_val)
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                if (fabs(m_flat[i*(j_max+2) + j]) > max_val) {
                    max_val = fabs(m_flat[i*(j_max+2) + j]);
                }
            }
        }
    }
    
    free(m_flat);
    return max_val;
}


double n_min_gpu(int n, ...) {
    va_list args;
    va_start(args, n);
    double min_val = va_arg(args, double);
    
    for (int i = 1; i < n; i++) {
        double val = va_arg(args, double);
        if (val < min_val) {
            min_val = val;
        }
    }
    
    va_end(args);
    return min_val;
}


int FG_GPU(double** F, double** G, double** u, double** v, int i_max, int j_max, 
           double Re, double g_x, double g_y, double delta_t, double delta_x, 
           double delta_y, double gamma) {
    
    // Alocar arrays contíguos
    double* F_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* G_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* u_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* v_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    
    // Copiar dados para arrays contíguos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            u_flat[i*(j_max+2) + j] = u[i][j];
            v_flat[i*(j_max+2) + j] = v[i][j];
            if (i <= i_max && j <= j_max) {
                F_flat[i*(j_max+2) + j] = F[i][j];
                G_flat[i*(j_max+2) + j] = G[i][j];
            }
        }
    }
    
    // Computação na GPU
    #pragma omp target data map(to: u_flat[0:(i_max+2)*(j_max+2)], v_flat[0:(i_max+2)*(j_max+2)], i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma) \
                         map(tofrom: F_flat[0:(i_max+2)*(j_max+2)], G_flat[0:(i_max+2)*(j_max+2)])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                if (i <= i_max - 1) {
                    F_flat[i*(j_max+2) + j] = u_flat[i*(j_max+2) + j] +
                                delta_t * (1.0/Re * (d2u_dx2_gpu(u_flat, i, j, j_max, delta_x) + 
                                                     d2u_dy2_gpu(u_flat, i, j, j_max, delta_y)) -
                                          du2_dx_gpu(u_flat, v_flat, i, j, j_max, delta_x, gamma) -
                                          duv_dy_gpu(u_flat, v_flat, i, j, j_max, delta_y, gamma) +
                                          g_x);
                }

                if (j <= j_max - 1) {
                    G_flat[i*(j_max+2) + j] = v_flat[i*(j_max+2) + j] +
                                delta_t * (1.0/Re * (d2v_dx2_gpu(v_flat, i, j, j_max, delta_x) + 
                                                     d2v_dy2_gpu(v_flat, i, j, j_max, delta_y)) -
                                          duv_dx_gpu(u_flat, v_flat, i, j, j_max, delta_x, gamma) -
                                          dv2_dy_gpu(u_flat, v_flat, i, j, j_max, delta_y, gamma) +
                                          g_y);
                }
            }
        }
    }
    
    // Copiar dados de volta
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            if (i <= i_max && j <= j_max) {
                F[i][j] = F_flat[i*(j_max+2) + j];
                G[i][j] = G_flat[i*(j_max+2) + j];
            }
        }
    }
    
    // Liberar memória
    free(F_flat);
    free(G_flat);
    free(u_flat);
    free(v_flat);
    
    return 0;
}


void calculate_RHS_GPU(double** RHS, double** F, double** G, int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    // Alocar arrays contíguos
    double* RHS_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* F_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* G_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    
    // Copiar dados para arrays contíguos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            if (i <= i_max && j <= j_max) {
                F_flat[i*(j_max+2) + j] = F[i][j];
                G_flat[i*(j_max+2) + j] = G[i][j];
                RHS_flat[i*(j_max+2) + j] = RHS[i][j];
            }
        }
    }
    
    // Computação na GPU
    #pragma omp target data map(to: F_flat[0:(i_max+2)*(j_max+2)], G_flat[0:(i_max+2)*(j_max+2)], i_max, j_max, delta_t, delta_x, delta_y) \
                         map(tofrom: RHS_flat[0:(i_max+2)*(j_max+2)])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                RHS_flat[i*(j_max+2) + j] = 1.0 / delta_t * (
                    (F_flat[i*(j_max+2) + j] - F_flat[(i-1)*(j_max+2) + j]) / delta_x + 
                    (G_flat[i*(j_max+2) + j] - G_flat[i*(j_max+2) + (j-1)]) / delta_y
                );
            }
        }
    }
    
    // Copiar dados de volta
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            RHS[i][j] = RHS_flat[i*(j_max+2) + j];
        }
    }
    
    // Liberar memória
    free(RHS_flat);
    free(F_flat);
    free(G_flat);
}


void update_velocities_GPU(double** u, double** v, double** F, double** G, double** p, 
                          int i_max, int j_max, double delta_t, double delta_x, double delta_y) {
    // Alocar arrays contíguos
    double* u_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* v_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* F_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* G_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* p_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    
    // Copiar dados para arrays contíguos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            u_flat[i*(j_max+2) + j] = u[i][j];
            v_flat[i*(j_max+2) + j] = v[i][j];
            p_flat[i*(j_max+2) + j] = p[i][j];
            if (i <= i_max && j <= j_max) {
                F_flat[i*(j_max+2) + j] = F[i][j];
                G_flat[i*(j_max+2) + j] = G[i][j];
            }
        }
    }
    
        
    // Computação na GPU
    #pragma omp target data map(to: F_flat[0:(i_max+2)*(j_max+2)], G_flat[0:(i_max+2)*(j_max+2)], p_flat[0:(i_max+2)*(j_max+2)], i_max, j_max, delta_t, delta_x, delta_y) \
                         map(tofrom: u_flat[0:(i_max+2)*(j_max+2)], v_flat[0:(i_max+2)*(j_max+2)])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 1; i <= i_max; i++) {
            for (int j = 1; j <= j_max; j++) {
                if (i <= i_max - 1) {
                    u_flat[i*(j_max+2) + j] = F_flat[i*(j_max+2) + j] - 
                                             delta_t * dp_dx_gpu(p_flat, i, j, j_max, delta_x);
                }
                
                if (j <= j_max - 1) {
                    v_flat[i*(j_max+2) + j] = G_flat[i*(j_max+2) + j] - 
                                             delta_t * dp_dy_gpu(p_flat, i, j, j_max, delta_y);
                }
            }
        }
    }
    
    // Copiar dados de volta
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            u[i][j] = u_flat[i*(j_max+2) + j];
            v[i][j] = v_flat[i*(j_max+2) + j];
        }
    }
    
    // Liberar memória
    free(u_flat);
    free(v_flat);
    free(F_flat);
    free(G_flat);
    free(p_flat);
}



int set_inflow_GPU(int i_max, int j_max, double** u, double** v, int side, double u_fix, double v_fix) {
    // Alocar arrays contíguos
    double* u_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    double* v_flat = (double*)malloc((i_max+2) * (j_max+2) * sizeof(double));
    
    // Copiar dados para arrays contíguos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            u_flat[i*(j_max+2) + j] = u[i][j];
            v_flat[i*(j_max+2) + j] = v[i][j];
        }
    }
    
    // Computação na GPU - usar um kernel para cada lado
    #pragma omp target data map(tofrom: u_flat[0:(i_max+2)*(j_max+2)], v_flat[0:(i_max+2)*(j_max+2)]) \
                         map(to: i_max, j_max, side, u_fix, v_fix)
    {
        switch(side) {
            case 0: // TOP
                #pragma omp target teams distribute parallel for
                for (int i = 1; i <= i_max; i++) {
                    v_flat[i*(j_max+2) + j_max] = v_fix;
                    u_flat[i*(j_max+2) + (j_max+1)] = 2 * u_fix - u_flat[i*(j_max+2) + j_max];
                }
                break;
                
            case 1: // BOTTOM
                #pragma omp target teams distribute parallel for
                for (int i = 1; i <= i_max; i++) {
                    v_flat[i*(j_max+2) + 0] = v_fix;
                    u_flat[i*(j_max+2) + 0] = 2 * u_fix - u_flat[i*(j_max+2) + 1];
                }
                break;
                
            case 2: // LEFT
                #pragma omp target teams distribute parallel for
                for (int j = 1; j <= j_max; j++) {
                    u_flat[0*(j_max+2) + j] = u_fix;
                    v_flat[0*(j_max+2) + j] = 2 * v_fix - v_flat[1*(j_max+2) + j];
                }
                break;
                
            case 3: // RIGHT
                #pragma omp target teams distribute parallel for
                for (int j = 1; j <= j_max; j++) {
                    u_flat[i_max*(j_max+2) + j] = u_fix;
                    v_flat[(i_max+1)*(j_max+2) + j] = 2 * v_fix - v_flat[i_max*(j_max+2) + j];
                }
                break;
        }
    }
    
    // Copiar dados de volta
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            u[i][j] = u_flat[i*(j_max+2) + j];
            v[i][j] = v_flat[i*(j_max+2) + j];
        }
    }
    
    // Liberar memória
    free(u_flat);
    free(v_flat);
    
    return 0;
}


int set_noslip_GPU(int i_max, int j_max, double** u, double** v, int side) {
    return set_inflow_GPU(i_max, j_max, u, v, side, 0.0, 0.0);
}



/**
 * @brief Main function.
 * 
 * This is the main function.
 * @return 0 on exit.
 */

int main(int argc, char* argv[])
{
    // Grid pointers.
	double** u;     // velocity x-component
	double** v;     // velocity y-component
	double** p;     // pressure

    double** F;     // F term
    double** G;     // G term
    double** res;   // SOR residuum
    double** RHS;   // RHS of poisson equation

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

    // default parameter file
    const char* param_file = "parameters.txt";

    // If a parameter file is provided as command line argument, use it
    if (argc > 1) {
        param_file = argv[1];
    }
    
    // Initialize all parameters.
	init(&problem, &f, &i_max, &j_max, &a, &b, &Re, &T, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);

    // Set step size in space.
    delta_x = a / i_max;
    delta_y = b / j_max;

    // Allocate memory for grids.
    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);

    // Time loop.
    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();
    double time_sor = 0.0;
    while (t < T) {
        // Adaptive stepsize and weight factor for Donor-Cell
        double u_max = max_mat_gpu(i_max, j_max, u);
        double v_max = max_mat_gpu(i_max, j_max, v);
        delta_t = tau * n_min_gpu(3, Re / 2.0 / (1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y), 
                           delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);

        // Set boundary conditions.
        if (problem == 1) {
            set_noslip_GPU(i_max, j_max, u, v, 2); // LEFT
            set_noslip_GPU(i_max, j_max, u, v, 3); // RIGHT
            set_noslip_GPU(i_max, j_max, u, v, 1); // BOTTOM
            set_inflow_GPU(i_max, j_max, u, v, 0, 1.0, 0.0); // TOP
        } else if (problem == 2) {
            set_noslip_GPU(i_max, j_max, u, v, 2); // LEFT
            set_noslip_GPU(i_max, j_max, u, v, 3); // RIGHT
            set_noslip_GPU(i_max, j_max, u, v, 1); // BOTTOM
            set_inflow_GPU(i_max, j_max, u, v, 0, sin(f*t), 0.0); // TOP        
        } else {
            printf("Unknown probem type (see parameters.txt).\n");
            exit(EXIT_FAILURE);
        }

        // Calculate F and G.
        FG_GPU(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);

        // RHS of Poisson equation.
        calculate_RHS_GPU(RHS, F, G, i_max, j_max, delta_t, delta_x, delta_y);
        
        // Solve Poisson equation for pressure
        clock_t start_sor = clock();
        SOR_GPU(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it);
        clock_t end_sor = clock();
        time_sor += (double)(end_sor - start_sor) / CLOCKS_PER_SEC;

        // Update velocities.
        update_velocities_GPU(u, v, F, G, p, i_max, j_max, delta_t, delta_x, delta_y);

        t += delta_t;
        n++;
    }
    printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
    printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_sor);

    // Free grid memory.
    free_memory(&u, &v, &p, &res, &RHS, &F, &G);
    return 0;
}