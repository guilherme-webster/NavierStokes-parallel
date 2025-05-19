#include "memory.h"
#include "io.h"
#include "integration.h"
#include "boundaries.h"
#include "cuda_kernels.h"
#include "utils.h"

#include <time.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    double** u;     
    double** v;     
    double** p;     

    double** F;     
    double** G;     
    double** res;   
    double** RHS;   

    int i_max, j_max;                   
    double a, b;                        
    double Re;                          
    double delta_t, delta_x, delta_y;   
    double gamma;                       
    double T;                           
    double g_x;                         
    double g_y;                         
    double tau;                         
    double omega;                       
    double epsilon;                     
    int max_it;                         
    int n_print;                        
    int problem;                        
    double f;                           

    const char* param_file = "parameters.txt"; 

    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "CUDA: Could not open param_file\n");
        } else {
            param_file = argv[1];
            fclose(fp);
        }
    }
    
    init(&problem, &f, &i_max, &j_max, &a, &b, &T, &Re, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);
    printf("Initialized!\n");

    delta_x = a / i_max;
    delta_y = b / j_max;

    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);
    printf("Memory allocated.\n");

    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();

    while (t < T) {
        printf("%.5f / %.5f\n---------------------\n", t, T);

        double u_max = max_mat(i_max, j_max, u);
        double v_max = max_mat(i_max, j_max, v);
        delta_t = tau * n_min(3, Re / 2.0 / ( 1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y ), delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);

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
        } else {
            printf("Unknown problem type (see parameters.txt).\n");
            exit(EXIT_FAILURE);
        }

        printf("Conditions set!\n");

        //FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);
        cudaFG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);
        printf("F, G calculated!\n");

        // for (i = 1; i <= i_max; i++ ) {
        //     for (j = 1; j <= j_max; j++) {
        //         RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + (G[i][j] - G[i][j-1])/delta_y);
        //     }
        // }
        cudaCalculateRHS(F, G, RHS, i_max, j_max, delta_t, delta_x, delta_y);
        printf("RHS calculated!\n");

        // Call the CUDA kernel for SOR
        cudaSOR(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it);
        printf("SOR complete!\n");

        // for (i = 1; i <= i_max; i++ ) {
        //     for (j = 1; j <= j_max; j++) {
        //         if (i <= i_max - 1) u[i][j] = F[i][j] - delta_t * dp_dx(p, i, j, delta_x);
        //         if (j <= j_max - 1) v[i][j] = G[i][j] - delta_t * dp_dy(p, i, j, delta_y);
        //     }
        // }
        cudaUpdateVelocity(u, v, F, G, p, i_max, j_max, delta_t, delta_x, delta_y);
        printf("Velocities updated!\n");

        if (n % n_print == 0) {
            printf("TIMESTEP: %d TIME: %.6f\n", n_out, t);
            printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
            printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
            printf("P-CENTER: %.6f\n", p[i_max/2][j_max/2]);
            n_out++;
        }

        t += delta_t;
        n++;
    }

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_spent);

    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    return 0;
}