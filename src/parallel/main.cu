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
#include "kernel.h"

#include <time.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief Main function.
 * 
 * This is the main function.
 * @return 0 on exit.
 */

int main(int argc, char* argv[])
{
    printf("DEBUG: Programa iniciado\n");
    printf("DEBUG: argc = %d\n", argc);
    
    for(int i = 0; i < argc; i++) {
        printf("DEBUG: argv[%d] = %s\n", i, argv[i]);
    }
    
    char* param_file = "parameters.txt";
    if (argc > 1) {
        param_file = argv[1];
    }
    
    printf("DEBUG: Tentando abrir arquivo: %s\n", param_file);
    
    FILE* fp = fopen(param_file, "r");
    if (fp == NULL) {
        printf("DEBUG: ERRO ao abrir arquivo\n");
        perror("fopen");
        return -1;
    }
    
    printf("DEBUG: Arquivo aberto com sucesso\n");
    fclose(fp);
    
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
    printf("Initialized!\n");

    // Set step size in space.
    delta_x = a / i_max;
    delta_y = b / j_max;

    // Allocate memory for grids.
    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);
    printf("Memory allocated.\n");

    // Time loop.
    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();
    int total_point;
    BoundaryPoint* boundary_points = generate_boundary_indices(i_max, j_max, &total_point);

    init_memory(i_max, j_max, boundary_points, total_point, &tau, &Re, &g_x, &g_y, &omega, &epsilon, &max_it);

    while (t < T) {
        printf("%.5f / %.5f\n---------------------\n", t, T);

        t += orchestration(i_max, j_max);

        n++;
    }

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_spent);

    // Free grid memory.
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    return 0;
}