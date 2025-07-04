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
        double u_max = max_mat(i_max, j_max, u);
        double v_max = max_mat(i_max, j_max, v);
    	delta_t = tau * n_min(3, Re / 2.0 / ( 1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y ), delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);

        // Set boundary conditions.
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
            printf("Unknown probem type (see parameters.txt).\n");
            exit(EXIT_FAILURE);
        }


        // Calculate F and G.
        FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);


        // RHS of Poisson equation.
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + (G[i][j] - G[i][j-1])/delta_y);
            }
        }
        //clock_t start_UVA = clock();
        clock_t start_sor = clock();
        SOR(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it);
        clock_t end_sor = clock();
        time_sor += (double)(end_sor - start_sor) / CLOCKS_PER_SEC;
        //clock_t end_UVA = clock();
        //double time_UVA = (double)(end_UVA - start_UVA) / CLOCKS_PER_SEC;
        //fprintf(stderr, "SOR UVA time: %.6f\n", time_UVA);

        // Update velocities.
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                if (i <= i_max - 1) u[i][j] = F[i][j] - delta_t * dp_dx(p, i, j, delta_x);
                if (j <= j_max - 1) v[i][j] = G[i][j] - delta_t * dp_dy(p, i, j, delta_y);
            }
        }
        // Print to file every ..th step.
        // if (n % n_print == 0) {
        //     char out_prefix[12];
        //     sprintf(out_prefix, "out/%d", n_out);
        //     output(i_max, j_max, u, v, p, t, a, b, out_prefix);
        //     n_out++;
        // }

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