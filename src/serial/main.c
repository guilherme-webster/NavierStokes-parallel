#include "memory.h"
#include "io.h"
#include "integration.h"
#include "boundaries.h"

#include <time.h>
#include <math.h>
#include <stdio.h>

#define MAX_TIMESTEPS 1000  // Fixed number of timesteps instead of time limit

int main(int argc, char* argv[])
{
    // Grid pointers and parameter declarations
    // [unchanged code...]

    // Initialize all parameters.
    init(&problem, &f, &i_max, &j_max, &a, &b, &Re, &T, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);
    printf("Initialized!\n");

    delta_x = a / i_max;
    delta_y = b / j_max;

    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);
    printf("Memory allocated.\n");

    // Initialize time measurement
    clock_t start = clock();
    
    // Fixed number of timesteps instead of time limit
    int timestep;
    double t = 0.0;

    printf("Starting %d timesteps simulation...\n", MAX_TIMESTEPS);
    
    for (timestep = 0; timestep < MAX_TIMESTEPS; timestep++) {
        // Adaptive stepsize and weight factor for Donor-Cell
        double u_max = max_mat(i_max, j_max, u);
        double v_max = max_mat(i_max, j_max, v);
        delta_t = tau * n_min(3, Re / 2.0 / (1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y), 
                             delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);

        // Set boundary conditions based on problem type
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

        // Print progress only every 100 timesteps
        if (timestep % 100 == 0) {
            printf("Step %d of %d (%.1f%%)\n", timestep, MAX_TIMESTEPS, 
                   (float)timestep/MAX_TIMESTEPS*100);
        }

        // Calculate F and G
        FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);

        // RHS of Poisson equation
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + (G[i][j] - G[i][j-1])/delta_y);
            }
        }

        // Execute SOR step
        SOR(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it);

        // Update velocities
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                if (i <= i_max - 1) u[i][j] = F[i][j] - delta_t * dp_dx(p, i, j, delta_x);
                if (j <= j_max - 1) v[i][j] = G[i][j] - delta_t * dp_dy(p, i, j, delta_y);
            }
        }

        // Update time
        t += delta_t;
    }

    // Stop timer after all timesteps
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Print final results only once at the end
    printf("\n==================== FINAL RESULTS ====================\n");
    printf("Simulation completed: %d steps in %.6f seconds\n", MAX_TIMESTEPS, time_spent);
    printf("Final time reached: %.6f\n", t);
    printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
    printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
    printf("P-CENTER: %.6f\n", p[i_max/2][j_max/2]);
    printf("Average time per step: %.6f seconds\n", time_spent/MAX_TIMESTEPS);
    printf("=========================================================\n");

    // Write timing to stderr for easy extraction
    fprintf(stderr, "%.6f", time_spent);

    // Free memory
    free_memory(&u, &v, &p, &res, &RHS, &F, &G);
    return 0;
}