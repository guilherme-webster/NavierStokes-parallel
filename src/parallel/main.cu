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
#include <cuda_runtime.h>

// CUDA kernel for red points (i+j is even)
__global__ void sor_kernel_red(double *p, double *res, const double *RHS, 
                             int i_max, int j_max, double delta_x, double delta_y, 
                             double omega, double *max_res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Check if valid point and is "red" (i+j is even)
    if (i <= i_max && j <= j_max && (i + j) % 2 == 0) {
        int idx = i * (j_max + 2) + j;
        
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        double coeff = 2.0 * (1.0/dx2 + 1.0/dy2);
        
        // SOR update
        double p_old = p[idx];
        double p_new = (1.0 - omega) * p_old + 
                      omega / coeff * 
                      ((p[(i+1) * (j_max + 2) + j] + p[(i-1) * (j_max + 2) + j]) / dx2 +
                       (p[i * (j_max + 2) + (j+1)] + p[i * (j_max + 2) + (j-1)]) / dy2 -
                       RHS[idx]);
        
        // Calculate residual
        double r = fabs(p_new - p_old);
        atomicMax(max_res, r);
        
        p[idx] = p_new;
        res[idx] = r;
    }
}

// CUDA kernel for black points (i+j is odd)
__global__ void sor_kernel_black(double *p, double *res, const double *RHS, 
                               int i_max, int j_max, double delta_x, double delta_y, 
                               double omega, double *max_res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Check if valid point and is "black" (i+j is odd)
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
        int idx = i * (j_max + 2) + j;
        
        double dx2 = delta_x * delta_x;
        double dy2 = delta_y * delta_y;
        double coeff = 2.0 * (1.0/dx2 + 1.0/dy2);
        
        // SOR update
        double p_old = p[idx];
        double p_new = (1.0 - omega) * p_old + 
                      omega / coeff * 
                      ((p[(i+1) * (j_max + 2) + j] + p[(i-1) * (j_max + 2) + j]) / dx2 +
                       (p[i * (j_max + 2) + (j+1)] + p[i * (j_max + 2) + (j-1)]) / dy2 -
                       RHS[idx]);
        
        // Calculate residual
        double r = fabs(p_new - p_old);
        atomicMax(max_res, r);
        
        p[idx] = p_new;
        res[idx] = r;
    }
}

// Parallelized SOR function
int SOR_parallel(double **p, int i_max, int j_max, double delta_x, double delta_y, 
        double **res, double **RHS, double omega, double epsilon, int max_it) {
    
    // Allocate device memory
    double *d_p, *d_res, *d_RHS;
    double *d_max_res;
    double h_max_res;
    
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    cudaMalloc((void**)&d_p, size);
    cudaMalloc((void**)&d_res, size);
    cudaMalloc((void**)&d_RHS, size);
    cudaMalloc((void**)&d_max_res, sizeof(double));
    
    // Flatten 2D arrays to 1D for GPU
    double *h_p_flat = (double*)malloc(size);
    double *h_res_flat = (double*)malloc(size);
    double *h_RHS_flat = (double*)malloc(size);
    
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            h_p_flat[i * (j_max + 2) + j] = p[i][j];
            h_res_flat[i * (j_max + 2) + j] = res[i][j];
            h_RHS_flat[i * (j_max + 2) + j] = RHS[i][j];
        }
    }
    
    cudaMemcpy(d_p, h_p_flat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, h_res_flat, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHS, h_RHS_flat, size, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((i_max + blockDim.x - 1) / blockDim.x, 
                 (j_max + blockDim.y - 1) / blockDim.y);
    
    // Iteration loop
    int it;
    for (it = 0; it < max_it; it++) {
        // Reset max residual
        h_max_res = 0.0;
        cudaMemcpy(d_max_res, &h_max_res, sizeof(double), cudaMemcpyHostToDevice);
        
        // Execute red points first
        sor_kernel_red<<<gridDim, blockDim>>>(d_p, d_res, d_RHS, i_max, j_max, 
                                           delta_x, delta_y, omega, d_max_res);
        
        // Then execute black points (which use updated red points)
        sor_kernel_black<<<gridDim, blockDim>>>(d_p, d_res, d_RHS, i_max, j_max, 
                                             delta_x, delta_y, omega, d_max_res);
        
        // Get max residual from device
        cudaMemcpy(&h_max_res, d_max_res, sizeof(double), cudaMemcpyDeviceToHost);
        
        // Check for convergence
        if (h_max_res < epsilon) {
            break;
        }
    }
    
    // Copy results back to host
    cudaMemcpy(h_p_flat, d_p, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_flat, d_res, size, cudaMemcpyDeviceToHost);
    
    // Convert back to 2D arrays
    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            p[i][j] = h_p_flat[i * (j_max + 2) + j];
            res[i][j] = h_res_flat[i * (j_max + 2) + j];
        }
    }
    
    // Free device memory
    cudaFree(d_p);
    cudaFree(d_res);
    cudaFree(d_RHS);
    cudaFree(d_max_res);
    
    // Free host memory
    free(h_p_flat);
    free(h_res_flat);
    free(h_RHS_flat);
    
    // Return -1 if maximum iterations reached without convergence
    return (it >= max_it) ? -1 : it;
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

    const char* param_file = "parameters.txt"; // file containing parameters
    
    
    // Test if we can open the file directly
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "CUDA: Could not open param_file\n");
        } else {
            param_file = argv[1];
            fclose(fp);
        }
    }
    
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

    while (t < T) {
        printf("%.5f / %.5f\n---------------------\n", t, T);

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

        printf("Conditions set!\n");

        // Calculate F and G.
        FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);

        printf("F, G calculated!\n");

        // RHS of Poisson equation.
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + (G[i][j] - G[i][j-1])/delta_y);
            }
        }
        printf("RHS calculated!\n");

        // Execute parallelized SOR step instead of the original
        clock_t start_sor = clock();
        SOR_parallel(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it)
        printf("SOR complete!\n");
        clock_t end_sor = clock();
        double time_sor = (double)(end_sor - start_sor) / CLOCKS_PER_SEC;
        fprintf(stderr, "%.6f", time_sor);

        // Update velocities.
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                if (i <= i_max - 1) u[i][j] = F[i][j] - delta_t * dp_dx(p, i, j, delta_x);
                if (j <= j_max - 1) v[i][j] = G[i][j] - delta_t * dp_dy(p, i, j, delta_y);
            }
        }
        printf("Velocities updatet!\n");

        if (n % n_print == 0) {
            // Instead of outputting to files, print the data to stdout
            printf("TIMESTEP: %d TIME: %.6f\n", n_out, t);

            // Print some key values from u, v, p matrices
            printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
            printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
            printf("P-CENTER: %.6f\n", p[i_max/2][j_max/2]);

            n_out++;
        }

        t += delta_t;
        n++;
    }

    // Free grid memory.
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    return 0;
}