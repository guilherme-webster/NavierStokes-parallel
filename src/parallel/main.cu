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

/**
 * CUDA kernel for updating ghost cells
 */
__global__ void update_ghost_cells_kernel(double* d_p, int i_max, int j_max, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Update left and right ghost cells
    if (i >= 1 && i <= j_max) {
        d_p[i * pitch + 0] = d_p[i * pitch + 1];                  // Left boundary
        d_p[i * pitch + (i_max + 1)] = d_p[i * pitch + i_max];    // Right boundary
    }
    
    // Update top and bottom ghost cells
    if (i >= 1 && i <= i_max) {
        d_p[0 * pitch + i] = d_p[1 * pitch + i];                  // Bottom boundary
        d_p[(j_max + 1) * pitch + i] = d_p[j_max * pitch + i];    // Top boundary
    }
}

/**
 * CUDA kernel for SOR iteration
 */
__global__ void sor_red_kernel(double* d_p, double* d_RHS, int i_max, int j_max, 
                              double omega, double dxdx, double dydy, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max && (i + j) % 2 == 0) {
        d_p[j * pitch + i] = (1.0 - omega) * d_p[j * pitch + i] + 
                             omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) * 
                             ((d_p[j * pitch + (i+1)] + d_p[j * pitch + (i-1)]) / dxdx + 
                              (d_p[(j+1) * pitch + i] + d_p[(j-1) * pitch + i]) / dydy - 
                              d_RHS[j * pitch + i]);
    }
}

/**
 * CUDA kernel for SOR iteration - Black cells (i+j is odd)
 */
__global__ void sor_black_kernel(double* d_p, double* d_RHS, int i_max, int j_max, 
                                double omega, double dxdx, double dydy, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max && (i + j) % 2 == 1) {
        d_p[j * pitch + i] = (1.0 - omega) * d_p[j * pitch + i] + 
                             omega / (2.0 * (1.0 / dxdx + 1.0 / dydy)) * 
                             ((d_p[j * pitch + (i+1)] + d_p[j * pitch + (i-1)]) / dxdx + 
                              (d_p[(j+1) * pitch + i] + d_p[(j-1) * pitch + i]) / dydy - 
                              d_RHS[j * pitch + i]);
    }
}

/**
 * CUDA kernel for calculating residuals
 */
__global__ void calculate_residuals_kernel(double* d_p, double* d_RHS, double* d_res, 
                                         int i_max, int j_max, double dxdx, double dydy, int pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i <= i_max && j <= j_max) {
        d_res[j * pitch + i] = (d_p[j * pitch + (i+1)] - 2.0 * d_p[j * pitch + i] + d_p[j * pitch + (i-1)]) / dxdx + 
                              (d_p[(j+1) * pitch + i] - 2.0 * d_p[j * pitch + i] + d_p[(j-1) * pitch + i]) / dydy - 
                              d_RHS[j * pitch + i];
    }
}

/**
 * CUDA kernel for calculating L2 norm (reduction)
 */
__global__ void l2_norm_kernel(double* d_res, double* d_norm, int i_max, int j_max, int pitch) {
    extern __shared__ double sdata[];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    sdata[tid] = 0.0;
    
    // Load data to shared memory
    if (i <= i_max && j <= j_max) {
        double val = d_res[j * pitch + i];
        sdata[tid] = val * val;
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(d_norm, sdata[0]);
    }
}

/**
 * CUDA version of SOR function
 */
int SOR_CUDA(double** p, int i_max, int j_max, double delta_x, double delta_y, 
             double** res, double** RHS, double omega, double eps, int max_it) {
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    int it = 0;
    cudaError_t cudaStatus;
    
    // Compute matrix dimensions
    size_t pitch;
    int width = i_max + 2;  // Include ghost cells
    int height = j_max + 2; // Include ghost cells
    
    // Allocate device memory
    double *d_p, *d_res, *d_RHS, *d_norm;
    cudaMallocPitch(&d_p, &pitch, width * sizeof(double), height);
    cudaMallocPitch(&d_res, &pitch, width * sizeof(double), height);
    cudaMallocPitch(&d_RHS, &pitch, width * sizeof(double), height);
    cudaMalloc(&d_norm, sizeof(double));
    
    // Convert pitch from bytes to elements
    pitch /= sizeof(double);
    
    // Copy data from host to device
    for (int j = 0; j < height; j++) {
        cudaMemcpy(d_p + j * pitch, p[j], width * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_res + j * pitch, res[j], width * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_RHS + j * pitch, RHS[j], width * sizeof(double), cudaMemcpyHostToDevice);
    }
    
    // Calculate L2 norm of initial p
    double norm_p = L2(p, i_max, j_max);
    
    // Debug: Print initial values
    double sum_p = 0.0, sum_rhs = 0.0;
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            sum_p += fabs(p[i][j]);
            sum_rhs += fabs(RHS[i][j]);
        }
    }
    printf("Initial values: sum_p=%g, sum_rhs=%g, norm_p=%g\n", sum_p, sum_rhs, norm_p);
    
    // Define kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                  (j_max + blockSize.y - 1) / blockSize.y);
    
    dim3 ghostBlockSize(256);
    dim3 ghostGridSize((max(i_max, j_max) + ghostBlockSize.x - 1) / ghostBlockSize.x);
    
    // Temporary host variable for norm
    double h_norm;
    
    // SOR iteration loop
    while (it < max_it) {
        // Update ghost cells
        update_ghost_cells_kernel<<<ghostGridSize, ghostBlockSize>>>(d_p, i_max, j_max, pitch);
        
        // Perform SOR iteration
        sor_red_kernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, omega, dxdx, dydy, pitch);
        cudaDeviceSynchronize(); // Synchronize before black update
        
        sor_black_kernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, omega, dxdx, dydy, pitch);
        cudaDeviceSynchronize(); // Synchronize after black update
    
        // Calculate residuals
        calculate_residuals_kernel<<<gridSize, blockSize>>>(d_p, d_RHS, d_res, i_max, j_max, dxdx, dydy, pitch);
        
        // Calculate L2 norm of residuals
        cudaMemset(d_norm, 0, sizeof(double));
        l2_norm_kernel<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(double)>>>(
            d_res, d_norm, i_max, j_max, pitch);
        
        // Copy norm result back to host
        cudaMemcpy(&h_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost);
        h_norm = sqrt(h_norm / (i_max * j_max));
        
        // Debug: Print progress
        if (it == 5 || it % 100 == 0) {
            // Copy a sample of data back to host to check progress
            cudaMemcpy(p[1], d_p + 1 * pitch, width * sizeof(double), cudaMemcpyDeviceToHost);
            printf("Iteration %d: p[1][1]=%g, p[1][2]=%g, h_norm=%g\n", 
                   it, p[1][1], p[1][2], h_norm);
        }
        
        // Check convergence
        if (h_norm <= eps * (norm_p + 2.0)) {
            // Copy final results back to host
            for (int j = 0; j < height; j++) {
                cudaMemcpy(p[j], d_p + j * pitch, width * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(res[j], d_res + j * pitch, width * sizeof(double), cudaMemcpyDeviceToHost);
            }
            
            // Free device memory
            cudaFree(d_p);
            cudaFree(d_res);
            cudaFree(d_RHS);
            cudaFree(d_norm);
            
            return 0;
        }
        
        it++;
    }
    
    // Copy final results back to host
    for (int j = 0; j < height; j++) {
        cudaMemcpy(p[j], d_p + j * pitch, width * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(res[j], d_res + j * pitch, width * sizeof(double), cudaMemcpyDeviceToHost);
    }
    
    // Free device memory
    cudaFree(d_p);
    cudaFree(d_res);
    cudaFree(d_RHS);
    cudaFree(d_norm);
    
    // Return -1 if maximum iterations were exceeded
    return -1;
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

        // Execute SOR step.
        int sor_result = SOR_CUDA(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it);

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

        if (n % n_print == 0) {
            // Instead of outputting to files, print the data to stdout
            printf("TIMESTEP: %d TIME: %.6f\n", n_out, t);

            // Print some key values from u, v, p matrices
            // For example, print central values and some boundary values
            printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
            printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
            // Add more key values as needed
            n_out++;
        }

        t += delta_t;
        n++;
    }

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_spent);

    // Free grid memory.
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    return 0;
}