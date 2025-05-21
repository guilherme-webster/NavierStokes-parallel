#include "memory.h"
#include "io.h"
#include "integration.h"
#include "boundaries.h"
#include "cuda_kernels.h"
#include "utils.h"

#include <time.h>
#include <math.h>
#include <stdio.h>

#define MAX_TIMESTEPS 1000  // Fixed number of timesteps instead of time limit

// CUDA kernel for finding the maximum value in an array
__global__ void findMaxKernel(double* array, double* max_val, int i_max, int j_max) {
    // Use block dimensions that match your indexing
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;
    __shared__ double s_max[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    s_max[threadId] = 0.0;
    if (i <= i_max && j <= j_max) {
        s_max[threadId] = fabs(array[i * (j_max + 2) + j]);
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = totalThreads / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            s_max[threadId] = fmax(s_max[threadId], s_max[threadId + s]);
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (threadId == 0) {
        atomicMax((unsigned long long int*)max_val, 
                 __double_as_longlong(s_max[0]));
    }
}

// CUDA kernel to set boundary conditions
__global__ void setBoundaryConditionsKernel(
    double* u, double* v, int i_max, int j_max, 
    int problem, double lid_velocity, double time_val, double f) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // LEFT boundary (no-slip)
    if (j >= 0 && j <= j_max+1 && i == 0) {
        u[i*(j_max+2) + j] = 0.0;
        v[i*(j_max+2) + j] = -v[(i+1)*(j_max+2) + j];
    }
    
    // RIGHT boundary (no-slip)
    if (j >= 0 && j <= j_max+1 && i == i_max+1) {
        u[i*(j_max+2) + j] = 0.0;
        v[i*(j_max+2) + j] = -v[(i-1)*(j_max+2) + j];
    }
    
    // BOTTOM boundary (no-slip)
    if (i >= 0 && i <= i_max+1 && j == 0) {
        u[i*(j_max+2) + j] = -u[i*(j_max+2) + (j+1)];
        v[i*(j_max+2) + j] = 0.0;
    }
    
    // TOP boundary (inflow)
    if (i >= 1 && i <= i_max && j == j_max+1) {
        double u_inflow = (problem == 1) ? lid_velocity : sin(f*time_val);
        u[i*(j_max+2) + j] = 2.0 * u_inflow - u[i*(j_max+2) + j-1];
        v[i*(j_max+2) + j] = 0.0;
    }
}

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
    
    // Initialize parameters & allocate host memory
    init(&problem, &f, &i_max, &j_max, &a, &b, &T, &Re, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);
    printf("Initialized!\n");

    delta_x = a / i_max;
    delta_y = b / j_max;

    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);
    printf("Memory allocated.\n");
    
    // Allocate device memory once for entire simulation
    double *d_u, *d_v, *d_p, *d_F, *d_G, *d_RHS, *d_res;
    double *d_u_max, *d_v_max; // For max value calculation on GPU
    size_t size = (i_max + 2) * (j_max + 2) * sizeof(double);
    
    CUDACHECK(cudaMalloc((void**)&d_u, size));
    CUDACHECK(cudaMalloc((void**)&d_v, size));
    CUDACHECK(cudaMalloc((void**)&d_p, size));
    CUDACHECK(cudaMalloc((void**)&d_F, size));
    CUDACHECK(cudaMalloc((void**)&d_G, size));
    CUDACHECK(cudaMalloc((void**)&d_RHS, size));
    CUDACHECK(cudaMalloc((void**)&d_res, size));
    CUDACHECK(cudaMalloc((void**)&d_u_max, sizeof(double)));
    CUDACHECK(cudaMalloc((void**)&d_v_max, sizeof(double)));
    
    // Flattened host array for initial/final transfers only
    double *h_flat = (double*)malloc(size);
    
    // Initialize device memory with initial conditions (just once)
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            h_flat[i * (j_max + 2) + j] = u[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(d_u, h_flat, size, cudaMemcpyHostToDevice));
    
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            h_flat[i * (j_max + 2) + j] = v[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(d_v, h_flat, size, cudaMemcpyHostToDevice));
    
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            h_flat[i * (j_max + 2) + j] = p[i][j];
        }
    }
    CUDACHECK(cudaMemcpy(d_p, h_flat, size, cudaMemcpyHostToDevice));
    
    // Initialize other arrays with zeros
    CUDACHECK(cudaMemset(d_F, 0, size));
    CUDACHECK(cudaMemset(d_G, 0, size));
    CUDACHECK(cudaMemset(d_RHS, 0, size));
    CUDACHECK(cudaMemset(d_res, 0, size));

    // Grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    dim3 boundaryGridSize((i_max + 2 + blockSize.x - 1) / blockSize.x, 
                        (j_max + 2 + blockSize.y - 1) / blockSize.y);
    
    // Start timer
    clock_t start = clock();
    double t = 0.0;
    
    printf("Starting %d timesteps simulation...\n", MAX_TIMESTEPS);
    
    // Main simulation loop - ALL COMPUTATION STAYS ON GPU
    for (int timestep = 0; timestep < MAX_TIMESTEPS; timestep++) {
        // Reset max values
        double h_u_max = 0.0, h_v_max = 0.0;
        CUDACHECK(cudaMemcpy(d_u_max, &h_u_max, sizeof(double), cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(d_v_max, &h_v_max, sizeof(double), cudaMemcpyHostToDevice));
        
        // Find max values on GPU
        dim3 reduceBlock(16, 16);
        findMaxKernel<<<gridSize, reduceBlock>>>(d_u, d_u_max, i_max, j_max);
        CUDACHECK(cudaGetLastError());
        findMaxKernel<<<gridSize, reduceBlock>>>(d_v, d_v_max, i_max, j_max);
        CUDACHECK(cudaGetLastError());

        // Get max values back to host (small transfer - just 2 doubles)
        CUDACHECK(cudaMemcpy(&h_u_max, d_u_max, sizeof(double), cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpy(&h_v_max, d_v_max, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Calculate timestep on host
        delta_t = tau * n_min(3, Re / 2.0 / (1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y), 
                             delta_x / fabs(h_u_max), delta_y / fabs(h_v_max));
        gamma = fmax(h_u_max * delta_t / delta_x, h_v_max * delta_t / delta_y);

        // Set boundary conditions directly on GPU
        setBoundaryConditionsKernel<<<boundaryGridSize, blockSize>>>(
            d_u, d_v, i_max, j_max, problem, 1.0, t, f);
        CUDACHECK(cudaGetLastError());

        // Print progress occasionally
        if (timestep % 100 == 0) {
            printf("Step %d of %d (%.1f%%)\n", timestep, MAX_TIMESTEPS, 
                   (float)timestep/MAX_TIMESTEPS*100.0);
        }

        // Calculate F and G directly on device
        NavierStokesStepKernel<<<gridSize, blockSize>>>(
            d_u, d_v, NULL, d_F, d_G, NULL, NULL,
            i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma, 0.0, 0.0, 0);
        CUDACHECK(cudaGetLastError());

        // Calculate RHS directly on device
        CalculateRHSKernel<<<gridSize, blockSize>>>(
            d_F, d_G, d_RHS, i_max, j_max, delta_t, delta_x, delta_y);
        CUDACHECK(cudaGetLastError());

        // SOR for pressure directly on device
        double dxdx = delta_x * delta_x;
        double dydy = delta_y * delta_y;
        int it = 0;
        double residual = 1.0;
        
        while (it < max_it && residual > epsilon) {
            // Update boundary conditions for pressure
            UpdateBoundaryKernel<<<boundaryGridSize, blockSize>>>(d_p, i_max, j_max);
            CUDACHECK(cudaGetLastError());
            
            // Red-black SOR iterations
            RedSORKernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, omega, dxdx, dydy);
            CUDACHECK(cudaGetLastError());
            BlackSORKernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, omega, dxdx, dydy);
            CUDACHECK(cudaGetLastError());
            
            // Check convergence occasionally
            if (it % 10 == 0) {
                CalculateResidualKernel<<<gridSize, blockSize>>>(
                    d_p, d_res, d_RHS, i_max, j_max, dxdx, dydy);
                CUDACHECK(cudaGetLastError());
                
                // Compute residual norm - implement on gpu with reduction
                CUDACHECK(cudaMemcpy(h_flat, d_res, size, cudaMemcpyDeviceToHost));
                residual = 0.0;
                for (int i = 1; i <= i_max; i++) {
                    for (int j = 1; j <= j_max; j++) {
                        residual += h_flat[i * (j_max + 2) + j] * h_flat[i * (j_max + 2) + j];
                    }
                }
                residual = sqrt(residual / (i_max * j_max));
            }
            it++;
        }

        // Update velocities directly on device
        UpdateVelocityKernel<<<gridSize, blockSize>>>(
            d_u, d_v, d_F, d_G, d_p, i_max, j_max, delta_t, delta_x, delta_y);

        // Update time
        t += delta_t;
    }
    
    // ONLY NOW: Copy results back to host for final output    
    CUDACHECK(cudaMemcpy(h_flat, d_u, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            u[i][j] = h_flat[i * (j_max + 2) + j];
        }
    }
    
    CUDACHECK(cudaMemcpy(h_flat, d_v, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            v[i][j] = h_flat[i * (j_max + 2) + j];
        }
    }
    
    CUDACHECK(cudaMemcpy(h_flat, d_p, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i <= i_max+1; i++) {
        for (int j = 0; j <= j_max+1; j++) {
            p[i][j] = h_flat[i * (j_max + 2) + j];
        }
    }

    // Stop timer
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Final results
    printf("\n==================== FINAL RESULTS ====================\n");
    printf("Simulation completed: %d steps in %.6f seconds\n", MAX_TIMESTEPS, time_spent);
    printf("Final time reached: %.6f\n", t);
    printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
    printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
    printf("P-CENTER: %.6f\n", p[i_max/2][j_max/2]);
    printf("Average time per step: %.6f seconds\n", time_spent/MAX_TIMESTEPS);
    printf("=========================================================\n");

    // Free all memory
    CUDACHECK(cudaFree(d_u));
    CUDACHECK(cudaFree(d_v));
    CUDACHECK(cudaFree(d_p));
    CUDACHECK(cudaFree(d_F));
    CUDACHECK(cudaFree(d_G));
    CUDACHECK(cudaFree(d_RHS));
    CUDACHECK(cudaFree(d_res));
    CUDACHECK(cudaFree(d_u_max));
    CUDACHECK(cudaFree(d_v_max));
    free(h_flat);
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    
    return 0;
}