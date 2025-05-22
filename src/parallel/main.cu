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

__device__ double atomicMaxDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;
        double currentVal = __longlong_as_double(assumed);
        if (val > currentVal) {
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        } else {
            break;
        }
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// CUDA kernel for finding the maximum value in an array
__global__ void findMaxKernel(double* array, double* max_val, int i_max, int j_max) {
    // Checking thread bounds explicitly
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int threadId = ty * blockDim.x + tx;
    int totalThreads = blockDim.x * blockDim.y;
    
    // Add this safety check to prevent shared memory out-of-bounds
    if (threadId >= 256) return;
    
    __shared__ double s_max[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    s_max[threadId] = 0.0;
    if (i <= i_max+1 && j <= j_max+1) {
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
        atomicMaxDouble(max_val, s_max[0]);
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

__global__ void calculateResidualSumKernel(double* res, double* sum, int i_max, int j_max) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int threadId = ty * blockDim.x + tx;
    int totalThreads = blockDim.x * blockDim.y;
    
    if (threadId >= 256) return; // Safety check
    
    __shared__ double s_sum[256];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    // Initialize shared memory
    s_sum[threadId] = 0.0;
    
    // Load residual squared value
    if (i <= i_max && j <= j_max) {
        double val = res[i * (j_max + 2) + j];
        s_sum[threadId] = val * val;
    }
    
    __syncthreads();
    
    // Parallel reduction sum
    for (int s = totalThreads / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            s_sum[threadId] += s_sum[threadId + s];
        }
        __syncthreads();
    }
    
    // Add block result to global sum using atomic operation
    if (threadId == 0) {
        atomicAdd(sum, s_sum[0]);
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

    double* d_sum;

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

    // Allocate memory
    CUDACHECK(cudaMallocManaged((void**)&d_u, size));
    CUDACHECK(cudaMallocManaged((void**)&d_v, size));
    CUDACHECK(cudaMallocManaged((void**)&d_p, size));
    CUDACHECK(cudaMallocManaged((void**)&d_F, size));
    CUDACHECK(cudaMallocManaged((void**)&d_G, size));
    CUDACHECK(cudaMallocManaged((void**)&d_RHS, size));
    CUDACHECK(cudaMallocManaged((void**)&d_res, size));
    CUDACHECK(cudaMallocManaged((void**)&d_u_max, sizeof(double)));
    CUDACHECK(cudaMallocManaged((void**)&d_v_max, sizeof(double)));
    CUDACHECK(cudaMallocManaged((void**)&d_sum, sizeof(double)));

    memset(d_u, 0, size);
    memset(d_v, 0, size);
    memset(d_p, 0, size);

    // Initialize
    for (int i = 0; i <= i_max; i++) {
        for (int j = 0; j <= j_max; j++) {
            d_u[i * (j_max + 2) + j] = u[i][j];
            d_v[i * (j_max + 2) + j] = v[i][j];
        }
    }

    for (int i = 0; i <= i_max + 1; i++) {
        for (int j = 0; j <= j_max + 1; j++) {
            d_p[i * (j_max + 2) + j] = p[i][j];
        }
    }

    // Zero out other arrays
    memset(d_F, 0, size);
    memset(d_G, 0, size);
    memset(d_RHS, 0, size);
    memset(d_res, 0, size);

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

    // Add a counter to track iterations for debug
    int sor_iterations = 0;
    const int min_iterations = 10;
    
    // Main simulation loop - ALL COMPUTATION STAYS ON GPU
    for (int timestep = 0; timestep < MAX_TIMESTEPS; timestep++) {

        // Set boundary conditions directly on GPU
        setBoundaryConditionsKernel<<<boundaryGridSize, blockSize>>>(d_u, d_v, i_max, j_max, problem, 1.0, t, f);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());

        if (timestep == 0) {
        printf("Boundary check: problem=%d, lid_velocity=%.2f\n", problem, 1.0);
        
        // Check a few boundary points
        printf("Top-mid u(i=%d,j=%d): %.6f\n", 
            i_max/2, j_max+1, d_u[(i_max/2) * (j_max + 2) + (j_max+1)]);
        printf("Top-left u(i=1,j=%d): %.6f\n", 
            j_max+1, d_u[1 * (j_max + 2) + (j_max+1)]);
        printf("Top-right u(i=%d,j=%d): %.6f\n", 
            i_max, j_max+1, d_u[i_max * (j_max + 2) + (j_max+1)]);
        }

        if (timestep < 2) {
        printf("After boundary setup (ts=%d): Top-mid u=%.6f\n", 
            timestep, d_u[(i_max/2) * (j_max + 2) + (j_max+1)]);
        
        // Force a synchronization to make sure value is current
        CUDACHECK(cudaDeviceSynchronize());

        // Reset max values - direct access to managed memory
        *d_u_max = 0.0;
        *d_v_max = 0.0;
        
        // Find max values on GPU
        dim3 reduceBlock(16, 16);
        findMaxKernel<<<gridSize, reduceBlock>>>(d_u, d_u_max, i_max, j_max);
        CUDACHECK(cudaGetLastError());
        
        findMaxKernel<<<gridSize, reduceBlock>>>(d_v, d_v_max, i_max, j_max);
        CUDACHECK(cudaGetLastError());

        // Access max values directly from managed memory - no copies needed
        double safe_u_max = fabs(*d_u_max) < 1e-9 ? 1e-9 : fabs(*d_u_max);
        double safe_v_max = fabs(*d_v_max) < 1e-9 ? 1e-9 : fabs(*d_v_max);
        double delta_t = tau * n_min(3, Re / 2.0 / (1.0 / (delta_x * delta_x) + 1.0 / (delta_y * delta_y)), // Corrected denominator grouping
                            delta_x / safe_u_max, delta_y / safe_v_max);
        gamma = fmax(*d_u_max * delta_t / delta_x, *d_v_max * delta_t / delta_y);

        // Print progress occasionally
        if (timestep % 100 == 0) {
            printf("Step %d of %d (%.1f%%), u_max=%.6f, v_max=%.6f\n", timestep, MAX_TIMESTEPS, 
                (float)timestep/MAX_TIMESTEPS*100.0, *d_u_max, *d_v_max);
        }

        // Calculate F and G directly on device
        NavierStokesStepKernel<<<gridSize, blockSize>>>(
            d_u, d_v, NULL, d_F, d_G, NULL, NULL,
            i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma, 0.0, 0.0, 0);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());

        if (timestep == 0) {
        // Check a few F values
        printf("F at center(i=%d,j=%d): %.6e\n", 
            i_max/2, j_max/2, d_F[(i_max/2) * (j_max + 2) + (j_max/2)]);
        printf("G at center(i=%d,j=%d): %.6e\n", 
            i_max/2, j_max/2, d_G[(i_max/2) * (j_max + 2) + (j_max/2)]);
        }

        // This is critical for correct RHS calculation
        setBoundaryFGKernel<<<boundaryGridSize, blockSize>>>(d_F, d_G, i_max, j_max);
        CUDACHECK(cudaGetLastError());
        CUDACHECK(cudaDeviceSynchronize());

        // Calculate RHS directly on device
        CalculateRHSKernel<<<gridSize, blockSize>>>(
            d_F, d_G, d_RHS, i_max, j_max, delta_t, delta_x, delta_y);
        CUDACHECK(cudaGetLastError());

        // After calculating RHS
            if (timestep < 2) {
                // Check RHS values
                printf("RHS at center(i=%d,j=%d): %.6e\n", 
                    i_max/2, j_max/2, d_RHS[(i_max/2) * (j_max + 2) + (j_max/2)]);
                
                // Check another point near the boundary
                printf("RHS near top(i=%d,j=%d): %.6e\n", 
                    i_max/2, j_max-1, d_RHS[(i_max/2) * (j_max + 2) + (j_max-1)]);
        }

        // SOR for pressure directly on device
        double dxdx = delta_x * delta_x;
        double dydy = delta_y * delta_y;
        int it = 0;
        double residual = 1.0;

        // Add boundary conditions for F and G arrays (missing step)
    // This is critical for correct RHS calculation
    setBoundaryFGKernel<<<boundaryGridSize, blockSize>>>(d_F, d_G, i_max, j_max);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaDeviceSynchronize());

    // Use a more conservative omega value for the first few iterations
    double current_omega = omega;
    if (timestep < 5) current_omega = 1.0; // Use Gauss-Seidel for stability initially

    while (it < min_iterations || (it < max_it && residual > epsilon)) {
        UpdateBoundaryKernel<<<boundaryGridSize, blockSize>>>(d_p, i_max, j_max);
        CUDACHECK(cudaGetLastError());

        // Use a more conservative omega for stability
        RedSORKernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, current_omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());

        UpdateBoundaryKernel<<<boundaryGridSize, blockSize>>>(d_p, i_max, j_max);
        CUDACHECK(cudaGetLastError());

        BlackSORKernel<<<gridSize, blockSize>>>(d_p, d_RHS, i_max, j_max, current_omega, dxdx, dydy);
        CUDACHECK(cudaGetLastError());
        
        // Calculate residual every iteration for early timesteps
        if (timestep < 5 || it % 10 == 0) {
            CalculateResidualKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, i_max, j_max, dxdx, dydy);
            CUDACHECK(cudaGetLastError());

            *d_sum = 0.0;
            calculateResidualSumKernel<<<gridSize, blockSize>>>(d_res, d_sum, i_max, j_max);
            CUDACHECK(cudaGetLastError());
            CUDACHECK(cudaDeviceSynchronize());
            
            residual = sqrt(*d_sum / (i_max * j_max));
            
            if (timestep < 2 && (it < 10 || it % 50 == 0)) {
                printf("SOR it %d: residual=%.8e\n", it, residual);
            }
        }
        
        sor_iterations++;
        it++;
    }

        // Print SOR stats
        if (timestep < 2) {
            printf("SOR completed after %d iterations. Final residual: %.8e\n", 
                sor_iterations, residual);
        }

        // Update velocities directly on device
        UpdateVelocityKernel<<<gridSize, blockSize>>>(
            d_u, d_v, d_F, d_G, d_p, i_max, j_max, delta_t, delta_x, delta_y);
        CUDACHECK(cudaGetLastError());

        // Update time
        t += delta_t;
    }

    // Stop timer
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Final results - can directly access center values from managed memory
    printf("\n==================== FINAL RESULTS ====================\n");
    printf("Simulation completed: %d steps in %.6f seconds\n", MAX_TIMESTEPS, time_spent);
    printf("Final time reached: %.6f\n", t);
    printf("U-CENTER: %.6f\n", d_u[(i_max/2) * (j_max + 2) + (j_max/2)]);
    printf("V-CENTER: %.6f\n", d_v[(i_max/2) * (j_max + 2) + (j_max/2)]);
    printf("P-CENTER: %.6f\n", d_p[(i_max/2) * (j_max + 2) + (j_max/2)]);
    printf("Average time per step: %.6f seconds\n", time_spent/MAX_TIMESTEPS);
    printf("=========================================================\n");

    fprintf(stderr, "%.6f\n", time_spent);

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
    CUDACHECK(cudaFree(d_sum));

    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);

    return 0;
    }
}