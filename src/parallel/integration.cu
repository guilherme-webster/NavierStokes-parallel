#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "integration.h"
#include "cuda_kernels.h"

double L2(double** m, int i_max, int j_max) {
    double norm = 0.0;
    int i, j;
    for (i = 1; i <= i_max; i++) {
        for(j = 1; j <= j_max; j++) {
            norm += m[i][j] * m[i][j];
        }
    }
    return sqrt(norm / i_max / j_max);
}

int SOR(double** p, int i_max, int j_max, double delta_x, double delta_y, double** res, double** RHS, double omega, double eps, int max_it) {
    int it = 0;
    double dydy = delta_y * delta_y;
    double dxdx = delta_x * delta_x;
    double norm_p = L2(p, i_max, j_max);    // L2 norm of grid.

    // Allocate device memory for p, res, and RHS
    double *d_p, *d_res, *d_RHS;
    cudaMalloc((void**)&d_p, (i_max + 2) * (j_max + 2) * sizeof(double));
    cudaMalloc((void**)&d_res, (i_max + 2) * (j_max + 2) * sizeof(double));
    cudaMalloc((void**)&d_RHS, (i_max + 2) * (j_max + 2) * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_p, &p[0][0], (i_max + 2) * (j_max + 2) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RHS, &RHS[0][0], (i_max + 2) * (j_max + 2) * sizeof(double), cudaMemcpyHostToDevice);

    while (it < max_it) {
        // Fill ghost cells with values of neighboring cells for new iteration step.
        for (int j = 1; j <= j_max; j++) {
            p[0][j] = p[1][j];
            p[i_max + 1][j] = p[i_max][j];
        }
        for (int i = 1; i <= i_max; i++) {
            p[i][0] = p[i][1];
            p[i][j_max + 1] = p[i][j_max];
        }

        // Launch CUDA kernel for SOR iteration
        dim3 blockSize(16, 16);
        dim3 gridSize((i_max + 1 + blockSize.x - 1) / blockSize.x, (j_max + 1 + blockSize.y - 1) / blockSize.y);
        SORKernel<<<gridSize, blockSize>>>(d_p, d_res, d_RHS, i_max, j_max, dxdx, dydy, omega);

        // Copy result back to host
        cudaMemcpy(&res[0][0], d_res, (i_max + 2) * (j_max + 2) * sizeof(double), cudaMemcpyDeviceToHost);

        // Abortion condition.
        if (L2(res, i_max, j_max) <= eps * (norm_p + 0.01)) {
            cudaFree(d_p);
            cudaFree(d_res);
            cudaFree(d_RHS);
            return 0;
        }

        it++;
    }

    // Free device memory
    cudaFree(d_p);
    cudaFree(d_res);
    cudaFree(d_RHS);

    // Return -1 if maximum iterations were exceeded.
    return -1;
}