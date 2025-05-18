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

void FG(double** F, double** G, double** u, double** v, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma) {
    int i, j;
    double du2dx, duvdy, d2udx2, d2udy2;
    double duvdx, dv2dy, d2vdx2, d2vdy2;
    double alpha = 0.9;  // Donor cell parameter
    
    for (i = 1; i <= i_max; i++) {
        for (j = 1; j <= j_max; j++) {
            // F calculation for internal points
            if (i < i_max) {
                // Convection terms
                du2dx = ((u[i][j] + u[i+1][j]) * (u[i][j] + u[i+1][j]) / 4.0 - 
                         (u[i-1][j] + u[i][j]) * (u[i-1][j] + u[i][j]) / 4.0) / delta_x;
                du2dx += alpha * ((fabs(u[i][j] + u[i+1][j]) * (u[i][j] - u[i+1][j]) / 4.0) - 
                                 (fabs(u[i-1][j] + u[i][j]) * (u[i-1][j] - u[i][j]) / 4.0)) / delta_x;
                
                duvdy = ((v[i][j] + v[i+1][j]) * (u[i][j] + u[i][j+1]) / 4.0 - 
                         (v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] + u[i][j]) / 4.0) / delta_y;
                duvdy += alpha * ((fabs(v[i][j] + v[i+1][j]) * (u[i][j] - u[i][j+1]) / 4.0) - 
                                 (fabs(v[i][j-1] + v[i+1][j-1]) * (u[i][j-1] - u[i][j]) / 4.0)) / delta_y;
                
                // Diffusion terms
                d2udx2 = (u[i+1][j] - 2.0 * u[i][j] + u[i-1][j]) / (delta_x * delta_x);
                d2udy2 = (u[i][j+1] - 2.0 * u[i][j] + u[i][j-1]) / (delta_y * delta_y);
                
                F[i][j] = u[i][j] + delta_t * (1.0 / Re * (d2udx2 + d2udy2) - du2dx - duvdy + g_x);
            }
            
            // G calculation for internal points
            if (j < j_max) {
                // Convection terms
                duvdx = ((u[i][j] + u[i][j+1]) * (v[i][j] + v[i+1][j]) / 4.0 - 
                         (u[i-1][j] + u[i-1][j+1]) * (v[i-1][j] + v[i][j]) / 4.0) / delta_x;
                duvdx += alpha * ((fabs(u[i][j] + u[i][j+1]) * (v[i][j] - v[i+1][j]) / 4.0) - 
                                 (fabs(u[i-1][j] + u[i-1][j+1]) * (v[i-1][j] - v[i][j]) / 4.0)) / delta_x;
                
                dv2dy = ((v[i][j] + v[i][j+1]) * (v[i][j] + v[i][j+1]) / 4.0 - 
                         (v[i][j-1] + v[i][j]) * (v[i][j-1] + v[i][j]) / 4.0) / delta_y;
                dv2dy += alpha * ((fabs(v[i][j] + v[i][j+1]) * (v[i][j] - v[i][j+1]) / 4.0) - 
                                 (fabs(v[i][j-1] + v[i][j]) * (v[i][j-1] - v[i][j]) / 4.0)) / delta_y;
                
                // Diffusion terms
                d2vdx2 = (v[i+1][j] - 2.0 * v[i][j] + v[i-1][j]) / (delta_x * delta_x);
                d2vdy2 = (v[i][j+1] - 2.0 * v[i][j] + v[i][j-1]) / (delta_y * delta_y);
                
                G[i][j] = v[i][j] + delta_t * (1.0 / Re * (d2vdx2 + d2vdy2) - duvdx - dv2dy + g_y);
            }
        }
    }
}