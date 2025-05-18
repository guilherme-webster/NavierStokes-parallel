// This file handles input and output operations, including reading parameters and writing results to files.

#include "io.h"
#include <stdio.h>
#include <stdlib.h>

int init(int* problem, double* f, int* i_max, int* j_max, double* a, double* b, 
         double* T, double* R, double* g_x, double* g_y, double* tau, 
         double* omega, double* epsilon, int* max_it, int* n_print, 
         const char* param_file) {
    
    FILE* fp = fopen(param_file, "r");
    if (fp == NULL) {
        printf("Error opening parameter file %s\n", param_file);
        return -1;
    }
    
    char buffer[256];
    
    // Read values, skipping comment lines
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%d", problem);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", f);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%d", i_max);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%d", j_max);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", a);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", b);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", T);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", Re);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", g_x);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", g_y);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", tau);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", omega);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%lf", epsilon);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%d", max_it);
    
    if (fgets(buffer, 256, fp) == NULL) return -1;
    sscanf(buffer, "%d", n_print);
    
    fclose(fp);
    return 0;
}

void output_results(int i_max, int j_max, double** u, double** v, double** p, double t, double a, double b, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening output file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "Time: %.6f\n", t);
    fprintf(file, "Grid Size: %d x %d\n", i_max, j_max);
    fprintf(file, "Velocity (u):\n");
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            fprintf(file, "%.6f ", u[i][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "Velocity (v):\n");
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            fprintf(file, "%.6f ", v[i][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "Pressure (p):\n");
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            fprintf(file, "%.6f ", p[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}