// This file handles input and output operations, including reading parameters and writing results to files.

#include "io.h"
#include <stdio.h>
#include <stdlib.h>

void read_parameters(const char* filename, int* problem, double* f, int* i_max, int* j_max, double* a, double* b, double* Re, double* T, double* g_x, double* g_y, double* tau, double* omega, double* epsilon, int* max_it, int* n_print) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening parameter file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", problem);
    fscanf(file, "%lf", f);
    fscanf(file, "%d", i_max);
    fscanf(file, "%d", j_max);
    fscanf(file, "%lf", a);
    fscanf(file, "%lf", b);
    fscanf(file, "%lf", Re);
    fscanf(file, "%lf", T);
    fscanf(file, "%lf", g_x);
    fscanf(file, "%lf", g_y);
    fscanf(file, "%lf", tau);
    fscanf(file, "%lf", omega);
    fscanf(file, "%lf", epsilon);
    fscanf(file, "%d", max_it);
    fscanf(file, "%d", n_print);

    fclose(file);
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