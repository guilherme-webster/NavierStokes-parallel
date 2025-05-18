#include "utils.h"
#include <stdarg.h>
#include <math.h>

double max_mat(int i_max, int j_max, double** matrix) {
    double max_val = fabs(matrix[1][1]);
    for (int i = 1; i <= i_max; i++) {
        for (int j = 1; j <= j_max; j++) {
            if (fabs(matrix[i][j]) > max_val) {
                max_val = fabs(matrix[i][j]);
            }
        }
    }
    return max_val;
}

double n_min(int count, ...) {
    va_list args;
    va_start(args, count);
    double min_val = va_arg(args, double);
    
    for (int i = 1; i < count; i++) {
        double val = va_arg(args, double);
        if (val < min_val) {
            min_val = val;
        }
    }
    
    va_end(args);
    return min_val;
}

double dp_dx(double** p, int i, int j, double delta_x) {
    return (p[i+1][j] - p[i][j]) / delta_x;
}

double dp_dy(double** p, int i, int j, double delta_y) {
    return (p[i][j+1] - p[i][j]) / delta_y;
}