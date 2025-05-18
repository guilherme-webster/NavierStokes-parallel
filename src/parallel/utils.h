#ifndef UTILS_H
#define UTILS_H

// Boundary definitions
#define TOP 0
#define BOTTOM 1
#define LEFT 2
#define RIGHT 3

// Utility functions
double max_mat(int i_max, int j_max, double** matrix);
double n_min(int count, ...);
double dp_dx(double** p, int i, int j, double delta_x);
double dp_dy(double** p, int i, int j, double delta_y);

#endif // UTILS_H