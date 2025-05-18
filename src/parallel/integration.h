#ifndef INTEGRATION_H
#define INTEGRATION_H

// Function declarations for the SOR method and related functions
int SOR(double** p, int i_max, int j_max, double delta_x, double delta_y, double** res, double** RHS, double omega, double eps, int max_it);
double L2(double** m, int i_max, int j_max);

#endif // INTEGRATION_H