#ifndef INTEGRATION_H
#define INTEGRATION_H

// Function declarations for the SOR method and related functions
double L2(double** m, int i_max, int j_max);
void FG(double** F, double** G, double** u, double** v, int i_max, int j_max, double Re, double g_x, double g_y, double delta_t, double delta_x, double delta_y, double gamma);

#endif // INTEGRATION_H