#ifndef BOUNDARIES_H
#define BOUNDARIES_H

// Function declarations for setting boundary conditions
int set_noslip(int i_max, int j_max, double** u, double** v, int side);
int set_inflow(int i_max, int j_max, double** u, double** v, int side, double u_fix, double v_fix);

#endif // BOUNDARIES_H