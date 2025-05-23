#include "memory.h"

int allocate_memory(double*** u, double*** v, double*** p, double*** res, double*** RHS, double*** F, double*** G, int i_max, int j_max) {
   int i, j;
   *p = (double**) calloc(i_max + 2, sizeof(double*));
   *u = (double**) calloc(i_max + 1, sizeof(double*)); /* u values on the right edges of the ghost cells are not needed */
   *v = (double**) calloc(i_max + 2, sizeof(double*));
   *res = (double**) calloc(i_max + 2, sizeof(double*));
   *RHS = (double**) calloc(i_max + 2, sizeof(double*));
   *F = (double**) calloc(i_max + 2, sizeof(double*));
   *G = (double**) calloc(i_max + 2, sizeof(double*));
   for (i = 0; i < i_max + 2; i++)
   {
       (*p)[i] = (double*) calloc(j_max + 2, sizeof(double));
       (*v)[i] = (double*) calloc(j_max + 1, sizeof(double)); /* v values on the top edges of the ghost cells are not needed */
       if (i < i_max + 1) {
           (*u)[i] = (double*) calloc(j_max + 2, sizeof(double));
       }
       (*res)[i] = (double*) calloc(j_max + 2, sizeof(double));
       (*RHS)[i] = (double*) calloc(j_max + 2, sizeof(double));
       (*F)[i] = (double*) calloc(j_max + 2, sizeof(double));
       (*G)[i] = (double*) calloc(j_max + 2, sizeof(double));
   }
   
   return 0;
}

int free_memory(double*** u, double*** v, double*** p, double*** res, double*** RHS, double*** F, double*** G, int i_max) {
    int i;
    
    // Free each row first for u (different size)
    for (i = 0; i < i_max + 1; i++) {
        if ((*u)[i] != NULL)
            free((*u)[i]);
    }
    
    // Free each row for other arrays
    for (i = 0; i < i_max + 2; i++) {
        if ((*p)[i] != NULL) free((*p)[i]);
        if ((*v)[i] != NULL) free((*v)[i]);
        if ((*res)[i] != NULL) free((*res)[i]);
        if ((*RHS)[i] != NULL) free((*RHS)[i]);
        if ((*F)[i] != NULL) free((*F)[i]);
        if ((*G)[i] != NULL) free((*G)[i]);
    }
    
    // Now free the pointer arrays
    free(*u); free(*v); free(*p);
    free(*res); free(*RHS); free(*F); free(*G);
    
    return 0;
}