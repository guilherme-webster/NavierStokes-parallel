#include "io.h"
#include "memory.h"
#include "integration.h"


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

int init(int* problem, double* f, int* i_max, int* j_max, double* a, double* b, double* Re, double* T, double* g_x, double* g_y, double* tau, double* omega, double* epsilon, int* max_it, int* n_print, const char* filename)
{
    char buffer[256];
	FILE *fp;
    fp = fopen(filename, "r");

   if (fp == NULL)
   {
      perror("Error while opening the file.\n");
      exit(EXIT_FAILURE);
   }

    // Read file line-by-line to buffer and extract the values.
    fgets(buffer, 256, fp);
    sscanf(buffer, "%d", problem);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", f);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%d", i_max);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%d", j_max);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", a);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", b);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", T);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", Re);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", g_x);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", g_y);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", tau);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", omega);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%lf", epsilon);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%d", max_it);
    fgets(buffer, 256, fp);
    sscanf(buffer, "%d", n_print);

    fclose(fp);

    return 0;
}

int output(int i_max, int j_max, double** u, double** v, double** p, double t, double a, double b, const char* prefix)
{
	int i, j;

    char fname_u[64];
    char fname_v[64];
    char fname_p[64];
    char fname_t[64];

    strcpy(fname_u, prefix);
    strcpy(fname_v, prefix);
    strcpy(fname_p, prefix);

    strcat(fname_u, "_u.txt");
    strcat(fname_v, "_v.txt");
    strcat(fname_p, "_p.txt");

    FILE* fp_u,* fp_v, * fp_p;
    fp_u = fopen(fname_u, "w");
    fp_v = fopen(fname_v, "w");
    fp_p = fopen(fname_p, "w");

    if (fp_u == NULL || fp_v == NULL || fp_p == NULL)
    {
      perror("Error opening one ore more output files. Make sure the directory 'out' exists.\n");
      exit(EXIT_FAILURE);
    }

    // Header
    fprintf(fp_p, "%.5f\n", t);
    fprintf(fp_p, "%.5f\n", a);
    fprintf(fp_p, "%.5f\n", b);

    fprintf(fp_u, "%.5f\n", t);
    fprintf(fp_u, "%.5f\n", a);
    fprintf(fp_u, "%.5f\n", b);

    fprintf(fp_v, "%.5f\n", t);
    fprintf(fp_v, "%.5f\n", a);
    fprintf(fp_v, "%.5f\n", b);

    // Rows first, then columns
    for (j = 0; j < j_max + 2; j++) {
        for (i = 0; i < i_max + 2; i++) {
            if (i < i_max + 1) fprintf(fp_u, "%.5f ", u[i][j]); // 5 decimal places
            if (j < j_max + 1) fprintf(fp_v, "%.5f ", v[i][j]);
            fprintf(fp_p, "%.5f ", p[i][j]);
        }
        fprintf(fp_u, "\n"); 
        fprintf(fp_v, "\n");
        fprintf(fp_p, "\n");
    }

    fclose(fp_u);
    fclose(fp_v);
    fclose(fp_p);

    printf("Output created!\n");	
    return 0;
}

double max_mat(int i_max, int j_max, double** u)
{
   	double max = u[0][0];
   	int i,j;

   	for (i = 1; i <= i_max; i++)
   	{
       	for (j = 1; j <= j_max; j++)
       	{
           	if (max < u[i][j])
           	{
               	max = u[i][j];
           	}
    	} 
    }

	return max;
}

double n_min(int num, ...)
{
    va_list valist;
    va_start(valist, num);

   	double min = va_arg(valist, double);
   	int i;

   	for (i = 0; i < num-1; i++)
   	{
        double val = va_arg(valist, double);

        if (min > val)
        {
            min = val;
        }
    	 
    }
    
	return min;
}