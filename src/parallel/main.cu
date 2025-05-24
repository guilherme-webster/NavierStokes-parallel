#include "memory.h"
#include "io.h"
#include "cuda_kernels.h"
#include <time.h>
#include <math.h>
#include <stdio.h>

// Estrutura para armazenar valores finais para impressão
typedef struct {
    double u_center;
    double v_center;
    double p_center;
    double final_time;
    int iterations;
    double execution_time;
} SimulationResult;

int main(int argc, char* argv[])
{
    double** u;     
    double** v;     
    double** p;     

    double** F;     
    double** G;     
    double** res;   
    double** RHS;   

    int i_max, j_max;                   
    double a, b;                        
    double Re;                          
    double delta_t, delta_x, delta_y;   
    double gamma;                       
    double T;                           
    double g_x;                         
    double g_y;                         
    double tau;                         
    double omega;                       
    double epsilon;                     
    int max_it;                         
    int n_print;                        
    int problem;                        
    double f;                           

    const char* param_file = "parameters.txt"; 

    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "CUDA: Could not open param_file\n");
        } else {
            param_file = argv[1];
            fclose(fp);
        }
    }
    
    int init_result = init(&problem, &f, &i_max, &j_max, &a, &b, &T, &Re, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);
    if (init_result != 0) {
        fprintf(stderr, "Failed to initialize parameters\n");
        return 1;
    }

    delta_x = a / i_max;
    delta_y = b / j_max;

    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);

    // Verificar alocações antes de inicializar CUDA
    if (p == NULL || u == NULL || v == NULL || res == NULL || RHS == NULL || F == NULL || G == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed before CUDA initialization\n");
        return 1;
    }
    
    // Inicialização dos arrays CUDA
    int cuda_result = initCudaArrays(p, u, v, res, RHS, i_max, j_max);
    if (cuda_result != 0) {
        fprintf(stderr, "ERROR: CUDA arrays initialization failed with code %d\n", cuda_result);
        free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
        return 1;
    }

    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    // Estrutura para armazenar resultados finais
    SimulationResult result = {0};

    clock_t start = clock();

    while (t < T) {
        // Remover print intermediário
        /*if (n % n_print == 0) {
            printf("%.5f / %.5f\n---------------------\n", t, T);
        }*/
        
        int sor_result = cudaSOR(p, u, v, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it, F, G, tau,
                                 Re, problem, f, &t, &n_out, g_x, g_y);
        
        // t já é atualizado dentro de cudaSOR
        n++;
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Obter valores finais
    double h_center_values[3];
    int center_i = i_max/2;
    int center_j = j_max/2;
    int center_idx = center_i * (j_max + 2) + center_j;
    
    // Alocar memória temporária para valores centrais
    double *d_center_values;
    cudaMalloc(&d_center_values, 3 * sizeof(double));
    
    // Extrair valores centrais
    extract_value_kernel<<<1, 1>>>(device_u, center_idx, &d_center_values[0]);
    extract_value_kernel<<<1, 1>>>(device_v, center_idx, &d_center_values[1]);
    extract_value_kernel<<<1, 1>>>(device_p, center_idx, &d_center_values[2]);
    cudaDeviceSynchronize();
    
    // Copiar para o host
    cudaMemcpy(h_center_values, d_center_values, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_center_values);
    
    // Armazenar resultados na estrutura
    result.u_center = h_center_values[0];
    result.v_center = h_center_values[1];
    result.p_center = h_center_values[2];
    result.final_time = t;
    result.iterations = n;
    result.execution_time = time_spent;
    
    // Print final dos resultados

    printf("U-CENTER: %.6f\n", result.u_center);
    printf("V-CENTER: %.6f\n", result.v_center);
    printf("P-CENTER: %.6f\n", result.p_center);

    
    // Continuar usando stderr para o tempo de execução (para compatibilidade)
    fprintf(stderr, "%.6f", time_spent);
    
    // Liberar memória CUDA
    freeCudaArrays();
    
    // Liberar memória CPU
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    return 0;
}