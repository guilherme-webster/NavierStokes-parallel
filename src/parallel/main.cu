/**
 * @file main.c
 * @author Hollweck, Wigg
 * @date 10 April 2019
 * @brief Main file.
 *
 * Here typically goes a more extensive explanation of what the header
 * defines. Doxygens tags are words preceeded by either a backslash @\
 * or by an at symbol @@.
 * @see http://www.stack.nl/~dimitri/doxygen/docblocks.html
 * @see http://www.stack.nl/~dimitri/doxygen/commands.html
 */

#include "memory.h"
#include "io.h"
#include "integration.h"
#include "boundaries.h"

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Macro para verificação de erros CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Constantes do kernel - usar memória constante para acesso rápido
__constant__ double d_omega, d_dxdx, d_dydy;

/**
 * Kernel otimizado para atualização de células fantasma
 * Usa um único kernel para todas as bordas com melhor coalescência de memória
 */
__global__ void update_ghost_cells_kernel(double* d_p, int i_max, int j_max, int pitch) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Processamento horizontal e vertical em um único kernel
    if (idx < i_max + j_max + 2) {
        if (idx <= i_max) {
            // Atualizar bordas horizontais - threads 0 até i_max
            const int i = idx;
            // Evita condições de corrida e divergência
            if (i >= 1 && i <= i_max) {
                d_p[0 * pitch + i] = d_p[1 * pitch + i];                  // Bottom boundary
                d_p[(j_max + 1) * pitch + i] = d_p[j_max * pitch + i];    // Top boundary
            }
        }
        
        if (idx > i_max && idx <= i_max + j_max + 1) {
            // Atualizar bordas verticais - threads (i_max+1) até (i_max+j_max+1)
            const int j = idx - i_max - 1;
            if (j >= 1 && j <= j_max) {
                d_p[j * pitch + 0] = d_p[j * pitch + 1];                  // Left boundary
                d_p[j * pitch + (i_max + 1)] = d_p[j * pitch + i_max];    // Right boundary
            }
        }
    }
}

/**
 * SOR Red-Black otimizado usando shared memory para reduzir acessos globais
 */
__global__ void sor_red_black_kernel(double* d_p, double* d_RHS, int i_max, int j_max, 
                                    int pitch, int color) {
    // Tamanho fixo de memória compartilhada baseado no tamanho do bloco (16x16) + halo
    __shared__ double s_p[18][18];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = blockIdx.x * blockDim.x + tx + 1;
    const int j = blockIdx.y * blockDim.y + ty + 1;
    
    // Carregar dados para memória compartilhada incluindo halo (16x16 bloco + 1 de borda)
    // Cada thread carrega seu próprio valor + vizinhos necessários nas bordas
    if (i <= i_max + 1 && j <= j_max + 1) {
        s_p[ty+1][tx+1] = d_p[j * pitch + i];
        
        // Threads nas bordas carregam valores adicionais para o halo
        if (tx == 0 && i > 1) s_p[ty+1][0] = d_p[j * pitch + (i-1)];
        if (tx == blockDim.x-1 && i < i_max) s_p[ty+1][tx+2] = d_p[j * pitch + (i+1)];
        if (ty == 0 && j > 1) s_p[0][tx+1] = d_p[(j-1) * pitch + i];
        if (ty == blockDim.y-1 && j < j_max) s_p[ty+2][tx+1] = d_p[(j+1) * pitch + i];
    }
    __syncthreads();
    
    // Cálculo do SOR para a cor atual (red=0, black=1)
    if (i <= i_max && j <= j_max && ((i + j) % 2 == color)) {
        double update = (1.0 - d_omega) * s_p[ty+1][tx+1] + 
                      d_omega / (2.0 * (1.0 / d_dxdx + 1.0 / d_dydy)) * 
                      ((s_p[ty+1][tx+2] + s_p[ty+1][tx]) / d_dxdx + 
                       (s_p[ty+2][tx+1] + s_p[ty][tx+1]) / d_dydy - 
                       d_RHS[j * pitch + i]);
        
        d_p[j * pitch + i] = update;
    }
}

/**
 * Cálculo de resíduos otimizado usando memória compartilhada
 */
__global__ void calculate_residuals_kernel(double* d_p, double* d_RHS, double* d_res, 
                                         int i_max, int j_max, int pitch) {
    __shared__ double s_p[18][18];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = blockIdx.x * blockDim.x + tx + 1;
    const int j = blockIdx.y * blockDim.y + ty + 1;
    
    // Carregar dados para memória compartilhada
    if (i <= i_max + 1 && j <= j_max + 1) {
        s_p[ty+1][tx+1] = d_p[j * pitch + i];
        
        // Threads nas bordas carregam valores adicionais
        if (tx == 0 && i > 1) s_p[ty+1][0] = d_p[j * pitch + (i-1)];
        if (tx == blockDim.x-1 && i < i_max) s_p[ty+1][tx+2] = d_p[j * pitch + (i+1)];
        if (ty == 0 && j > 1) s_p[0][tx+1] = d_p[(j-1) * pitch + i];
        if (ty == blockDim.y-1 && j < j_max) s_p[ty+2][tx+1] = d_p[(j+1) * pitch + i];
    }
    __syncthreads();
    
    if (i <= i_max && j <= j_max) {
        d_res[j * pitch + i] = (s_p[ty+1][tx+2] - 2.0 * s_p[ty+1][tx+1] + s_p[ty+1][tx]) / d_dxdx + 
                              (s_p[ty+2][tx+1] - 2.0 * s_p[ty+1][tx+1] + s_p[ty][tx+1]) / d_dydy - 
                              d_RHS[j * pitch + i];
    }
}

/**
 * Kernel de redução L2-Norm otimizado usando warp shuffle
 * Redução em duas fases para melhor desempenho
 */
__global__ void l2_norm_kernel_optimized(double* d_res, double* d_norm, int i_max, int j_max, int pitch) {
    extern __shared__ double sdata[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = blockIdx.x * blockDim.x + tx + 1;
    const int j = blockIdx.y * blockDim.y + ty + 1;
    const int tid = ty * blockDim.x + tx;
    const int block_size = blockDim.x * blockDim.y;
    
    // Inicializar memória compartilhada
    sdata[tid] = 0.0;
    
    // Carregar e processar múltiplos elementos por thread para maior eficiência
    double thread_sum = 0.0;
    if (i <= i_max && j <= j_max) {
        double val = d_res[j * pitch + i];
        thread_sum = val * val;
    }
    
    // Armazenar soma parcial
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Redução em memória compartilhada - versão otimizada
    // Usar técnica de redução sem divergência condicional
    for (int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Redução final usando warp-level primitives (mais eficiente para últimos 32 threads)
    if (tid < 32) {
        // Volatile para evitar otimizações do compilador que quebrariam sincronização implícita
        volatile double* smem = sdata;
        if (block_size >= 64) smem[tid] += smem[tid + 32];
        if (block_size >= 32) smem[tid] += smem[tid + 16];
        if (block_size >= 16) smem[tid] += smem[tid + 8];
        if (block_size >= 8) smem[tid] += smem[tid + 4];
        if (block_size >= 4) smem[tid] += smem[tid + 2];
        if (block_size >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Apenas a thread 0 escreve o resultado para memória global
    if (tid == 0) {
        atomicAdd(d_norm, sdata[0]);
    }
}

/**
 * Implementação paralela otimizada do SOR usando CUDA
 */
int SOR_CUDA(double** p, int i_max, int j_max, double delta_x, double delta_y, 
             double** res, double** RHS, double omega, double eps, int max_it) {
    // Calcular constantes uma vez e enviá-las para a GPU
    double h_dxdx = delta_x * delta_x;
    double h_dydy = delta_y * delta_y;
    CUDA_CHECK(cudaMemcpyToSymbol(d_dxdx, &h_dxdx, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dydy, &h_dydy, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_omega, &omega, sizeof(double)));
    
    int it = 0;
    
    // Compute matrix dimensions
    size_t pitch;
    int width = i_max + 2;  // Include ghost cells
    int height = j_max + 2; // Include ghost cells
    
    // Usar memória pinned para transferências mais rápidas
    double **h_p_pinned, **h_res_pinned;
    CUDA_CHECK(cudaMallocHost((void**)&h_p_pinned, height * sizeof(double*)));
    CUDA_CHECK(cudaMallocHost((void**)&h_res_pinned, height * sizeof(double*)));
    
    for (int j = 0; j < height; j++) {
        CUDA_CHECK(cudaMallocHost((void**)&h_p_pinned[j], width * sizeof(double)));
        CUDA_CHECK(cudaMallocHost((void**)&h_res_pinned[j], width * sizeof(double)));
        
        // Copiar dados para memória pinned
        memcpy(h_p_pinned[j], p[j], width * sizeof(double));
        memcpy(h_res_pinned[j], res[j], width * sizeof(double));
    }
    
    // Alocar memória na GPU
    double *d_p, *d_res, *d_RHS, *d_norm;
    CUDA_CHECK(cudaMallocPitch(&d_p, &pitch, width * sizeof(double), height));
    CUDA_CHECK(cudaMallocPitch(&d_res, &pitch, width * sizeof(double), height));
    CUDA_CHECK(cudaMallocPitch(&d_RHS, &pitch, width * sizeof(double), height));
    CUDA_CHECK(cudaMalloc(&d_norm, sizeof(double)));
    
    // Converter pitch de bytes para elementos
    pitch /= sizeof(double);
    
    // Criar streams para operações assíncronas
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    // Transferir dados de forma assíncrona em chunks
    for (int j = 0; j < height; j++) {
        CUDA_CHECK(cudaMemcpyAsync(d_p + j * pitch, h_p_pinned[j], 
                                  width * sizeof(double), cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_res + j * pitch, h_res_pinned[j], 
                                  width * sizeof(double), cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_RHS + j * pitch, RHS[j], 
                                  width * sizeof(double), cudaMemcpyHostToDevice, stream2));
    }
    
    // Sincronizar streams antes de continuar
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    // Calcular norma L2 inicial
    double norm_p = L2(p, i_max, j_max);
    
    // Definir parâmetros de lançamento de kernel otimizados
    dim3 blockSize(16, 16);  // Potência de 2 para melhor desempenho na redução
    dim3 gridSize((i_max + blockSize.x - 1) / blockSize.x, 
                 (j_max + blockSize.y - 1) / blockSize.y);
    
    // Kernel de borda é 1D - otimizar para throughput
    dim3 ghostBlockSize(256);  // Máxima ocupação
    dim3 ghostGridSize(((i_max + j_max + 2) + ghostBlockSize.x - 1) / ghostBlockSize.x);
    
    // Variável temporária para norma
    double h_norm;
    
    // Loop de iteração SOR
    while (it < max_it) {
        // Atualizar células fantasma no stream1
        update_ghost_cells_kernel<<<ghostGridSize, ghostBlockSize, 0, stream1>>>(
            d_p, i_max, j_max, pitch);
        
        // Verificar erros
        CUDA_CHECK(cudaGetLastError());
        
        // Sincronizar antes de prosseguir
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        
        // Executar iteração SOR Red-Black (cores alternadas)
        // Vermelho (even)
        sor_red_black_kernel<<<gridSize, blockSize, 0, stream1>>>(
            d_p, d_RHS, i_max, j_max, pitch, 0);
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        
        // Preto (odd)
        sor_red_black_kernel<<<gridSize, blockSize, 0, stream1>>>(
            d_p, d_RHS, i_max, j_max, pitch, 1);
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        
        // Calcular resíduos no stream1
        calculate_residuals_kernel<<<gridSize, blockSize, 0, stream1>>>(
            d_p, d_RHS, d_res, i_max, j_max, pitch);
        
        // Reset do acumulador de norma
        CUDA_CHECK(cudaMemsetAsync(d_norm, 0, sizeof(double), stream1));
        
        // Calcular norma L2 dos resíduos - usando memória compartilhada otimizada
        l2_norm_kernel_optimized<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(double), stream1>>>(
            d_res, d_norm, i_max, j_max, pitch);
        
        // Copiar resultado da norma de volta para host
        CUDA_CHECK(cudaMemcpyAsync(&h_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        
        // Calcular norma final
        h_norm = sqrt(h_norm / (i_max * j_max));
        
        // Verificar convergência
        if (h_norm <= eps * (norm_p + 0.01)) {
            // Copiar resultados finais de volta para host de forma assíncrona
            for (int j = 0; j < height; j++) {
                CUDA_CHECK(cudaMemcpyAsync(h_p_pinned[j], d_p + j * pitch, 
                                         width * sizeof(double), cudaMemcpyDeviceToHost, stream1));
                CUDA_CHECK(cudaMemcpyAsync(h_res_pinned[j], d_res + j * pitch, 
                                         width * sizeof(double), cudaMemcpyDeviceToHost, stream2));
            }
            
            // Sincronizar e copiar de volta para os arrays originais
            CUDA_CHECK(cudaStreamSynchronize(stream1));
            CUDA_CHECK(cudaStreamSynchronize(stream2));
            
            for (int j = 0; j < height; j++) {
                memcpy(p[j], h_p_pinned[j], width * sizeof(double));
                memcpy(res[j], h_res_pinned[j], width * sizeof(double));
            }
            
            // Liberar recursos
            CUDA_CHECK(cudaFree(d_p));
            CUDA_CHECK(cudaFree(d_res));
            CUDA_CHECK(cudaFree(d_RHS));
            CUDA_CHECK(cudaFree(d_norm));
            
            for (int j = 0; j < height; j++) {
                CUDA_CHECK(cudaFreeHost(h_p_pinned[j]));
                CUDA_CHECK(cudaFreeHost(h_res_pinned[j]));
            }
            CUDA_CHECK(cudaFreeHost(h_p_pinned));
            CUDA_CHECK(cudaFreeHost(h_res_pinned));
            
            CUDA_CHECK(cudaStreamDestroy(stream1));
            CUDA_CHECK(cudaStreamDestroy(stream2));
            
            return 0;
        }
        
        it++;
    }
    
    // Copiar resultados finais após máximo de iterações
    for (int j = 0; j < height; j++) {
        CUDA_CHECK(cudaMemcpyAsync(h_p_pinned[j], d_p + j * pitch, 
                                 width * sizeof(double), cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaMemcpyAsync(h_res_pinned[j], d_res + j * pitch, 
                                 width * sizeof(double), cudaMemcpyDeviceToHost, stream2));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    
    for (int j = 0; j < height; j++) {
        memcpy(p[j], h_p_pinned[j], width * sizeof(double));
        memcpy(res[j], h_res_pinned[j], width * sizeof(double));
    }
    
    // Liberar recursos
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_RHS));
    CUDA_CHECK(cudaFree(d_norm));
    
    for (int j = 0; j < height; j++) {
        CUDA_CHECK(cudaFreeHost(h_p_pinned[j]));
        CUDA_CHECK(cudaFreeHost(h_res_pinned[j]));
    }
    CUDA_CHECK(cudaFreeHost(h_p_pinned));
    CUDA_CHECK(cudaFreeHost(h_res_pinned));
    
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    
    return -1;
}

/**
* @brief Main function.
* 
* This is the main function.
* @return 0 on exit.
*/

int main(int argc, char* argv[])
{
    // Grid pointers.
	double** u;     // velocity x-component
	double** v;     // velocity y-component
	double** p;     // pressure

    double** F;     // F term
    double** G;     // G term
    double** res;   // SOR residuum
    double** RHS;   // RHS of poisson equation

    // Simulation parameters.
    int i_max, j_max;                   // number of grid points in each direction
    double a, b;                        // sizes of the grid
    double Re;                          // reynolds number
    double delta_t, delta_x, delta_y;   // step sizes
    double gamma;                       // weight for Donor-Cell-stencil
    double T;                           // max time for integration
    double g_x;                         // x-component of g
    double g_y;                         // y-component of g
    double tau;                         // security factor for adaptive step size
    double omega;                       // relaxation parameter
    double epsilon;                     // relative tolerance for SOR
    int max_it;                         // maximum iterations for SOR
    int n_print;                        // output to file every ..th step
    int problem;                        // problem type
    double f;                           // frequency of periodic boundary conditions (if problem == 2)

    const char* param_file = "parameters.txt"; // file containing parameters

    // fprintf(stderr, "CUDA: Working directory test\n");
    
    // Test if we can open the file directly
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            fprintf(stderr, "CUDA: Could not open param_file\n");
        } else {
            // fprintf(stderr, "CUDA: Successfully opened '%s'\n", argv[1]);
            param_file = argv[1];
            fclose(fp);
        }
    }
    
    // if (argc > 1) {
    //     param_file = argv[1];
    // }
    
    // Initialize all parameters.
	init(&problem, &f, &i_max, &j_max, &a, &b, &Re, &T, &g_x, &g_y, &tau, &omega, &epsilon, &max_it, &n_print, param_file);

    // Set step size in space.
    delta_x = a / i_max;
    delta_y = b / j_max;

    // Allocate memory for grids.
    allocate_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max, j_max);

    // Time loop.
    double t = 0;
    int i, j;
    int n = 0;
    int n_out = 0;

    clock_t start = clock();

    while (t < T) {

    	// Adaptive stepsize and weight factor for Donor-Cell
        double u_max = max_mat(i_max, j_max, u);
        double v_max = max_mat(i_max, j_max, v);
    	delta_t = tau * n_min(3, Re / 2.0 / ( 1.0 / delta_x / delta_x + 1.0 / delta_y / delta_y ), delta_x / fabs(u_max), delta_y / fabs(v_max));
        gamma = fmax(u_max * delta_t / delta_x, v_max * delta_t / delta_y);

        // Set boundary conditions.
        if (problem == 1) {
            set_noslip(i_max, j_max, u, v, LEFT);
            set_noslip(i_max, j_max, u, v, RIGHT);
            set_noslip(i_max, j_max, u, v, BOTTOM);
            set_inflow(i_max, j_max, u, v, TOP, 1.0, 0.0);
        } else if (problem == 2) {
            set_noslip(i_max, j_max, u, v, LEFT);
            set_noslip(i_max, j_max, u, v, RIGHT);
            set_noslip(i_max, j_max, u, v, BOTTOM);
            set_inflow(i_max, j_max, u, v, TOP, sin(f*t), 0.0);           
        } else {
            printf("Unknown probem type (see parameters.txt).\n");
            exit(EXIT_FAILURE);
        }


        // Calculate F and G.
        FG(F, G, u, v, i_max, j_max, Re, g_x, g_y, delta_t, delta_x, delta_y, gamma);


        // RHS of Poisson equation.
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                RHS[i][j] = 1.0 / delta_t * ((F[i][j] - F[i-1][j])/delta_x + (G[i][j] - G[i][j-1])/delta_y);
            }
        }

        // Execute SOR step.
        int sor_result = SOR_CUDA(p, i_max, j_max, delta_x, delta_y, res, RHS, omega, epsilon, max_it);

        // Update velocities.
        for (i = 1; i <= i_max; i++ ) {
            for (j = 1; j <= j_max; j++) {
                if (i <= i_max - 1) u[i][j] = F[i][j] - delta_t * dp_dx(p, i, j, delta_x);
                if (j <= j_max - 1) v[i][j] = G[i][j] - delta_t * dp_dy(p, i, j, delta_y);
            }
        }

        // Print to file every ..th step.
        // if (n % n_print == 0) {
        //     char out_prefix[12];
        //     sprintf(out_prefix, "out/%d", n_out);
        //     output(i_max, j_max, u, v, p, t, a, b, out_prefix);
        //     n_out++;
        // }

        if (n % n_print == 0) {
            // Instead of outputting to files, print the data to stdout
            printf("TIMESTEP: %d TIME: %.6f\n", n_out, t);

            // Print some key values from u, v, p matrices
            // For example, print central values and some boundary values
            printf("U-CENTER: %.6f\n", u[i_max/2][j_max/2]);
            printf("V-CENTER: %.6f\n", v[i_max/2][j_max/2]);
            // Add more key values as needed
            n_out++;
        }

        t += delta_t;
        n++;
    }

    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    fprintf(stderr, "%.6f", time_spent);

    // Free grid memory.
    free_memory(&u, &v, &p, &res, &RHS, &F, &G, i_max);
    return 0;
}