cmake_minimum_required(VERSION 3.10)

# Define o nome do projeto
project(NavierStokes LANGUAGES C CUDA)

# Importa os pacotes
find_package(OpenMP REQUIRED)

# Adiciona os diretórios de inclusão
include_directories(.)
include_directories(src)

# Adiciona os arquivos fonte
set(SOURCES
    src/serial/main.c
    src/serial/io.c
    src/serial/boundaries.c
    src/serial/integration.c
    src/serial/memory.c
)
set(CUDA_SOURCES
    src/parallel/main.cu
    src/parallel/io.cu
    src/parallel/boundaries.cu
    src/parallel/integration.cu
    src/parallel/memory.cu
    # Missing files that contain required functions:
    src/parallel/utils.cu
    src/parallel/cuda_kernels.cu
)

# Define o executável
add_executable(serial   ${SOURCES})
add_executable(parallel ${CUDA_SOURCES})

# Use -DARCH=sm_XX to override architecture, default is sm_75
if(NOT DEFINED ARCH)
    set(ARCH "sm_75")
endif()

# Vincula as bibliotecas necessárias
target_link_libraries(serial m OpenMP::OpenMP_C)
target_link_libraries(parallel m OpenMP::OpenMP_C)

set_property(
    TARGET   parallel
    PROPERTY CUDA_ARCHITECTURES OFF)