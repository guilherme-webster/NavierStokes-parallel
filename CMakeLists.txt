cmake_minimum_required(VERSION 3.10)

# Define o nome do projeto
project(NavierStokes LANGUAGES C CUDA) #CUDA CXX) 

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
    src/parallel/memory.cu
    src/parallel/kernel.cu
)

# Define o executável
add_executable(serial   ${SOURCES})
add_executable(parallel ${CUDA_SOURCES})

# Use a arquitetura apenas se for especificada pelo usuário
if(DEFINED ARCH)
    target_compile_options(parallel PRIVATE "-arch=${ARCH}")
endif()

# Vincula as bibliotecas necessárias
target_link_libraries(serial m OpenMP::OpenMP_C)
target_link_libraries(parallel m OpenMP::OpenMP_C)

set_property(
    TARGET   parallel
    PROPERTY CUDA_ARCHITECTURES OFF)