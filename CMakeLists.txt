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
    src/parallel/boundaries.cu
    src/parallel/integration.cu
    src/parallel/memory.cu
)
set(Openmp_SOURCES
    src/openmp/main.c
    src/openmp/io.c
    src/openmp/boundaries.c
    src/openmp/integration.c
    src/openmp/memory.c
)

# Define o executável
add_executable(serial   ${SOURCES})
add_executable(parallel ${CUDA_SOURCES})
add_executable(openmp    ${Openmp_SOURCES})

# Use a arquitetura apenas se for especificada pelo usuário
if(DEFINED ARCH)
    target_compile_options(parallel PRIVATE "-arch=${ARCH}")
endif()

# Vincula as bibliotecas necessárias
target_link_libraries(serial m OpenMP::OpenMP_C)
target_link_libraries(parallel m OpenMP::OpenMP_C)
target_link_libraries(openmp m OpenMP::OpenMP_C)

# Configurar flags específicas para OpenMP com offloading para GPU
if(COMPILER_SUPPORTS_OPENMP_GPU)
    target_compile_options(openmp PRIVATE 
        $<$<COMPILE_LANGUAGE:C>:-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda>)
    target_link_options(openmp PRIVATE -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda)
    message(STATUS "Configurando OpenMP com suporte a offloading para GPU")
else()
    message(WARNING "O compilador não suporta OpenMP offloading para GPU. O código OpenMP será executado apenas na CPU.")
endif()


set_property(
    TARGET   parallel
    PROPERTY CUDA_ARCHITECTURES OFF)