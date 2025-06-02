# CUDA Parallel Implementation

A parallel implementation using CUDA is provided in `src/parallel/main.cu`. This version accelerates the Navier-Stokes solver for incompressible flow using GPU parallelism, targeting significant speedup over the serial reference.

## Features
- Fully parallelized SOR (Successive Over-Relaxation) pressure solver using shared memory and halo regions for optimal memory access.
- Unified device array dimensions and robust memory management to avoid buffer overflows and memory corruption.
- Consistent 2D grid/block configuration for all CUDA kernels.
- Tolerance-based output validation to ensure numerical correctness against the serial reference.
- Detailed in-code documentation and comments for all major CUDA-specific changes and optimizations.

## Building the Parallel Version

The parallel version requires CUDA and a compatible NVIDIA GPU. To build:

1. Ensure you have the CUDA toolkit installed (e.g., `nvcc`).
2. Use the provided `CMakeLists.txt` to configure and build:
   ```bash
   mkdir -p build && cd build
   cmake ..
   make
   ```
   This will produce the parallel executable (e.g., `main_parallel` or similar, depending on CMake setup).

Alternatively, you can compile directly with nvcc:
   ```bash
   nvcc -O3 -o main_parallel src/parallel/main.cu
   ```

## Running the Parallel Version

- Prepare your `parameters.txt` as usual.
- Run the parallel executable:
  ```bash
  ./main_parallel
  ```
- Output files and formats are compatible with the serial version.

## Validation and Benchmarking

- Use the provided `colab-runner.ipynb` notebook to compare outputs from the serial and parallel versions. The notebook uses a robust, tolerance-based numerical comparator to ensure correctness.
- For benchmarking, adapt `get_time.sh` to run the parallel executable and compare timings with the serial version.
- Speedup and output correctness are reported in the notebook.

## Performance Notes

- The main performance gains come from parallelizing the SOR solver and minimizing global memory writes.
- Shared memory is used with padding to avoid bank conflicts.
- Further optimizations (kernel fusion, persistent kernels, 1D array refactor) are possible for advanced users.
- Profiling with Nsight or similar tools is recommended for identifying remaining bottlenecks.

## Documentation

- All major CUDA changes are documented in-code and summarized here.
- For further details, see comments in `src/parallel/main.cu` and the [docs](https://captainproton42.github.io/NavierStokes/).

---


## Original comment about the repository
Done as an assignment for university, the aim of this project is to implement a numeric solver for the Navier-Stokes equation for incompresible fluid flow using the method of finite differences. It is based on material handed out by the university which in return is mainly based on Griebel et al., 1998: *Numerical Simulation in Fluid Dynamics*. We implement the lid-driven cavity problem as a benchmark for our code and compare the results to those presented by Ghia et al., 1982: *High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method*. Lastly, we also experimented with rectangular box shapes and periodically oscillating boundary conditions.

# Building

The included makefile should work with gcc out of the box. Simply type `make` to start the build. For other compilers you need to compile by yourself.

# Parameters and Running

`i.in` stores all necessary information to run a simulation.

For more detailed descriptions of the code take a look at the [docs](https://captainproton42.github.io/NavierStokes/).

Our code used as inspiration the following article: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=a0367f904af5ceb5c15c99f4351fbb930081ada3

## Lid-Driven Cacvity
The original repository solved 2 problems, we only deal with the Lid-Driven Cavity on the CUDA ver.

# Results
You can see the benchmark and report of the parallel implementation in comparison with the serial one on the report file.

