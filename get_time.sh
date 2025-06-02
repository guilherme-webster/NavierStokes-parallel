#!/bin/bash
#SBATCH --job-name=cuda_bench
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --nodelist=sorgan-gpu[1-4]

# First, create all necessary directories
echo "Creating result directories..."
mkdir -p results
mkdir -p results/times

# Setup environment
echo "Setting up CUDA environment..."
module --ignore-cache load cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Store original directory
ORIG_DIR=$(pwd)

# Create and prepare the smoothing build directory
echo "Building smoothing with CMake..."
mkdir -p 03-Smoothing/build
cd 03-Smoothing/build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DARCH=sm_60
make -j > $ORIG_DIR/make_output.log 2>&1

# Return to original directory
cd $ORIG_DIR

# After running, check the logs
echo "CMake build complete. Check make_output.log for errors"

# Function to run benchmark and extract kernel time
run_benchmark() {
  local program=$1
  local input=$2
  local output_file=$3
  local thread_arg=$4
  
  # Check if files exist before running
  if [ ! -f "$program" ]; then
    echo "ERROR: Program file not found: $program"
    return 1
  fi
  
  if [ ! -f "$input" ]; then
    echo "ERROR: Input file not found: $input"
    return 1
  fi
  
  # Create directory for output if it doesn't exist
  mkdir -p $(dirname "$output_file")
  
  # Run the program and capture stderr (which has the timing)
  if [[ -n "$thread_arg" ]]; then
    $program $input $thread_arg 1>/dev/null 2> "$output_file"
  else
    $program $input 1>/dev/null 2> "$output_file"
  fi
  
  # Extract time from stderr (assuming it's the last line)
  local time=$(tail -n 1 "$output_file")
  echo $time
}

# Add this function to your script
calculate_efficiency() {
  local speedup=$1
  local threads_per_block=$2
  
  # P100 architecture constants
  local max_threads_per_sm=2048
  local max_blocks_per_sm=32
  local max_warps_per_sm=64  # 64 warps of 32 threads each
  
  # Use awk for all calculations
  local occupancy=$(awk -v tpb="$threads_per_block" \
                      -v max_t="$max_threads_per_sm" \
                      -v max_b="$max_blocks_per_sm" \
                      -v max_w="$max_warps_per_sm" \
                     'BEGIN {
                        warps_per_block = int((tpb + 31) / 32);
                        blocks_per_sm_by_threads = int(max_t / tpb);
                        blocks_per_sm = blocks_per_sm_by_threads < max_b ? blocks_per_sm_by_threads : max_b;
                        warps_per_sm = blocks_per_sm * warps_per_block;
                        occupancy = warps_per_sm / max_w;
                        printf "%.4f", occupancy;
                      }')
  
  # Calculate efficiency (speedup / occupancy)
  local efficiency=$(awk -v speedup="$speedup" -v occupancy="$occupancy" \
                      'BEGIN { printf "%.4f", speedup / occupancy }')
  
  echo "$efficiency"
}

# ---- SERIAL VS PARALLEL COMPARISON ----
echo "==== Serial vs Parallel Comparison ===="
echo -e "Smoothing:"

# Create CSV for smoothing
echo "test,serial_time,parallel_time,speedup" > results/smoothing_speedup.csv

# Run smoothing tests
for i in {1..5}; do
  echo "Running smoothing test $i..."

  # create symbolic links to the test files   
  ln -sf 03-Smoothing/tests tests

  # Update paths in your run_benchmark calls
  serial_time=$(run_benchmark "./03-Smoothing/build/serial" "03-Smoothing/tests/$i.in" "results/times/smoothing_serial_$i.time")
  parallel_time=$(run_benchmark "./03-Smoothing/build/parallel" "03-Smoothing/tests/$i.in" "results/times/smoothing_parallel_$i.time")
  
  # Calculate speedup
  speedup=$(awk "BEGIN {printf \"%.4f\", $serial_time / $parallel_time}")
  
  echo "Test $i: Serial=$serial_time s, Parallel=$parallel_time s, Speedup=${speedup}x"
  echo "$i,$serial_time,$parallel_time,$speedup" >> results/smoothing_speedup.csv
done

# ---- THREAD CONFIGURATION SCALING ----
echo -e "\n==== Thread Configuration Scaling ===="
echo -e "smoothing (Test 5):"
# Create CSV for smoothing thread scaling
echo "threads,time,speedup_vs_64,efficiency" > results/smoothing_thread_scaling.csv

# Run smoothing with different thread counts
base_time=""
for threads in 128 256 512 1024; do
  thread_time=$(run_benchmark "./03-Smoothing/build/parallel" "03-Smoothing/tests/5.in" "results/times/smoothing_t${threads}.time" "-t $threads")
  
  if [[ -z "$base_time" ]]; then
    base_time=$thread_time
    speedup="1.0000"
    efficiency="1.0000"  # Baseline efficiency
  else
    speedup=$(awk "BEGIN {printf \"%.4f\", $base_time / $thread_time}")
    efficiency=$(calculate_efficiency "$speedup" "$threads")
  fi
  
  echo "$threads threads: $thread_time s (${speedup}x vs 64 threads)"
  echo "$threads,$thread_time,$speedup,$efficiency" >> results/smoothing_thread_scaling.csv
done

echo -e "\nBenchmarking complete. Results saved in results/ directory."

# Generate a simple summary
echo -e "\n===== BENCHMARK SUMMARY ====="
echo "Average smoothing Speedup: $(awk -F, 'NR>1 {sum+=$4} END {print sum/(NR-1)}' results/smoothing_speedup.csv)"
echo "Best smoothing Thread Count: $(sort -t, -k2,2n results/smoothing_thread_scaling.csv | head -2 | tail -1 | cut -d, -f1)"