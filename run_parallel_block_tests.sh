#!/bin/bash
#SBATCH --job-name=navierstokes_parallel_blocks
#SBATCH --output=bench_parallel_blocks_%j.out
#SBATCH --error=bench_parallel_blocks_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=10-00:00:00

# First, create all necessary directories
echo "Creating result directories..."
mkdir -p results
mkdir -p results/times
mkdir -p results/parallel_blocks

# Setup environment
echo "Setting up CUDA environment..."
module --ignore-cache load cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Store original directory
ORIG_DIR=$(pwd)

# Create and prepare the build directory
echo "Building NavierStokes with CMake..."
mkdir -p build
cd build
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
  local block_size=$3
  local output_file=$4
  
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
  
  # Run the program with block size parameter and capture stderr (which has the timing)
  $program $input $block_size 1>/dev/null 2> "$output_file"
  
  # Extract time from stderr (last line with "Total execution time" or a simple time value)
  local time=$(grep -E "(Total execution time|time:)" "$output_file" | tail -n 1 | grep -oE '[0-9]+\.[0-9]+' | tail -n 1)
  if [[ -z "$time" ]]; then
    # If no specific pattern found, try to get the last floating point number
    time=$(tail -n 1 "$output_file" | grep -oE '[0-9]+\.[0-9]+' | tail -n 1)
  fi
  echo $time
}

# Function to run multiple benchmarks and calculate average and standard deviation
run_multiple_benchmarks() {
  local program=$1
  local input=$2
  local block_size=$3
  local runs=$4
  local prefix=$5
  
  local times=()
  local total_time=0
  local valid_runs=0
  
  echo "  Running $runs executions with block size $block_size..." >&2
  for ((r=1; r<=runs; r++)); do
    local temp_file="results/times/${prefix}_block${block_size}_run${r}.time"
    local time=$(run_benchmark "$program" "$input" "$block_size" "$temp_file")
    
    if [[ -n "$time" && "$time" != "0" ]]; then
      times+=("$time")
      total_time=$(awk "BEGIN {print $total_time + $time}")
      valid_runs=$((valid_runs + 1))
      echo "    Run $r: ${time}s" >&2
    else
      echo "    Run $r: FAILED" >&2
    fi
  done
  
  if [[ $valid_runs -gt 0 ]]; then
    local avg_time=$(awk "BEGIN {printf \"%.4f\", $total_time / $valid_runs}")
    
    # Calculate standard deviation
    local sum_sq_diff=0
    for time in "${times[@]}"; do
      local diff=$(awk "BEGIN {print $time - $avg_time}")
      sum_sq_diff=$(awk "BEGIN {print $sum_sq_diff + ($diff * $diff)}")
    done
    
    local std_dev
    if [[ $valid_runs -gt 1 ]]; then
      std_dev=$(awk "BEGIN {printf \"%.4f\", sqrt($sum_sq_diff / ($valid_runs - 1))}")
    else
      std_dev="0.0000"
    fi
    
    echo "    Average: ${avg_time}s ± ${std_dev}s (${valid_runs}/${runs} valid runs)" >&2
    echo "$avg_time $std_dev" # Esta linha vai para stdout
  else
    echo "    Average: FAILED (0/${runs} valid runs)" >&2
    echo "0 0" # Esta linha vai para stdout
  fi
}

# ---- PARALLEL BLOCK SIZE COMPARISON ----
echo "==== Parallel Block Size Comparison (3 runs each) ===="
echo -e "NavierStokes Parallel Solver with different block sizes:"

# Define block sizes to test
BLOCK_SIZES=(8 16 32)

# Create CSV for block size comparison
echo "test,block_size,avg_time,std_dev" > results/parallel_blocks/block_size_comparison.csv

# Number of runs for each test
BENCHMARK_RUNS=3

# Run NavierStokes parallel tests with different block sizes
for block_size in "${BLOCK_SIZES[@]}"; do
  echo "Testing block size: $block_size"
  
  for i in {1..4}; do
    echo "  Running test $i with block size $block_size..."

    # Run tests with multiple executions and calculate average
    parallel_result=$(run_multiple_benchmarks "./build/parallel" "tests/$i.in" "$block_size" "$BENCHMARK_RUNS" "navierstokes_parallel_test${i}")
    
    # Extract time and standard deviation from results
    parallel_time=$(echo $parallel_result | cut -d' ' -f1)
    parallel_std=$(echo $parallel_result | cut -d' ' -f2)
    
    echo "  Test $i, Block size $block_size: Time=${parallel_time}s±${parallel_std}s"
    echo "$i,$block_size,$parallel_time,$parallel_std" >> results/parallel_blocks/block_size_comparison.csv
  done
  echo ""
done

echo -e "\nBenchmarking complete. Results saved in results/parallel_blocks/ directory."

# Generate a simple summary for each block size
echo -e "\n===== BLOCK SIZE BENCHMARK SUMMARY ====="
for block_size in "${BLOCK_SIZES[@]}"; do
  avg_time=$(awk -F, -v bs="$block_size" 'NR>1 && $2==bs {sum+=$3; count++} END {if(count>0) printf "%.4f", sum/count; else print "0.0000"}' results/parallel_blocks/block_size_comparison.csv)
  echo "Block size $block_size average time: ${avg_time}s"
done

echo "Detailed results saved in: results/parallel_blocks/block_size_comparison.csv"
