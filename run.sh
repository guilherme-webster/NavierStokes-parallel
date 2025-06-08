#!/bin/bash
#SBATCH --job-name=navierstokes_bench
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=10-00:00:00
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
  local output_file=$3
  
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
  $program $input 1>/dev/null 2> "$output_file"
  
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
  local runs=$3
  local prefix=$4
  
  local times=()
  local total_time=0
  local valid_runs=0
  
  echo "  Running $runs executions..." >&2 # Redireciona para stderr
  for ((r=1; r<=runs; r++)); do
    local temp_file="results/times/${prefix}_run${r}.time"
    # Note: Removed thread_arg from this call, ensure run_benchmark definition is also updated
    local time=$(run_benchmark "$program" "$input" "$temp_file")
    
    if [[ -n "$time" && "$time" != "0" ]]; then
      times+=("$time")
      total_time=$(awk "BEGIN {print $total_time + $time}")
      valid_runs=$((valid_runs + 1))
      echo "    Run $r: ${time}s" >&2 # Redireciona para stderr
    else
      echo "    Run $r: FAILED" >&2 # Redireciona para stderr
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
    
    echo "    Average: ${avg_time}s ± ${std_dev}s (${valid_runs}/${runs} valid runs)" >&2 # Redireciona para stderr
    echo "$avg_time $std_dev" # Esta linha vai para stdout
  else
    echo "    Average: FAILED (0/${runs} valid runs)" >&2 # Redireciona para stderr
    echo "0 0" # Esta linha vai para stdout
  fi
}

# ---- SERIAL VS PARALLEL COMPARISON ----
echo "==== Serial vs Parallel Comparison (3 runs each) ===="
echo -e "NavierStokes Solver:"

# Create CSV for NavierStokes
echo "test,serial_time,serial_std,parallel_time,parallel_std,speedup" > results/navierstokes_speedup.csv

# Number of runs for each test
BENCHMARK_RUNS=3

# Run NavierStokes tests
for i in {1..4}; do
  echo "Running NavierStokes test $i..."

  # Run tests with multiple executions and calculate average
  serial_result=$(run_multiple_benchmarks "./build/serial" "tests/$i.in" "$BENCHMARK_RUNS" "navierstokes_serial_$i")
  parallel_result=$(run_multiple_benchmarks "./build/parallel" "tests/$i.in" "$BENCHMARK_RUNS" "navierstokes_parallel_$i")
  
  # Extract time and standard deviation from results
  serial_time=$(echo $serial_result | cut -d' ' -f1)
  serial_std=$(echo $serial_result | cut -d' ' -f2)
  parallel_time=$(echo $parallel_result | cut -d' ' -f1)
  parallel_std=$(echo $parallel_result | cut -d' ' -f2)
  
  # Calculate speedup
  if [[ "$parallel_time" != "0" && -n "$parallel_time" ]]; then
    speedup=$(awk "BEGIN {printf \"%.4f\", $serial_time / $parallel_time}")
  else
    speedup="0.0000"
  fi
  
  echo "Test $i: Serial=${serial_time}s±${serial_std}s, Parallel=${parallel_time}s±${parallel_std}s, Speedup=${speedup}x"
  echo "$i,$serial_time,$serial_std,$parallel_time,$parallel_std,$speedup" >> results/navierstokes_speedup.csv
done

echo -e "\nBenchmarking complete. Results saved in results/ directory."

# Generate a simple summary
echo -e "\n===== BENCHMARK SUMMARY ====="
echo "Average NavierStokes Speedup: $(awk -F, 'NR>1 {sum+=$6; count++} END {if(count>0) printf "%.4f", sum/count; else print "0.0000"}' results/navierstokes_speedup.csv)"
echo "Results saved in: results/navierstokes_speedup.csv"