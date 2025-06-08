#!/bin/bash
#SBATCH --job-name=navierstokes_serial
#SBATCH --output=bench_serial_%j.out
#SBATCH --error=bench_serial_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=10-00:00:00

# First, create all necessary directories
echo "Creating result directories..."
mkdir -p results
mkdir -p results/times
mkdir -p results/serial

# Setup environment
echo "Setting up CUDA environment..."
module --ignore-cache load cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Store original directory
ORIG_DIR=$(pwd)

# Check if build exists and build if necessary
echo "Checking build status..."
if [ ! -d "build" ] || [ ! -f "build/serial" ]; then
    echo "Building NavierStokes with CMake..."
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_BUILD_TYPE=Release -DARCH=sm_60
    make -j > $ORIG_DIR/make_output.log 2>&1
    cd $ORIG_DIR
    echo "CMake build complete. Check make_output.log for errors"
else
    echo "Build already exists and serial executable found."
fi

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
  
  echo "  Running $runs executions..." >&2
  for ((r=1; r<=runs; r++)); do
    local temp_file="results/times/${prefix}_run${r}.time"
    local time=$(run_benchmark "$program" "$input" "$temp_file")
    
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

# ---- SERIAL BENCHMARK ----
echo "==== Serial NavierStokes Benchmark (3 runs each) ===="
echo -e "NavierStokes Serial Solver:"

# Create CSV for serial times
echo "test,serial_time,serial_std" > results/serial/serial_times.csv

# Number of runs for each test
BENCHMARK_RUNS=3

# Run NavierStokes serial tests
for i in {1..4}; do
  echo "Running serial test $i..."

  # Run tests with multiple executions and calculate average
  serial_result=$(run_multiple_benchmarks "./build/serial" "tests/$i.in" "$BENCHMARK_RUNS" "navierstokes_serial_test${i}")
  
  # Extract time and standard deviation from results
  serial_time=$(echo $serial_result | cut -d' ' -f1)
  serial_std=$(echo $serial_result | cut -d' ' -f2)
  
  echo "  Test $i: Serial time=${serial_time}s±${serial_std}s"
  echo "$i,$serial_time,$serial_std" >> results/serial/serial_times.csv
done

echo -e "\nSerial benchmarking complete. Results saved in results/serial/ directory."

# Generate a simple summary
echo -e "\n===== SERIAL BENCHMARK SUMMARY ====="
avg_time=$(awk -F, 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.4f", sum/count; else print "0.0000"}' results/serial/serial_times.csv)
echo "Average serial time across all tests: ${avg_time}s"
echo "Detailed results saved in: results/serial/serial_times.csv"

# Show individual test results
echo -e "\n===== INDIVIDUAL TEST RESULTS ====="
echo "Test | Serial Time (s) | Std Dev (s)"
echo "-----|-----------------|------------"
awk -F, 'NR>1 {printf "%4s | %13s   | %10s\n", $1, $2, $3}' results/serial/serial_times.csv