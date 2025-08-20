# Benchmark Comparison Workflow

This document explains how to compare performance between the main branch and a PR branch with optimizations.

## Scripts Overview

### Core Scripts
1. **`benchmark.py`** - Runs comprehensive performance benchmarks across multiple dataset sizes and saves results to JSON
2. **`benchmark_profile.py`** - Runs profiling for a single configuration with detailed memory tracking and timing breakdown
3. **`benchmark_compare.py`** - Compares results from two benchmark runs

### Supporting Files
4. **`benchmark_setup.py`** - Shared configuration (TT_TARGETS, MAPPER, utilities) used by both main scripts
5. **`benchmark_make_data.py`** - Synthetic data generation for standardized testing
   - `make_data(N, scramble_data=False)` - Generate N households with optional data scrambling
   - By default, data is kept in sorted p_id order for optimal performance
   - Set `scramble_data=True` to test performance with unsorted data
6. **`benchmark_compare.py`** - Stage-by-stage comparison tool

## Key Features

### 3-Stage Timing Analysis
All scripts break down execution into:
- **Stage 1**: Data preprocessing & DAG creation
- **Stage 2**: Core computation (tax/transfer calculations)  
- **Stage 3**: DataFrame formatting (JAX → pandas conversion)

### Memory Tracking
- Both `benchmark.py` and `benchmark_profile.py` now include comprehensive memory tracking
- Continuous monitoring of peak memory usage during execution
- Memory delta reporting (initial → final)

## Workflow

### Step 1: Run benchmark on main branch

```bash
# Switch to main branch (ttsim)
git checkout main

# Run comprehensive benchmark across all dataset sizes (default: sorted data)
python benchmark.py

# Optional: Run with scrambled data to test worst-case performance
python benchmark.py -scramble

# This creates a file like: benchmark_results_20250819_143022_sorted.json
# or: benchmark_results_20250819_143022_scrambled.json
```

### Step 2: Run benchmark on PR branch 

```bash
# Switch to PR branch (ttsim)
git checkout JW/dev/speedup-JAX

# Run benchmark (default: sorted data)
python benchmark.py

# Optional: Run with scrambled data to test worst-case performance
python benchmark.py -scramble

# This creates another file like: benchmark_results_20250819_145133_sorted.json
# or: benchmark_results_20250819_145133_scrambled.json
```

### Step 3: Compare results

```bash
# Compare the two result files (first file=main branch, second file=PR branch)
python benchmark_compare.py benchmark_results_20250819_143022.json benchmark_results_20250819_145133.json

# Optional: Save comparison to file
python benchmark_compare.py benchmark_results_20250819_143022.json benchmark_results_20250819_145133.json --save-comparison
```

### Alternative: Profiling Single Configuration

For detailed analysis of a specific configuration:

```bash
# Run profiling with memory tracking (default: sorted data)
python benchmark_profile.py -N 32768 -b numpy

# Run with scrambled data to test worst-case performance
python benchmark_profile.py -N 32768 -b numpy -scramble

# Or with JAX backend
python benchmark_profile.py -N 65536 -b jax
python benchmark_profile.py -N 65536 -b jax -scramble

# For external profiling (e.g., with py-spy)
py-spy record -o profile.svg -- python benchmark_profile.py -N 32768 -b numpy
py-spy record -o profile_scrambled.svg -- python benchmark_profile.py -N 32768 -b numpy -scramble
```

## Interpreting Results

- **Speedup > 1.0**: PR branch is faster than main branch
- **Identical Hashes**: Optimizations maintain numerical accuracy
- **Hash Mismatches**: Potential numerical differences (investigate!)
- **Memory Δ**: Memory increase during execution (should be reasonable)
- **Peak Memory**: Maximum memory usage (important for large datasets)

## Dataset Sizes Tested

- 32,767 households (2^15 - 1)
- 32,768 households (2^15)
- 65,536 households (2^16)
- 131,072 households (2^17)
- 262,144 households (2^18)
- 524,288 households (2^19)
- 1,048,576 households (2^20)

## Data Generation Options

The `benchmark_make_data.py` module provides the `make_data()` function with the following options:

```python
# Generate sorted data (default - optimal performance)
data = make_data(N=32768, scramble_data=False)

# Generate scrambled data (tests performance with unsorted p_id order)
data = make_data(N=32768, scramble_data=True)
```
