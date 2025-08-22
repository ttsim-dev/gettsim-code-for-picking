"""Performance comparison script for numpy vs jax backends."""
import json
import hashlib
import time
import argparse
from datetime import datetime

from gettsim import main, InputData, MainTarget, TTTargets, Labels, SpecializedEnvironment, RawResults

# Import shared benchmark configuration and utilities
from benchmark_setup import (
    TT_TARGETS, MAPPER, JAX_AVAILABLE,
    sync_jax_if_needed, clear_jax_cache, get_memory_usage_mb, MemoryTracker,
    force_garbage_collection, reset_session_state, BENCHMARK_HOUSEHOLD_SIZES, BACKENDS
)
from benchmark_make_data import make_data


def run_benchmark(
        N_households, backend,
        reset_session=False,
        sync_jax=False,
        scramble_data=False,
    ):
    """Run a single benchmark with 3-stage timing as in gettsim_profile_stages.py."""
    print(f"Running benchmark: {N_households:,} households, {backend} backend")
    
    # Reset session state to ensure clean environment
    if reset_session:
        reset_session_state(backend)
    
    # Generate data
    print("  Generating data...")
    data = make_data(N_households, scramble_data=scramble_data)
    
    # Memory tracking setup - always track peak memory for benchmarking
    tracker = MemoryTracker()
    
    # Initial memory reading
    initial_memory = get_memory_usage_mb()
    tracker.start_monitoring()
    
    try:
        # STAGE 1: Data preprocessing and DAG creation
        print("  Stage 1: Data preprocessing and DAG creation...")
        stage1_start = time.time()

        tmp = main(
            policy_date_str="2025-01-01",
            input_data=InputData.df_and_mapper(
                df=data,
                mapper=MAPPER,
            ),
            main_targets=[
                MainTarget.specialized_environment.tt_dag,
                MainTarget.processed_data,
                MainTarget.labels.root_nodes,
                MainTarget.input_data.flat,  # Need this for stage 3
                MainTarget.tt_function,
            ],
            tt_targets=TTTargets(
                tree=TT_TARGETS,
            ),
            include_fail_nodes=True,
            include_warn_nodes=False,
            backend=backend,
        )    

        # Force JAX synchronization before recording end time
        if sync_jax:
            sync_jax_if_needed(backend)

        stage1_end = time.time()
        stage1_time = stage1_end - stage1_start

        # Generate hash for Stage 1 output (tmp)
        stage1_hash = hashlib.md5(str(tmp).encode('utf-8')).hexdigest()

        # STAGE 2: Computation only (no data preprocessing)
        print("  Stage 2: Computation only...")
        
        stage2_start = time.time()

        raw_results_stage2 = main(
            policy_date_str="2025-01-01",
            main_targets=[
                MainTarget.raw_results.columns,
                MainTarget.raw_results.params,
                MainTarget.raw_results.from_input_data,
            ],
            tt_targets=TTTargets(
                tree=TT_TARGETS,
            ),
            processed_data=tmp["processed_data"],
            input_data=InputData.flat(tmp["input_data"]["flat"]),  # Provide the flat input data from stage 1
            labels=Labels(root_nodes=tmp["labels"]["root_nodes"]),
            tt_function=tmp["tt_function"],  # Reuse pre-compiled JAX function
            include_fail_nodes=False,
            include_warn_nodes=False,
            backend=backend,
        )

        # Force JAX synchronization before recording end time
        if sync_jax:
            sync_jax_if_needed(backend)

        stage2_end = time.time()
        stage2_time = stage2_end - stage2_start

        # Generate hash for Stage 2 output (raw_results_stage2)
        stage2_hash = hashlib.md5(str(raw_results_stage2).encode('utf-8')).hexdigest()

        # STAGE 3: Convert raw results to DataFrame (no computation, just formatting)
        print("  Stage 3: Convert raw results to DataFrame...")
        stage3_start = time.time()

        result = main(
            policy_date_str="2025-01-01",
            main_target=MainTarget.results.df_with_mapper,
            tt_targets=TTTargets(
                tree=TT_TARGETS,
            ),
            raw_results=raw_results_stage2["raw_results"],
            input_data=InputData.flat(tmp["input_data"]["flat"]),  # Provide the flat input data from stage 1
            processed_data=tmp["processed_data"],
            labels=Labels(root_nodes=tmp["labels"]["root_nodes"]),
            specialized_environment=SpecializedEnvironment(
                tt_dag=tmp["specialized_environment"]["tt_dag"]
            ),
            include_fail_nodes=False,
            include_warn_nodes=False,
            backend=backend,
        )

        # Force JAX synchronization before recording end time
        if sync_jax:
            sync_jax_if_needed(backend)

        stage3_end = time.time()
        stage3_time = stage3_end - stage3_start
        total_time = stage1_time + stage2_time + stage3_time
        
        # Generate hash for Stage 3 output (result)
        stage3_hash = hashlib.md5(str(result).encode('utf-8')).hexdigest()
        
        # Final memory reading
        final_memory = get_memory_usage_mb()
        tracker.stop_monitoring()
        peak_memory = tracker.get_peak()

        # Calculate memory delta
        memory_delta = final_memory - initial_memory
        
        # Results shape
        result_shape = result.shape if hasattr(result, 'shape') else 'N/A'
        
        print(f"  Stage 1: {stage1_time:.4f}s ({stage1_time/total_time*100:.1f}%)")
        print(f"  Stage 2: {stage2_time:.4f}s ({stage2_time/total_time*100:.1f}%)")
        print(f"  Stage 3: {stage3_time:.4f}s ({stage3_time/total_time*100:.1f}%)")
        print(f"  Total: {total_time:.4f}s")
        print(f"  Result shape: {result_shape}")
        print(f"  Memory: {initial_memory:.1f} -> {final_memory:.1f} MB (Δ{memory_delta:+.1f}, peak: {peak_memory:.1f})")
        
        return {
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'stage3_time': stage3_time,
            'execution_time': total_time,
            'stage1_hash': stage1_hash,
            'stage2_hash': stage2_hash,
            'stage3_hash': stage3_hash,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'peak_memory': peak_memory,
            'memory_delta': memory_delta,
            'result_shape': result_shape,
            'backend': backend,
            'N_households': N_households,
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        tracker.stop_monitoring()
        return None


def main_cli():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Run GETTSIM performance benchmarks')
    parser.add_argument('-scramble', '--scramble-data', action='store_true',
                        help='Scramble data to create unsorted p_id order (default: sorted)')
    
    args = parser.parse_args()
    
    # Dataset sizes (number of households)
    household_sizes = BENCHMARK_HOUSEHOLD_SIZES
    backends = BACKENDS
    
    results = {}
    
    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "household_sizes": household_sizes,
        "backends": backends,
        "scrambled_data": args.scramble_data
    }
    
    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Testing {backend} backend")
        if args.scramble_data:
            print("Data scrambling: ENABLED (unsorted p_id order)")
        else:
            print("Data scrambling: DISABLED (sorted p_id order)")
        print(f"{'='*60}")
        
        # Clear all caches and reset session before starting new backend
        print(f"Preparing environment for {backend} backend...")
        reset_session_state(backend)
        
        for N_households in household_sizes:
            # Add extra session reset for larger datasets to ensure clean state
            reset_between_sizes = N_households >= 2**18  # Reset for 256k+ households
            
            result = run_benchmark(
                N_households, 
                backend, 
                reset_session=False, # reset_between_sizes (no impact on results)
                sync_jax=True,  # Set to True if you want to force JAX synchronization
                                # Seems necessary for realistic (reported time = wall clock time) JAX timings
                scramble_data=args.scramble_data,
            )
            if result and result.get('execution_time'):
                # Store all stage timing data
                results[f"{N_households}_{backend}_stage1_time"] = result['stage1_time']
                results[f"{N_households}_{backend}_stage2_time"] = result['stage2_time'] 
                results[f"{N_households}_{backend}_stage3_time"] = result['stage3_time']
                results[f"{N_households}_{backend}_time"] = result['execution_time']  # Total time
                results[f"{N_households}_{backend}_stage1_hash"] = result['stage1_hash']
                results[f"{N_households}_{backend}_stage2_hash"] = result['stage2_hash']
                results[f"{N_households}_{backend}_stage3_hash"] = result['stage3_hash']
                results[f"{N_households}_{backend}_initial_memory"] = result['initial_memory']
                results[f"{N_households}_{backend}_final_memory"] = result['final_memory']
                results[f"{N_households}_{backend}_memory_delta"] = result['memory_delta']
                results[f"{N_households}_{backend}_peak_memory"] = result['peak_memory']
                results[f"{N_households}_{backend}_result_shape"] = result['result_shape']
            else:
                # Store None values for failed runs
                results[f"{N_households}_{backend}_stage1_time"] = None
                results[f"{N_households}_{backend}_stage2_time"] = None 
                results[f"{N_households}_{backend}_stage3_time"] = None
                results[f"{N_households}_{backend}_time"] = None
                results[f"{N_households}_{backend}_hash"] = None
                results[f"{N_households}_{backend}_initial_memory"] = None
                results[f"{N_households}_{backend}_final_memory"] = None
                results[f"{N_households}_{backend}_memory_delta"] = None
                results[f"{N_households}_{backend}_peak_memory"] = None
                results[f"{N_households}_{backend}_result_shape"] = None
            print()
        
        # Comprehensive cleanup after completing all sizes for this backend
        print(f"Completing {backend} backend tests...")
        print(f"{backend} backend tests completed with full cleanup")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scramble_suffix = "_scrambled" if args.scramble_data else "_sorted"
    filename = f"benchmark_results_{timestamp}{scramble_suffix}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")
    
    print(f"\n{'='*120}")
    print("3-STAGE TIMING BREAKDOWN")
    if args.scramble_data:
        print("Data ordering: SCRAMBLED (unsorted p_id)")
    else:
        print("Data ordering: SORTED (sequential p_id)")
    print(f"{'='*120}")
    
    # Print comparison table in the requested format
    print(f"\n{'='*101}")
    print("PERFORMANCE COMPARISON NUMPY <-> JAX")
    print(f"{'='*104}")
    print(f"{'Households':<12}{'Stage':<18}{'NUMPY hash':<12}{'JAX hash':<12}{'NUMPY (s)':<12}{'JAX (s)':<12}{'Speedup':<12}")
    print("-" * 104)
    
    for N_households in household_sizes:
        # Get timing data for all stages
        numpy_s1 = results.get(f"{N_households}_numpy_stage1_time")
        numpy_s2 = results.get(f"{N_households}_numpy_stage2_time")
        numpy_s3 = results.get(f"{N_households}_numpy_stage3_time")
        numpy_total = results.get(f"{N_households}_numpy_time")
        
        jax_s1 = results.get(f"{N_households}_jax_stage1_time")
        jax_s2 = results.get(f"{N_households}_jax_stage2_time")
        jax_s3 = results.get(f"{N_households}_jax_stage3_time")
        jax_total = results.get(f"{N_households}_jax_time")
        
        # Get stage-specific hashes
        numpy_s1_hash = results.get(f"{N_households}_numpy_stage1_hash")
        numpy_s2_hash = results.get(f"{N_households}_numpy_stage2_hash")
        numpy_s3_hash = results.get(f"{N_households}_numpy_stage3_hash")
        
        jax_s1_hash = results.get(f"{N_households}_jax_stage1_hash")
        jax_s2_hash = results.get(f"{N_households}_jax_stage2_hash")
        jax_s3_hash = results.get(f"{N_households}_jax_stage3_hash")
        
        # Truncate hashes for display, handling both successful and failed cases
        def format_hash_display(hash_value, time_value):
            """Format hash display based on whether the stage succeeded."""
            if time_value is None:
                return "FAILED"
            elif hash_value:
                return hash_value[:8]
            else:
                return "N/A"
        
        numpy_s1_hash_display = format_hash_display(numpy_s1_hash, numpy_s1)
        numpy_s2_hash_display = format_hash_display(numpy_s2_hash, numpy_s2)
        numpy_s3_hash_display = format_hash_display(numpy_s3_hash, numpy_s3)
        
        jax_s1_hash_display = format_hash_display(jax_s1_hash, jax_s1)
        jax_s2_hash_display = format_hash_display(jax_s2_hash, jax_s2)
        jax_s3_hash_display = format_hash_display(jax_s3_hash, jax_s3)
        
        # Helper function to format time display
        def format_time_display(time_value):
            """Format time display for successful or failed runs."""
            return f"{time_value:.4f}" if time_value is not None else "FAILED"
        
        # Helper function to calculate speedup
        def calculate_speedup(numpy_time, jax_time):
            """Calculate speedup string, handling failed cases."""
            if numpy_time is None and jax_time is None:
                return "FAILED"
            elif numpy_time is None:
                return "N/A"
            elif jax_time is None:
                return "N/A"
            elif jax_time > 0:
                speedup = numpy_time / jax_time
                return f"{speedup:.2f}x" if speedup >= 1 else f"1/{jax_time/numpy_time:.2f}x"
            else:
                return "N/A"
        
        # Determine if we should show stage breakdown or overall FAILED
        show_stages = (numpy_total is not None) or (jax_total is not None)
        
        if show_stages:
            # Show individual stage results
            
            # Pre-processing row (Stage 1 hashes often unstable due to dict return)
            s1_speedup_str = calculate_speedup(numpy_s1, jax_s1)
            print(f"{N_households:<12,}{'pre-processing':<18}{'-':<12}{'-':<12}{format_time_display(numpy_s1):<12}{format_time_display(jax_s1):<12}{s1_speedup_str:<12}")
            
            # Computation row (Stage 2 hashes should be stable)
            s2_speedup_str = calculate_speedup(numpy_s2, jax_s2)
            print(f"{'':>12}{'computation':<18}{numpy_s2_hash_display:<12}{jax_s2_hash_display:<12}{format_time_display(numpy_s2):<12}{format_time_display(jax_s2):<12}{s2_speedup_str:<12}")
            
            # Post-processing row (Stage 3 hashes should be stable)
            s3_speedup_str = calculate_speedup(numpy_s3, jax_s3)
            print(f"{'':>12}{'post-processing':<18}{numpy_s3_hash_display:<12}{jax_s3_hash_display:<12}{format_time_display(numpy_s3):<12}{format_time_display(jax_s3):<12}{s3_speedup_str:<12}")
            
            # Total time row
            total_speedup_str = calculate_speedup(numpy_total, jax_total)
            print(f"{'':>12}{'total time':<18}{'':>12}{'':>12}{format_time_display(numpy_total):<12}{format_time_display(jax_total):<12}{total_speedup_str:<12}")
            
            print("-" * 104)
        else:
            # Both backends completely failed
            print(f"{N_households:<12,}{'FAILED':<18}{'FAILED':<12}{'FAILED':<12}{'FAILED':<12}{'FAILED':<12}{'FAILED':<12}")
            print("-" * 104)
    
    # Print memory comparison
    print(f"\n{'='*140}")
    print("MEMORY USAGE COMPARISON")
    print(f"{'='*140}")
    print(f"{'Households':<12}{'NumPy Init':<12}{'NumPy Final':<13}{'NumPy Δ':<12}{'NumPy Peak':<13}{'JAX Init':<12}{'JAX Final':<12}{'JAX Δ':<12}{'JAX Peak':<12}")
    print("-" * 140)
    
    for N_households in household_sizes:
        numpy_init = results.get(f"{N_households}_numpy_initial_memory")
        numpy_final = results.get(f"{N_households}_numpy_final_memory")
        numpy_delta = results.get(f"{N_households}_numpy_memory_delta")
        numpy_peak = results.get(f"{N_households}_numpy_peak_memory")
        jax_init = results.get(f"{N_households}_jax_initial_memory")
        jax_final = results.get(f"{N_households}_jax_final_memory")
        jax_delta = results.get(f"{N_households}_jax_memory_delta")
        jax_peak = results.get(f"{N_households}_jax_peak_memory")
        
        # Helper function to format memory values
        def format_memory(value):
            return f"{value:.1f}" if value is not None else "FAILED"
        
        # Show memory data even if only one backend succeeded
        print(f"{N_households:<12,}{format_memory(numpy_init):<12}{format_memory(numpy_final):<13}{format_memory(numpy_delta):<12}{format_memory(numpy_peak):<13}{format_memory(jax_init):<12}{format_memory(jax_final):<12}{format_memory(jax_delta):<12}{format_memory(jax_peak):<12}")
    
    print("-" * 140)
    print("\nLegend:")
    print("  Stage 1: Data preprocessing & DAG creation")
    print("  Stage 2: Core computation (tax/transfer calculations)")
    print("  Stage 3: Preparing results DataFrame")
    print("  Init/Final: Memory usage before/after execution")
    print("  Δ: Memory increase during execution")
    print("  Peak: Maximum memory usage during execution")
    
    print(f"\n{'='*120}")
    print("BENCHMARK COMPLETED")
    print(f"{'='*120}")
    print(f"Results saved to: {filename}")
    print(f"Generated at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main_cli()