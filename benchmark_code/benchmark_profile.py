"""
GETTSIM Profiling Script

This script profiles GETTSIM/TTSIM with synthetic data.
It supports both JAX and NumPy backends with memory tracking.

Usage:
    python benchmark_profile.py -N 32768 -b numpy (without profile)
    py-spy record -o profile.svg -- python benchmark_profile.py -N 32768 -b numpy (with profile)

"""

import time
import argparse
import hashlib

# Import shared benchmark configuration and utilities
from benchmark_setup import (
    main, TT_TARGETS, MAPPER, JAX_AVAILABLE,
    sync_jax_if_needed, get_memory_usage_mb, MemoryTracker,
    PROFILE_HOUSEHOLD_SIZES, BACKENDS
)
from benchmark_make_data import make_data

# Import GETTSIM/TTSIM components
from gettsim import InputData, MainTarget, TTTargets, Labels, SpecializedEnvironment, RawResults


def run_profile(N, backend, scramble_data=False):
    """Run GETTSIM profiling with specified parameters and memory tracking."""
    print(f"Generating dataset with {N:,} households...")
    data = make_data(N, scramble_data=scramble_data)
    print(f"Dataset created successfully. Shape: {data.shape}")
    
    print(f"Running GETTSIM with backend: {backend}")
    
    # Memory tracking setup
    tracker = MemoryTracker()
    initial_memory = get_memory_usage_mb()
    tracker.start_monitoring()
    
    try:
        # First stage - preprocessing and DAG creation
        print("\n=== STAGE 1: Data preprocessing and DAG creation ===")
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
                MainTarget.tt_function, # Use compiled tt_function in stage 2 with JAX backend
            ],
            tt_targets=TTTargets(
                tree=TT_TARGETS,
            ),
            include_fail_nodes=True,
            include_warn_nodes=False,
            backend=backend,
        )    

        # Force JAX synchronization before recording end time
        sync_jax_if_needed(backend)
        
        stage1_end = time.time()
        stage1_time = stage1_end - stage1_start
        
        # Generate hash for Stage 1 output (tmp) - avoid memory issues with large arrays
        stage1_hash = hashlib.md5(str(tmp).encode('utf-8')).hexdigest()

        print(f"Stage 1 completed in: {stage1_time:.4f} seconds")
        print(f"Processed data keys: {len(tmp['processed_data'])}")
        print(f"DAG nodes: {len(tmp['specialized_environment']['tt_dag'])}")
        print(f"Stage 1 hash: {stage1_hash[:16]}...")

        # Second stage - computation only (no data preprocessing)
        print("\n=== STAGE 2: Computation only (no preprocessing) ===")
        print(f"Wall clock time: {time.strftime('%H:%M:%S')} - Starting Stage 2")
        stage2_start = time.time()

        # Get all three raw results components that we need for stage 3
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
        sync_jax_if_needed(backend)

        stage2_end = time.time()
        print(f"Wall clock time: {time.strftime('%H:%M:%S')} - Completed Stage 2")
        stage2_time = stage2_end - stage2_start
        
        # Generate hash for Stage 2 output - avoid memory issues with large JAX arrays
        stage2_hash = hashlib.md5(str(raw_results_stage2).encode('utf-8')).hexdigest()
        
        print(f"Stage 2 completed in: {stage2_time:.4f} seconds")
        print(f"Raw results components: {list(raw_results_stage2['raw_results'].keys())}")
        print(f"Stage 2 hash: {stage2_hash[:16]}...")

        # Third stage - convert raw results to DataFrame (no computation, just formatting)
        print("\n=== STAGE 3: Convert raw results to DataFrame ===")
        print(f"Wall clock time: {time.strftime('%H:%M:%S')} - Starting Stage 3")
        stage3_start = time.time()

        final_results = main(
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
        sync_jax_if_needed(backend)

        stage3_end = time.time()
        print(f"Wall clock time: {time.strftime('%H:%M:%S')} - Completed Stage 3")
        stage3_time = stage3_end - stage3_start
        total_time = stage1_time + stage2_time + stage3_time
        
        # Generate hash for Stage 3 output - avoid memory issues
        stage3_hash = hashlib.md5(str(final_results).encode('utf-8')).hexdigest()
        
        # Stop memory tracking and get final readings
        tracker.stop_monitoring()
        final_memory = get_memory_usage_mb()
        peak_memory = tracker.get_peak()
        memory_delta = final_memory - initial_memory
        
        print(f"Stage 3 completed in: {stage3_time:.4f} seconds")
        print(f"Final DataFrame shape: {final_results.shape if hasattr(final_results, 'shape') else 'N/A'}")
        print(f"Final DataFrame type: {type(final_results)}")
        print(f"Stage 3 hash: {stage3_hash[:16]}...")
        print(f"Total execution time: {total_time:.4f} seconds")
        print(f"Stage 1 (preprocessing): {stage1_time:.4f}s ({stage1_time/total_time*100:.1f}%)")
        print(f"Stage 2 (computation): {stage2_time:.4f}s ({stage2_time/total_time*100:.1f}%)")
        print(f"Stage 3 (formatting): {stage3_time:.4f}s ({stage3_time/total_time*100:.1f}%)")
        print(f"Backend: {backend}")
        print(f"Households: {N:,}")
        print(f"People: {len(data):,}")
        print(f"Performance: {N / total_time:.0f} households/second")
        print(f"Memory: {initial_memory:.1f} -> {final_memory:.1f} MB (Δ{memory_delta:+.1f}, peak: {peak_memory:.1f})")
        print("\n=== STAGE HASHES ===")
        print(f"Stage 1 hash: {stage1_hash[:16]}...")
        print(f"Stage 2 hash: {stage2_hash[:16]}...")
        print(f"Stage 3 hash: {stage3_hash[:16]}...")
        
        return final_results, total_time
        
    except Exception as e:
        print(f"ERROR during profiling: {e}")
        tracker.stop_monitoring()
        return None, None


def main_cli():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Profile GETTSIM with synthetic data')
    parser.add_argument('-N', '--households', type=int, default=32768,
                        help='Number of households to generate (default: 32768)')
    parser.add_argument('-b', '--backend', choices=BACKENDS, default='numpy',
                        help='Backend to use: numpy or jax (default: numpy)')
    parser.add_argument('-scramble', '--scramble-data', action='store_true',
                        help='Scramble data to create unsorted p_id order (default: sorted)')
    
    args = parser.parse_args()
    
    print("GETTSIM Profiling Tool")
    print("=" * 50)
    
    result, exec_time = run_profile(args.households, args.backend, args.scramble_data)
    
    if result is not None:
        print("\n" + "=" * 50)
        print("Profiling completed successfully!")
    else:
        print("\n" + "=" * 50)
        print("Profiling failed!")
    
    return result, exec_time


if __name__ == "__main__":
    main_cli()

# %%
# For interactive use - you can also run this directly
# result, exec_time = run_profile(N=32768, backend="numpy", scramble_data=False)