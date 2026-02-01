#!/usr/bin/env python3
"""
Profile the worker to find bottlenecks
"""
import cProfile
import pstats
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up profiling
profiler = cProfile.Profile()

# Import and run worker with profiling
def profile_separation():
    from workers.bsroformer_worker import separate_impl, load_model_impl
    import torch
    import json
    
    # Load model
    models_dir = "/Users/ostinsolo/Documents/test_for_instalatin/ThirdPartyApps/Models"
    model, config, model_info = load_model_impl("bsroformer_4stem", models_dir)
    model = model.to(torch.device("cpu"))
    model.eval()
    
    # Run separation with profiling
    input_path = "/Users/ostinsolo/Documents/15_23_5_1_20_2026_.wav"
    output_dir = "/Users/ostinsolo/Desktop/SharedCloud/test_output_intel/profile_test"
    os.makedirs(output_dir, exist_ok=True)
    
    profiler.enable()
    try:
        separate_impl(
            model, config, input_path, output_dir, model_info,
            device='cpu',
            overlap=2,
            batch_size=4,
            use_fast=False
        )
    finally:
        profiler.disable()
    
    # Save stats
    profiler.dump_stats('/tmp/worker_profile.prof')
    
    # Print top time consumers
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n" + "="*70)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME:")
    print("="*70)
    stats.print_stats(30)
    
    print("\n" + "="*70)
    print("TOP 30 FUNCTIONS BY TOTAL TIME:")
    print("="*70)
    stats.sort_stats('tottime')
    stats.print_stats(30)

if __name__ == "__main__":
    profile_separation()
