#!/usr/bin/env python3
"""
Example script demonstrating how to use centralized constants for ARC-AGI training.

This script shows how to:
1. Use different model configurations
2. Use different training profiles
3. Access tokenization constants
4. Estimate model parameters
"""

from arc.utils.constants import (MODEL_CONFIGS, TRAINING_CONFIGS, VOCAB_SIZE,
                                 estimate_model_parameters,
                                 get_matched_configs, get_model_config,
                                 get_training_config)


def main():
    print("ARC-AGI Centralized Constants Demo")
    print("=" * 50)
    
    # Show available configurations
    print("\nAvailable Model Sizes:")
    for size in MODEL_CONFIGS.keys():
        config = get_model_config(size)
        params = estimate_model_parameters(config)
        print(f"  {size:8}: {params:8,} parameters, d_model={config['d_model']}")
    
    print("\nAvailable Training Profiles:")
    for profile in TRAINING_CONFIGS.keys():
        config = get_training_config(profile)
        print(f"  {profile:12}: batch_size={config['batch_size']:2}, max_len={config['max_sequence_length']:4}")
    
    # Example: Get a specific configuration
    print("\nExample: Using 'small' model with 'medium_gpu' training:")
    model_config, train_config = get_matched_configs('small', 'medium_gpu')
    
    print(f"Model: {model_config['d_model']}D, {model_config['n_layers']} layers, max_len={model_config['max_len']}")
    print(f"Training: batch_size={train_config['batch_size']}, lr={train_config['learning_rate']}, max_seq_len={train_config['max_sequence_length']}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"✓ Sequence lengths aligned: model_max_len == training_max_seq_len")
    
    # Memory estimation
    params = estimate_model_parameters(model_config)
    print(f"Estimated parameters: {params:,}")
    print(f"Estimated model size: ~{params * 4 / 1024**2:.1f} MB (FP32)")

if __name__ == "__main__":
    main()    main()