"""
Full Analysis Pipeline: Adversarial Training + Cross-Dataset Validation
======================================================================

This script:
1. Trains robust models using adversarial training (handles attacks)
2. Downloads WSNBFSF dataset from Kaggle
3. Performs cross-dataset validation using the Kaggle dataset

Usage:
    python run_full_analysis.py
"""

import sys
from advanced_analysis import (
    train_robust_models,
    run_advanced_analysis_on_saved_models,
    download_kaggle_dataset
)

if __name__ == "__main__":
    print("=" * 70)
    print("FULL ANALYSIS PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("  1. Train robust models that handle adversarial attacks")
    print("  2. Download WSNBFSF dataset from Kaggle")
    print("  3. Perform cross-dataset validation")
    print("=" * 70)
    
    try:
        # Step 1: Train robust models (handles adversarial attacks)
        print("\n" + "=" * 70)
        print("STEP 1: TRAINING ROBUST MODELS (Handles Adversarial Attacks)")
        print("=" * 70)
        robust_results = train_robust_models(
            data_dir="data",
            output_dir="outputs",
            test_size=0.33,
            random_state=42,
            adv_ratio=0.3,
            eps=0.1,
            n_iterations=3
        )
        
        print("\n" + "=" * 70)
        print("STEP 2: CROSS-DATASET VALIDATION WITH KAGGLE DATASET")
        print("=" * 70)
        
        # Step 2: Download Kaggle dataset
        print("\n[DOWNLOAD] Downloading WSNBFSF dataset from Kaggle...")
        kaggle_path = download_kaggle_dataset("celilokur/wsnbfsfdataset", "data")
        
        if kaggle_path:
            print(f"[DOWNLOAD] ✓ Dataset ready: {kaggle_path}")
        else:
            print("[DOWNLOAD] ⚠ Kaggle dataset download failed, will use local datasets")
        
        # Step 3: Run advanced analysis with cross-dataset validation
        print("\n[ANALYSIS] Running advanced analysis with cross-dataset validation...")
        analysis_results = run_advanced_analysis_on_saved_models(
            data_dir="data",
            output_dir="outputs",
            test_size=0.33,
            random_state=42
        )
        
        print("\n" + "=" * 70)
        print("✓ FULL ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nResults:")
        print("  ✓ Robust models trained (handle adversarial attacks)")
        print("  ✓ Cross-dataset validation completed")
        print("  ✓ All results saved to outputs/")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

