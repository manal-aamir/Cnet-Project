import argparse
import os

from preprocessing import load_and_preprocess
from training import build_models, train_models
from evaluation import evaluate_models, save_results, print_summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(dataset_path: str, dataset_name: str, output_dir: str):
    """Run the complete ML pipeline for a single dataset."""
    print(f"\n{'#'*60}")
    print(f"# PROCESSING: {dataset_name}")
    print(f"{'#'*60}")
    
    # Step 1: Load and preprocess data
    data = load_and_preprocess(dataset_path)
    
    # Step 2: Build models
    models = build_models()
    
    # Step 3: Train models
    trained_models = train_models(models, data.X_train, data.y_train)
    
    # Step 4: Evaluate models
    metrics_df, detailed_metrics = evaluate_models(
        trained_models, data.X_test, data.y_test, data.class_names
    )
    
    # Step 5: Save results
    save_results(dataset_name, metrics_df, detailed_metrics, output_dir)
    
    # Step 6: Print summary
    print_summary(metrics_df)
    
    print(f"\n{'#'*60}")
    print(f"# COMPLETED: {dataset_name}")
    print(f"{'#'*60}\n")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="WSN DoS Attack Detection using Ensemble ML",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        choices=['WSN-DS', 'WSN-BFSF', 'both'],
        default='both',
        help='Dataset to process (default: both)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    args = parser.parse_args()
    
    # Dataset configurations
    datasets = {
        'WSN-DS': 'WSN-DS.csv',
        'WSN-BFSF': 'WSNBFSFdataset.csv'
    }
    
    # Determine which datasets to process
    if args.dataset == 'both':
        to_process = datasets.items()
    else:
        to_process = [(args.dataset, datasets[args.dataset])]
    
    # Process each dataset
    for name, path in to_process:
        if not os.path.exists(path):
            print(f"\nWarning: '{path}' not found. Skipping {name}...")
            continue
        
        try:
            run_pipeline(path, name, args.output)
        except Exception as e:
            print(f"\nError processing {name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Pipeline execution complete!")
    print(f"Results saved to: {args.output}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
