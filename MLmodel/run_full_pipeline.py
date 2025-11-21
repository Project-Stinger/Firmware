#!/usr/bin/env python3
"""
Master script to run the complete ML pipeline.

This script runs all steps in sequence:
1. Data exploration
2. Preprocessing
3. Feature engineering
4. Train logistic regression
5. Train random forest
6. Train neural network
7. Evaluate and compare models
8. Export for deployment
"""

import sys
import subprocess
import time


def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False


def main():
    print("="*80)
    print("NERF GUN PRE-FIRE PREDICTION - FULL ML PIPELINE")
    print("="*80)
    print("\nThis will run all 8 steps of the ML pipeline.")
    print("Expected total time: 10-30 minutes (depending on hardware)")
    print("\nSteps:")
    print("  1. Data exploration")
    print("  2. Data preprocessing")
    print("  3. Feature engineering")
    print("  4. Train Logistic Regression")
    print("  5. Train Random Forest")
    print("  6. Train Neural Network (PyTorch)")
    print("  7. Evaluate and compare models")
    print("  8. Export for deployment")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    start_time = time.time()

    # Pipeline steps
    steps = [
        ("1_data_exploration.py", "Step 1: Data Exploration"),
        ("2_preprocessing.py", "Step 2: Data Preprocessing"),
        ("3_feature_engineering.py", "Step 3: Feature Engineering"),
        ("4_train_logistic_regression.py", "Step 4: Train Logistic Regression"),
        ("5_train_random_forest.py", "Step 5: Train Random Forest"),
        ("6_train_neural_network.py", "Step 6: Train Neural Network"),
        ("7_evaluate_models.py", "Step 7: Evaluate and Compare Models"),
        ("8_export_for_deployment.py", "Step 8: Export for Deployment"),
    ]

    failed_steps = []

    for script, description in steps:
        success = run_script(script, description)

        if not success:
            failed_steps.append(description)
            print(f"\n⚠ Error in {description}")
            response = input("Continue with next step? (y/n): ")
            if response.lower() != 'y':
                break

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Completed steps: {len(steps) - len(failed_steps)}/{len(steps)}")

    if failed_steps:
        print(f"\n⚠ Failed steps:")
        for step in failed_steps:
            print(f"  - {step}")
    else:
        print("\n✓ All steps completed successfully!")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Review results in outputs/evaluation/")
        print("   - model_comparison.csv")
        print("   - deployment_comparison.csv")
        print("   - ROC curves and visualizations")

        print("\n2. Check deployment files in outputs/deployment/")
        print("   - rf_model.h (C++ header file)")
        print("   - usage_example.cpp")
        print("   - README.md")

        print("\n3. Integrate with firmware:")
        print("   - Copy rf_model.h to src/")
        print("   - Update ml_predictor.cpp")
        print("   - Test on hardware")

        print("\n4. Monitor performance and iterate if needed")


if __name__ == '__main__':
    main()
