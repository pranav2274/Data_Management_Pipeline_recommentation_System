#!/usr/bin/env python3
import argparse
from src.orchestration.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end recommendation data pipeline")
    parser.add_argument("--regenerate-data", action="store_true", help="Regenerate deterministic synthetic CSV inputs")
    args = parser.parse_args()
    summary = run_pipeline(force_data=args.regenerate_data)
    print("\nRUN SUMMARY")
    print(f"Status: {summary['status']}")
    for key, value in summary.get("counts", {}).items():
        print(f"{key}: {value}")
