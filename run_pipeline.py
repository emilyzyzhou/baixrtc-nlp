#!/usr/bin/env python3
import sys
from pipeline import run_pipeline

if __name__ == "__main__":
    responses_df, summary_df = run_pipeline()
    print("Pipeline execution complete!")
