#!/usr/bin/env python
import argparse
from scripts.utils.io_utils import load_records, build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-glob", required=True, help="data/raw/record_data_*.csv")
    args = ap.parse_args()
    rec = load_records(args.records_glob)
    feats = build_features(rec)
    rec.to_csv("data/processed/race_results.csv", index=False)
    feats.to_csv("data/processed/features.csv", index=False)
    print("OK: wrote data/processed/{race_results.csv, features.csv}")

if __name__ == "__main__":
    main()
