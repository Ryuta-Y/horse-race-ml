#!/usr/bin/env python
import argparse, json
from scripts.api_utils import submit_vote_single

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="data/api_out/<race_id>.json")
    args = ap.parse_args()
    with open(args.json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    res = submit_vote_single(payload)
    print("Bet Response:", json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
