"""CLI for generating fictional data and evaluating forecasting methods."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .pipeline import evaluate, generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate")
    create.add_argument("--out", type=Path, default=Path("output/synthetic_sales.csv"))
    create.add_argument("--seed", type=int, default=42)
    compare = sub.add_parser("evaluate")
    compare.add_argument("--data", type=Path, default=Path("output/synthetic_sales.csv"))
    compare.add_argument("--test-months", type=int, default=6)
    args = parser.parse_args()
    if args.command == "generate":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        generate(seed=args.seed).to_csv(args.out, index=False)
        print(f"Wrote generated sample data to {args.out}")
    else:
        result = evaluate(pd.read_csv(args.data), args.test_months)
        print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
