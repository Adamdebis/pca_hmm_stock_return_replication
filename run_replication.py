"""Run the reproducible PCA + HMM demonstration from the repository root."""

from __future__ import annotations

import argparse

from src.replication import run_full_replication


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    parser.add_argument("--periods", type=int, default=320)
    parser.add_argument("--assets", type=int, default=24)
    parser.add_argument("--train-periods", type=int, default=260)
    parser.add_argument("--forecast-periods", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=35)
    parser.add_argument(
        "--paper-like",
        action="store_true",
        help="use the paper's 10-year/100-week layout on the offline synthetic panel",
    )
    args = parser.parse_args()

    if args.paper_like:
        args.periods = max(args.periods, 620)
        args.assets = max(args.assets, 30)
        args.train_periods = 520
        args.forecast_periods = 100

    outputs = run_full_replication(
        args.output,
        periods=args.periods,
        assets=args.assets,
        train_periods=args.train_periods,
        forecast_periods=args.forecast_periods,
        seed=args.seed,
        max_iter=args.max_iter,
    )
    print("PCA + HMM replication completed.")
    print(f"Forecast rows: {len(outputs['results'])}")
    print(f"Summary rows: {len(outputs['summary'])}")
    print(f"Reported-paper rows: {len(outputs['paper'])}")


if __name__ == "__main__":
    main()
