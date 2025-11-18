#!/usr/bin/env python3
"""
pretrain.py

Pretrain LSTM models for one or more tickers and save them under models/<TICKER>.h5.

Usage examples:
  # Train a single ticker with defaults:
  python pretrain.py --ticker AAPL

  # Train multiple tickers:
  python pretrain.py --tickers AAPL TSLA MSFT

  # Train tickers from a file (one ticker per line):
  python pretrain.py --tickers-file tickers.txt --epochs 50 --lookback 60

  # Force retrain even if model exists:
  python pretrain.py --ticker AAPL --force

Notes:
 - This script calls train_model_for_ticker(...) with force_retrain=True when --force is set.
 - The trained model will be saved by model_utils to models/<TICKER>.h5.
 - Use git lfs to track .h5 files if you intend to commit them:
     git lfs track "*.h5"
"""
import argparse
import logging
import os
import sys
from datetime import datetime

from model_utils import train_model_for_ticker, model_filepath_for_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pretrain")


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain LSTM models for given tickers.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", nargs="+", help="One or more tickers, e.g. --ticker AAPL TSLA")
    group.add_argument("--tickers-file", help="Path to a file with one ticker per line")
    p.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD) for training data")
    p.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD) for training data")
    p.add_argument("--lookback", type=int, default=60, help="Lookback window (timesteps)")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--force", action="store_true", help="Force retrain even if saved model exists")
    p.add_argument("--save-only", action="store_true", help="If set, do not train; just print model paths (for debugging)")
    p.add_argument("--max-failures", type=int, default=5, help="Stop after this many failures")
    return p.parse_args()


def load_tickers_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip().upper() for ln in f if ln.strip()]
    return lines


def main():
    args = parse_args()

    # build ticker list
    tickers = []
    if args.ticker:
        tickers = [t.upper().strip() for t in args.ticker]
    elif args.tickers_file:
        tickers = load_tickers_from_file(args.tickers_file)

    if not tickers:
        logger.error("No tickers provided.")
        sys.exit(1)

    logger.info("Pretraining models for %d tickers: %s", len(tickers), ", ".join(tickers))
    logger.info("Date range: %s -> %s | lookback=%d epochs=%d batch_size=%d | force=%s",
                args.start, args.end, args.lookback, args.epochs, args.batch_size, args.force)

    failures = 0
    for ticker in tickers:
        try:
            model_path = model_filepath_for_ticker(ticker)
            if os.path.exists(model_path) and not args.force:
                logger.info("Model already exists for %s at %s — skipping (use --force to retrain)", ticker, model_path)
                continue

            if args.save_only:
                logger.info("Save-only requested — would save model to: %s", model_path)
                continue

            logger.info("Training model for %s ...", ticker)
            res = train_model_for_ticker(
                ticker=ticker,
                start=args.start,
                end=args.end,
                lookback=args.lookback,
                epochs=args.epochs,
                batch_size=args.batch_size,
                future_days=0,
                force_retrain=args.force,
            )
            saved_path = model_filepath_for_ticker(ticker)
            logger.info("Finished training %s — model saved to: %s", ticker, saved_path)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Exiting.")
            sys.exit(2)
        except Exception as e:
            failures += 1
            logger.exception("Failed training for %s: %s", ticker, e)
            if failures >= args.max_failures:
                logger.error("Reached maximum failures (%d). Aborting.", failures)
                break

    logger.info("Pretraining run complete. failures=%d", failures)


if __name__ == "__main__":
    main()
