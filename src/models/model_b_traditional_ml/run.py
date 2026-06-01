"""Run RF and XGBoost on the test split (5 seeds each)."""
from __future__ import annotations

import argparse
import logging
import sys

from . import random_forest, xgboost_model


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--model", default="both", choices=["rf", "xgb", "both"])
    args = p.parse_args(argv)
    if args.model in ("rf", "both"):
        random_forest.run(args.split)
    if args.model in ("xgb", "both"):
        xgboost_model.run(args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
