"""Run both baselines (random + rule-based) on val and test."""
from __future__ import annotations

import argparse
import logging
import sys

from . import random_baseline, rule_based


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test", "both"])
    args = parser.parse_args(argv)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    for s in splits:
        random_baseline.run(s)
        rule_based.run(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
