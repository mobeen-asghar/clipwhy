"""Post-extraction CLI: download -> merge -> relabel -> split -> pca -> normalise.

Usage:
  python -m src.post_extraction.cli download
  python -m src.post_extraction.cli merge
  python -m src.post_extraction.cli relabel [--eng-mult 2.0] [--vpd-mult 3.0]
  python -m src.post_extraction.cli split
  python -m src.post_extraction.cli pca
  python -m src.post_extraction.cli normalise
  python -m src.post_extraction.cli all        # runs every step in order
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.post_extraction.cli")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="rclone pull features/, clip_embeddings/, pairs/ from R2")
    sub.add_parser("merge", help="Concatenate per-creator features CSVs")
    p_re = sub.add_parser("relabel", help="Recompute labels at V1's strict thresholds")
    p_re.add_argument("--eng-mult", type=float, default=2.0)
    p_re.add_argument("--vpd-mult", type=float, default=3.0)
    sub.add_parser("split", help="GroupShuffleSplit 70/15/15 by video_id (random_state=42)")
    sub.add_parser("pca", help="Fit PCA(32) on train CLIP, backfill clip_pca_* columns")
    sub.add_parser("normalise", help="Clip emotion to [0,1] and z-score from train stats")
    sub.add_parser("all", help="Run every step in order")

    args = p.parse_args(argv)

    from . import fit_clip_pca, merge, normalise, r2_sync, relabel, split_data

    if args.command == "download":
        return r2_sync.pull_all()
    if args.command == "merge":
        merge.merge()
        return 0
    if args.command == "relabel":
        relabel.relabel(eng_mult=args.eng_mult, vpd_mult=args.vpd_mult)
        return 0
    if args.command == "split":
        split_data.split()
        return 0
    if args.command == "pca":
        fit_clip_pca.fit_and_project()
        return 0
    if args.command == "normalise":
        normalise.normalise()
        return 0
    if args.command == "all":
        rc = r2_sync.pull_all()
        if rc != 0:
            return rc
        merge.merge()
        relabel.relabel()
        split_data.split()
        fit_clip_pca.fit_and_project()
        normalise.normalise()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
