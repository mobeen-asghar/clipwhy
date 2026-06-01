"""Produce a BERT-ready CSV by joining feature matrix segments with transcripts.

BERT fine-tuning uses the raw transcript text, not the engineered features.
Transcripts live on R2 at labeled/CRxxxx_segments.csv (column: transcript_text).
This helper either (a) reads from the local labeled/ cache if already pulled,
or (b) pulls labeled/ via rclone.

Output: data/post_extraction/bert_data_{train,val,test}.csv
Columns: segment_id, video_id, creator_id, transcript_text, label, split

Run on a GPU pod AFTER post-extraction is complete and labeled/ is synced.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.post_extraction import config as post_config

LABELED_DIR = post_config.R2_CACHE / "labeled"
BERT_TRAIN = post_config.OUT_DIR / "bert_data_train.csv"
BERT_VAL = post_config.OUT_DIR / "bert_data_val.csv"
BERT_TEST = post_config.OUT_DIR / "bert_data_test.csv"

log = logging.getLogger("clipwhy.model_c.prepare")


def pull_labeled_if_needed() -> None:
    existing = list(LABELED_DIR.glob("CR*_segments.csv"))
    if len(existing) >= 394:
        log.info("Labeled dir already has %d CSVs; skipping pull", len(existing))
        return
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("rclone copy labeled/ -> %s", LABELED_DIR)
    subprocess.run(
        ["rclone", "copy", f"{post_config.R2_REMOTE}/labeled/", str(LABELED_DIR),
         "--disable-http2", "--transfers", "16", "--update"],
        check=True,
    )


def build() -> None:
    pull_labeled_if_needed()
    files = sorted(LABELED_DIR.glob("CR*_segments.csv"))
    if not files:
        raise FileNotFoundError(f"No labeled CSVs in {LABELED_DIR}")
    log.info("Reading transcript text from %d labeled CSVs", len(files))
    pieces = []
    for f in files:
        df = pd.read_csv(f, usecols=["segment_id", "video_id", "creator_id",
                                      "transcript_text"])
        pieces.append(df)
    transcripts = pd.concat(pieces, ignore_index=True)
    transcripts["transcript_text"] = transcripts["transcript_text"].fillna("")

    splits = pd.read_csv(post_config.SEGMENTS_WITH_SPLITS_CSV,
                         usecols=["segment_id", "label", "split"])

    merged = splits.merge(transcripts, on="segment_id", how="left")
    missing = int(merged["transcript_text"].isna().sum())
    if missing:
        log.warning("%d segments have no transcript_text (will be empty string)", missing)
        merged["transcript_text"] = merged["transcript_text"].fillna("")

    for split_name, out_path in [("train", BERT_TRAIN), ("val", BERT_VAL), ("test", BERT_TEST)]:
        sub = merged[merged["split"] == split_name].copy()
        sub.to_csv(out_path, index=False)
        log.info("Wrote %s: %d rows, %d positives",
                 out_path.name, len(sub), int(sub["label"].sum()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.parse_args()
    build()


if __name__ == "__main__":
    main()
