"""
Post-run merge: combine per-creator CSVs into final master files.

Run after all VMs complete. Produces:
  - final/segments.csv          (all labeled segments)
  - final/video_pairs.csv       (all Whisper-matched pairs)
  - final/all_shorts_metadata.csv (all shorts with engagement)
  - final/creator_baselines.csv  (all creator baselines)

Also runs validation checks and prints a summary report.
"""

import logging

import pandas as pd

from . import config

log = logging.getLogger("clipwhy.merge")


def merge_all():
    """Merge all per-creator outputs into final master files."""
    config.ensure_directories()

    print("=" * 70)
    print("  MERGING PER-CREATOR OUTPUTS")
    print("=" * 70)

    # ── 1. Merge labeled segments ───────────────────────────────────────────
    seg_files = sorted(config.LABELED_DIR.glob("*_segments.csv"))
    if not seg_files:
        print("\nNo segment files found. Nothing to merge.")
        return

    print(f"\nMerging {len(seg_files)} segment files...")
    segments = pd.concat(
        [pd.read_csv(f) for f in seg_files], ignore_index=True
    )

    # Assign global segment IDs
    segments["segment_id"] = [
        f"SEG_{row['video_id'][:8]}_{row['segment_index']:03d}"
        for _, row in segments.iterrows()
    ]

    seg_path = config.FINAL_DIR / "segments.csv"
    segments.to_csv(seg_path, index=False)
    print(f"  segments.csv: {len(segments):,} rows")

    # ── 2. Merge pairs ──────────────────────────────────────────────────────
    pair_files = sorted(config.PAIRS_DIR.glob("*_whisper_pairs.csv"))
    if pair_files:
        print(f"Merging {len(pair_files)} pair files...")
        pairs = pd.concat(
            [pd.read_csv(f) for f in pair_files], ignore_index=True
        )
        # Assign global pair IDs
        pairs.insert(0, "pair_id", [f"WP{i:04d}" for i in range(1, len(pairs) + 1)])
        pairs_path = config.FINAL_DIR / "video_pairs.csv"
        pairs.to_csv(pairs_path, index=False)
        print(f"  video_pairs.csv: {len(pairs):,} rows")
    else:
        pairs = pd.DataFrame()
        print("  No pair files found")

    # ── 3. Merge shorts metadata ────────────────────────────────────────────
    meta_files = sorted(config.METADATA_DIR.glob("*_shorts.csv"))
    if meta_files:
        print(f"Merging {len(meta_files)} metadata files...")
        metadata = pd.concat(
            [pd.read_csv(f) for f in meta_files], ignore_index=True
        )
        meta_path = config.FINAL_DIR / "all_shorts_metadata.csv"
        metadata.to_csv(meta_path, index=False)
        print(f"  all_shorts_metadata.csv: {len(metadata):,} rows")
    else:
        metadata = pd.DataFrame()
        print("  No metadata files found")

    # ── 4. Merge baselines ──────────────────────────────────────────────────
    bl_files = sorted(config.METADATA_DIR.glob("*_baselines.csv"))
    if bl_files:
        print(f"Merging {len(bl_files)} baseline files...")
        baselines = pd.concat(
            [pd.read_csv(f) for f in bl_files], ignore_index=True
        )
        bl_path = config.FINAL_DIR / "creator_baselines.csv"
        baselines.to_csv(bl_path, index=False)
        print(f"  creator_baselines.csv: {len(baselines):,} rows")
    else:
        baselines = pd.DataFrame()

    # ── 5. Create master index ─────────────────────────────────────────────
    print(f"\nCreating master index...")
    master_rows = []
    for _, seg in segments.iterrows():
        vid = seg["video_id"]
        cid = seg["creator_id"]
        idx = seg["segment_index"]
        master_rows.append({
            "segment_id": seg["segment_id"],
            "creator_id": cid,
            "category": seg["category"],
            "video_id": vid,
            "segment_index": idx,
            "label": seg["label"],
            "matched_short_id": seg.get("matched_short_id", ""),
            "transcript_path": f"transcripts/segments/{vid}/segment_{idx:03d}.json",
            "audio_path": f"segments/{vid}/audio/segment_{idx:03d}.wav",
            "video_path": f"segments/{vid}/video/segment_{idx:03d}.mp4",
            "thumbnail_path": f"segments/{vid}/thumbnails/segment_{idx:03d}.jpg",
            "raw_long_audio": f"raw/long/{vid}.wav",
            "raw_long_video": f"raw/long/{vid}.mp4",
            "short_audio": f"raw/shorts_audio/{seg.get('matched_short_id', '')}.wav" if seg.get("matched_short_id") else "",
            "short_video": f"raw/shorts_video/{seg.get('matched_short_id', '')}.mp4" if seg.get("matched_short_id") else "",
            "short_transcript": f"transcripts/shorts/{seg.get('matched_short_id', '')}.json" if seg.get("matched_short_id") else "",
        })
    master = pd.DataFrame(master_rows)
    master_path = config.FINAL_DIR / "master_index.csv"
    master.to_csv(master_path, index=False)
    print(f"  master_index.csv: {len(master):,} rows, maps every segment to all its files")

    # ── Validation ──────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  VALIDATION")
    print("-" * 70)

    checks_passed = 0
    checks_failed = 0

    def check(name, condition):
        nonlocal checks_passed, checks_failed
        if condition:
            checks_passed += 1
            print(f"  PASS  {name}")
        else:
            checks_failed += 1
            print(f"  FAIL  {name}")

    # Segment checks
    check("Segments exist", len(segments) > 0)
    check("Labels are 0 or 1", set(segments["label"].unique()).issubset({0, 1}))
    check("Positive segments exist", segments["label"].sum() > 0)
    check("No duplicate segment IDs", segments["segment_id"].nunique() == len(segments))
    check("Position ratio in [0, 1]",
          segments["position_ratio"].between(0, 1).all())
    check("24 columns", len(segments.columns) == 24)

    # Pair checks
    if len(pairs) > 0:
        check("Pairs exist", len(pairs) > 0)
        check("No duplicate pair IDs", pairs["pair_id"].nunique() == len(pairs))

    # Metadata checks
    if len(metadata) > 0:
        check("Metadata exists", len(metadata) > 0)
        check("Engagement rates valid",
              metadata["engagement_rate"].between(0, 1).all())

    # Cross-reference checks
    if len(pairs) > 0:
        matched_vids = set(segments[segments["matched_short_id"] != ""]["video_id"])
        check("Matched segments have video_id",
              len(matched_vids) > 0)

    # ── Summary report ──────────────────────────────────────────────────────
    positive = segments[segments["label"] == 1]
    unique_vids = segments["video_id"].nunique()
    unique_creators = segments["creator_id"].nunique()

    print(f"\n  Checks: {checks_passed} passed, {checks_failed} failed")

    print("\n" + "=" * 70)
    print("  FINAL DATASET SUMMARY")
    print("=" * 70)

    print(f"\n  Creators: {unique_creators}")
    print(f"  Long videos: {unique_vids}")
    print(f"  Total segments: {len(segments):,}")
    print(f"  Positive segments: {len(positive)} ({len(positive)/len(segments)*100:.1f}%)")
    print(f"  Negative segments: {len(segments) - len(positive):,}")
    if len(pairs) > 0:
        print(f"  Total pairs: {len(pairs)}")
        viral = pairs[pairs["label"] == 1] if "label" in pairs.columns else pd.DataFrame()
        if len(viral) > 0:
            print(f"  Viral pairs: {len(viral)} ({len(viral)/len(pairs)*100:.1f}%)")
    if len(metadata) > 0:
        print(f"  Total shorts: {len(metadata):,}")

    # Per-category breakdown
    print("\n  Per category:")
    categories = config.CATEGORIES if hasattr(config, "CATEGORIES") else segments["category"].unique()
    for cat in sorted(segments["category"].unique()):
        cat_segs = segments[segments["category"] == cat]
        cat_pos = cat_segs["label"].sum()
        cat_vids = cat_segs["video_id"].nunique()
        cat_creators = cat_segs["creator_id"].nunique()
        print(f"    {cat:<15} {cat_creators:>3} creators, {cat_vids:>4} videos, "
              f"{len(cat_segs):>6} segments, {cat_pos:>3} positive")

    print(f"\n  Output: {config.FINAL_DIR}")
    print("=" * 70)
