"""CLI entry point for feature extraction.

Commands:
  run        Launch pod orchestrator; pulls from dynamic pool until empty.
  status     Print pool status (done / active / pending) from R2.
  release    Manually release a stuck claim (e.g., if you know a pod died).
  merge      Concatenate per-creator features CSVs into final/features.csv.
  verify     Sanity-check features CSVs on R2 (row counts, column shape, NaN).

Usage:
  python -m src.feature_extraction.cli run --vm-id gpu0
  python -m src.feature_extraction.cli status
  python -m src.feature_extraction.cli run --vm-id gpu0 --max-creators 3
"""
import argparse
import json
import logging
import sys

from . import claims, config, r2_client

log = logging.getLogger("clipwhy.features.cli")


def cmd_run(args: argparse.Namespace) -> int:
    from .orchestrator import run as orchestrator_run

    orchestrator_run(
        vm_id=args.vm_id,
        device=args.device,
        max_creators=args.max_creators,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    status = claims.pool_status()
    print(json.dumps(status, indent=2))
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    for creator_id in args.creator_ids:
        key = f"{config.R2_CLAIMS_PREFIX}/{creator_id}.json"
        r2_client.delete(key)
        print(f"Deleted claim: {key}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Concatenate per-creator features CSVs into final/features.csv."""
    import io

    import pandas as pd

    keys = r2_client.list_keys(config.R2_FEATURES_PREFIX + "/")
    per_creator_keys = [
        k for k in keys
        if k.split("/")[-1].endswith("_features.csv")
        and not k.endswith("final/features.csv")
    ]
    log.info("Merging %d per-creator feature CSVs", len(per_creator_keys))

    frames = []
    for k in sorted(per_creator_keys):
        blob = r2_client.get(k)
        if blob is None:
            log.warning("Missing: %s", k)
            continue
        frames.append(pd.read_csv(io.BytesIO(blob)))
    if not frames:
        log.error("No feature CSVs found")
        return 1

    merged = pd.concat(frames, ignore_index=True)
    log.info("Merged: %d rows, %d columns", len(merged), len(merged.columns))

    out = io.BytesIO()
    merged.to_csv(out, index=False)
    r2_client.put("final/features.csv", out.getvalue())
    log.info("Wrote final/features.csv: %d bytes", out.tell())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Spot-check per-creator feature CSVs on R2."""
    import io

    import pandas as pd

    keys = r2_client.list_keys(config.R2_FEATURES_PREFIX + "/")
    per_creator_keys = [k for k in keys if k.split("/")[-1].endswith("_features.csv")]
    print(f"Found {len(per_creator_keys)} feature CSVs")

    problems = []
    expected_cols = set(config.OUTPUT_COLUMN_ORDER)
    for k in sorted(per_creator_keys)[: args.limit]:
        blob = r2_client.get(k)
        if blob is None:
            problems.append((k, "missing"))
            continue
        df = pd.read_csv(io.BytesIO(blob))
        cols = set(df.columns)
        if cols != expected_cols:
            missing = expected_cols - cols
            extra = cols - expected_cols
            problems.append((k, f"col mismatch: missing={missing} extra={extra}"))
            continue
        nan_rate = df[list(config.FEATURE_COLUMNS)].isna().sum().sum() / (len(df) * len(config.FEATURE_COLUMNS))
        if nan_rate > 0.005:
            problems.append((k, f"nan_rate={nan_rate:.4f}"))

    if problems:
        for k, prob in problems[:20]:
            print(f"  ISSUE {k}: {prob}")
        print(f"Total issues: {len(problems)}")
        return 1
    print("All verified CSVs pass shape + NaN checks.")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(prog="python -m src.feature_extraction.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run pod orchestrator")
    p_run.add_argument("--vm-id", required=True, help="Unique pod id (e.g. gpu0)")
    p_run.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p_run.add_argument("--max-creators", type=int, default=None,
                       help="Stop after N creators (for testing)")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="Show pool status")
    p_status.set_defaults(func=cmd_status)

    p_release = sub.add_parser("release", help="Manually release claims")
    p_release.add_argument("creator_ids", nargs="+")
    p_release.set_defaults(func=cmd_release)

    p_merge = sub.add_parser("merge", help="Merge per-creator features into final/features.csv")
    p_merge.set_defaults(func=cmd_merge)

    p_verify = sub.add_parser("verify", help="Sanity-check features CSVs on R2")
    p_verify.add_argument("--limit", type=int, default=20)
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
