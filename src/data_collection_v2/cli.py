"""
CLI entry point for the V2 data collection pipeline.

Commands:
  assign  - Generate VM-to-creator assignment (run once)
  run     - Run the pipeline on a VM (run on each VM)
  status  - Check progress across all VMs
  merge   - Combine per-creator outputs into final files (run once after)
"""

import argparse
import logging
import sys
import threading

import pandas as pd

from . import config
from . import progress
from .orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
# Force flush for nohup/tmux
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

log = logging.getLogger("clipwhy.cli")


def cmd_assign(args):
    """Generate VM-to-creator assignment."""
    creators_df = pd.read_csv(args.creators_csv)
    log.info("Loaded %d creators", len(creators_df))

    # Sort by longs_count descending for load balancing
    creators_df = creators_df.sort_values("longs_count", ascending=False)

    # Round-robin assign to VMs
    num_vms = args.num_vms
    assignments = {f"vm{i}": [] for i in range(num_vms)}

    for idx, (_, row) in enumerate(creators_df.iterrows()):
        vm_key = f"vm{idx % num_vms}"
        assignments[vm_key].append(row["creator_id"])

    progress.save_assignment(assignments)

    # Also copy creators.csv to shared storage
    config.ensure_directories()
    import shutil
    dest = config.INPUT_DIR / "creators.csv"
    shutil.copy2(args.creators_csv, dest)
    log.info("Copied creators.csv to %s", dest)

    # Summary
    print("\nAssignment generated:")
    for vm_id, cids in assignments.items():
        print(f"  {vm_id}: {len(cids)} creators")
    print(f"\nSaved to: {config.INPUT_DIR / 'assignment.json'}")


def cmd_run(args):
    """Run the pipeline on this VM."""
    # Import here to avoid loading heavy deps at startup
    sys.path.insert(0, str(config._PROJECT_ROOT))
    from src.data_collection.youtube_api import YouTubeAPI
    from src.data_collection.notifier import Notifier

    # Create API with thread-safe wrapper
    api = YouTubeAPI(config.YOUTUBE_API_KEYS)
    _api_lock = threading.Lock()
    _original_call = api._call

    def _thread_safe_call(*a, **kw):
        with _api_lock:
            return _original_call(*a, **kw)

    api._call = _thread_safe_call

    notify = Notifier(
        webhook_url=config.DISCORD_WEBHOOK_URL,
        log_path=config.LOGS_DIR / f"{args.vm_id}.log",
    )

    orchestrator = Orchestrator(
        vm_id=args.vm_id,
        api=api,
        notify=notify,
        num_workers=args.workers,
        test_mode=args.test,
        phase=args.phase,
    )
    orchestrator.run()


def cmd_status(args):
    """Show progress across all VMs."""
    assignment_path = config.INPUT_DIR / "assignment.json"
    if not assignment_path.exists():
        print("No assignment found. Run 'assign' first.")
        return

    import json
    assignments = json.loads(assignment_path.read_text())
    all_creators = []
    for cids in assignments.values():
        all_creators.extend(cids)

    summary = progress.get_progress_summary(all_creators)
    total = summary["total"]

    print(f"\nOverall Progress: {summary['done'] + summary['skipped']}/{total} complete")
    print(f"  Done: {summary['done']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Pending: {summary['pending']}")
    print(f"  Segments: {summary['total_segments']:,}")
    print(f"  Positives: {summary['total_positives']} ({summary['positive_rate']})")
    print(f"  Pairs: {summary['total_pairs']}")

    # Per-VM breakdown
    print("\nPer VM:")
    for vm_id, cids in assignments.items():
        vm_summary = progress.get_progress_summary(cids)
        done = vm_summary["done"] + vm_summary["skipped"]
        print(f"  {vm_id}: {done}/{len(cids)} done, "
              f"{vm_summary['errors']} errors, "
              f"{vm_summary['total_segments']:,} segments")

    # Show errors
    if summary["errors"] > 0:
        print(f"\nErrored creators ({summary['errors']}):")
        for cid in all_creators:
            if progress.get_status(cid) == "error":
                err_path = config.PROGRESS_DIR / f"{cid}_error.json"
                if err_path.exists():
                    import json as j
                    err = j.loads(err_path.read_text())
                    print(f"  {cid}: {err.get('error', 'unknown')[:80]}")


def cmd_merge(args):
    """Merge per-creator outputs into final files."""
    from .merge import merge_all
    merge_all()


def main():
    parser = argparse.ArgumentParser(
        description="ClipWhy V2 Data Collection Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # assign
    assign_p = subparsers.add_parser("assign", help="Generate VM assignment")
    assign_p.add_argument("--creators-csv", required=True,
                          help="Path to V2 creators.csv")
    assign_p.add_argument("--num-vms", type=int, default=4)

    # run
    run_p = subparsers.add_parser("run", help="Run pipeline on this VM")
    run_p.add_argument("--vm-id", required=True,
                       choices=[f"vm{i}" for i in range(8)] + [f"gpu{i}" for i in range(8)] + [f"cpu{i}" for i in range(4)])
    run_p.add_argument("--workers", type=int, default=config.WORKERS_PER_VM)
    run_p.add_argument("--phase", default="all", choices=["all", "cpu", "gpu"],
                       help="Run phase: cpu (steps 1-4, no GPU), "
                            "gpu (steps 5-7, needs GPU), all (default)")
    run_p.add_argument("--test", action="store_true",
                       help="Test mode: 2 creators only")

    # status
    subparsers.add_parser("status", help="Check progress")

    # merge
    subparsers.add_parser("merge", help="Merge results")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    {
        "assign": cmd_assign,
        "run": cmd_run,
        "status": cmd_status,
        "merge": cmd_merge,
    }[args.command](args)


if __name__ == "__main__":
    main()
