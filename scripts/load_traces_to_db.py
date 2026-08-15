"""
scripts/load_traces_to_db.py — Standalone Script to Ingest Evaluation Trace Logs into SQLite

Usage:
    python scripts/load_traces_to_db.py [--traces-dir logs/traces] [--db-path logs/traces.db] [--latest-only]
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from lib.utils.config import PipelineConfig
from lib.utils.logger import get_logger
from lib.utils.trace_db import (
    get_table_name,
    init_trace_db,
    list_db_runs_and_checkpoints,
    sync_all_traces_to_db,
)

logger = get_logger("scripts.load_traces_to_db")


def main():
    parser = argparse.ArgumentParser(description="Ingest evaluation trace CSVs into SQLite database.")
    parser.add_argument(
        "--traces-dir",
        type=str,
        default=None,
        help="Directory containing partitioned trace CSVs (default: logs/traces from config)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database file (default: logs/traces.db from config)",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Ingest only the newest CSV trace file for each probe category",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe and recreate SQLite database tables before ingesting (removes stale run IDs)",
    )

    args = parser.parse_args()
    cfg = PipelineConfig()

    traces_dir = Path(args.traces_dir) if args.traces_dir else cfg.logging.log_dir / "traces"
    db_path = Path(args.db_path) if args.db_path else cfg.logging.trace_db_path

    if args.clean and db_path.exists():
        import os
        try:
            os.remove(db_path)
            print(f"Clean mode: Removed existing database at {db_path}")
        except Exception as e:
            print(f"Warning: Could not remove DB file ({e}); reinitializing schema.")

    print("=" * 60)
    print("  Evaluation Trace SQLite Ingestion Utility")
    print("=" * 60)
    print(f"Traces Directory: {traces_dir.resolve()}")
    print(f"SQLite Database:  {db_path.resolve()}")
    print(f"Mode:             {'Latest Only' if args.latest_only else 'All CSV Files'}")
    print("-" * 60)

    if not traces_dir.exists():
        print(f"Warning: Traces directory '{traces_dir}' does not exist.")
        sys.exit(0)

    init_trace_db(db_path)
    counts = sync_all_traces_to_db(traces_dir, db_path, latest_only=args.latest_only)

    print("\nIngestion Summary:")
    total = 0
    for probe, count in counts.items():
        table = get_table_name(probe)
        runs = list_db_runs_and_checkpoints(probe, db_path)
        print(f"  • {probe.upper()} ({table}): {count:,} records processed across {len(runs)} runs/checkpoints")
        total += count

    print(f"\nTotal records processed: {total:,}")
    print(f"Database successfully updated at: {db_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
