#!/usr/bin/env python
"""
Real-time Training Monitor Entrypoint for SpatialTranscriptFormer.
"""
import argparse
import logging
from spatial_transcript_former.dashboard.app import init_app, app


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Training Monitor")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to the experiment run directory containing training_logs.sqlite",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Path to a directory containing MULTIPLE experiment run directories for comparison",
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to host the dashboard on"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5000,
        help="Log polling interval in milliseconds",
    )
    args = parser.parse_args()
    if not args.run_dir and not args.runs_dir:
        parser.error("Must provide either --run-dir or --runs-dir")
    return args


if __name__ == "__main__":
    args = parse_args()

    # Initialize the dash app
    init_app(args)

    if getattr(args, "runs_dir", None):
        print(f"Tracking multiple runs in: {args.runs_dir}")
    else:
        print(f"Tracking single run at: {args.run_dir}")

    print(f"Starting dashboard on http://127.0.0.1:{args.port}/")

    # Run the server
    # Turn off debug to prevent double-reloading the data parser during polling
    app.run(debug=False, port=args.port)
