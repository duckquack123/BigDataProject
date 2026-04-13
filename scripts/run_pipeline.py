#!/usr/bin/env python3
"""CLI entrypoint for fraud detection pipeline."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fraud_detection_pic.main import format_summary, run_pipeline
from fraud_detection_pic.config import PipelineConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fraud detection PIC pipeline")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization artifacts (CSV and PNG charts when matplotlib is available).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store generated artifacts.",
    )
    parser.add_argument(
        "--heat-kernel",
        action="store_true",
        help="Enable heat-kernel annealing on Laplacian edge affinities.",
    )
    parser.add_argument(
        "--tau-start",
        type=float,
        default=1.2,
        help="Starting temperature for heat-kernel annealing.",
    )
    parser.add_argument(
        "--tau-end",
        type=float,
        default=0.2,
        help="Final temperature for heat-kernel annealing.",
    )
    parser.add_argument(
        "--cooling-steps",
        type=int,
        default=6,
        help="Number of cooling steps between tau-start and tau-end.",
    )
    parser.add_argument(
        "--faulty-node-quantile",
        type=float,
        default=0.95,
        help="Quantile cutoff to mark suspicious/faulty nodes.",
    )
    parser.add_argument(
        "--faulty-edge-boost",
        type=float,
        default=0.35,
        help="Boost factor for edges attached to suspicious/faulty nodes.",
    )
    args = parser.parse_args()

    pipeline_cfg = replace(
        PipelineConfig(),
        use_heat_kernel=args.heat_kernel,
        heat_tau_start=args.tau_start,
        heat_tau_end=args.tau_end,
        heat_steps=max(args.cooling_steps, 1),
        faulty_node_quantile=min(max(args.faulty_node_quantile, 0.0), 1.0),
        faulty_edge_boost=max(args.faulty_edge_boost, 0.0),
    )

    summary = run_pipeline(
        pipeline_cfg=pipeline_cfg,
        visualize=args.visualize,
        output_dir=args.output_dir,
    )
    print(format_summary(summary))
