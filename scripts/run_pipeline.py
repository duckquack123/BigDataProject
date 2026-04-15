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
from fraud_detection_pic.config import PipelineConfig, SparkConfig


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
        "--input-path",
        default=None,
        help="Path to a real edge dataset in CSV, Parquet, or JSON format.",
    )
    parser.add_argument(
        "--input-format",
        default="parquet",
        help="Input format for --input-path, such as parquet, csv, or json.",
    )
    parser.add_argument(
        "--source-col",
        default="src",
        help="Column name for the source node id.",
    )
    parser.add_argument(
        "--destination-col",
        default="dst",
        help="Column name for the destination node id.",
    )
    parser.add_argument(
        "--weight-col",
        default="weight",
        help="Column name for edge weight. Use an empty string if the dataset is unweighted.",
    )
    parser.add_argument(
        "--label-col",
        default="true_label",
        help="Optional label column for evaluation. Use an empty string if labels are unavailable.",
    )
    parser.add_argument(
        "--time-col",
        default="",
        help="Optional time column used for range filtering (for example timestamp).",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=None,
        help="Optional minimum value for --time-col.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=None,
        help="Optional maximum value for --time-col.",
    )
    parser.add_argument(
        "--input-has-header",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat CSV input as having a header row.",
    )
    parser.add_argument(
        "--input-delimiter",
        default=",",
        help="Delimiter used for CSV input.",
    )
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL, for example local[*] or yarn.",
    )
    parser.add_argument(
        "--driver-memory",
        default="2g",
        help="Spark driver memory setting.",
    )
    parser.add_argument(
        "--executor-memory",
        default="4g",
        help="Spark executor memory setting.",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=200,
        help="Spark SQL shuffle partition count.",
    )
    parser.add_argument(
        "--default-parallelism",
        type=int,
        default=200,
        help="Spark default parallelism.",
    )
    parser.add_argument(
        "--pic-k",
        type=int,
        default=200,
        help="Number of PIC clusters.",
    )
    parser.add_argument(
        "--fraud-score-threshold",
        type=float,
        default=0.35,
        help="Fraud score cutoff used to flag clusters.",
    )
    parser.add_argument(
        "--micro-cluster-max-size",
        type=int,
        default=2000,
        help="Maximum cluster size to be considered a fraud ring.",
    )
    parser.add_argument(
        "--min-internal-density",
        type=float,
        default=0.01,
        help="Minimum internal density required to flag a cluster.",
    )
    parser.add_argument(
        "--top-fraud-clusters",
        type=int,
        default=5,
        help="Always flag the top K clusters by fraud score as a fallback.",
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

    spark_cfg = SparkConfig(
        master=args.master,
        driver_memory=args.driver_memory,
        executor_memory=args.executor_memory,
        shuffle_partitions=str(args.shuffle_partitions),
        default_parallelism=str(args.default_parallelism),
    )
    pipeline_cfg = replace(
        PipelineConfig(),
        pic_k=args.pic_k,
        use_heat_kernel=args.heat_kernel,
        heat_tau_start=args.tau_start,
        heat_tau_end=args.tau_end,
        heat_steps=max(args.cooling_steps, 1),
        faulty_node_quantile=min(max(args.faulty_node_quantile, 0.0), 1.0),
        faulty_edge_boost=max(args.faulty_edge_boost, 0.0),
        fraud_score_threshold=args.fraud_score_threshold,
        micro_cluster_max_size=args.micro_cluster_max_size,
        min_internal_density=max(args.min_internal_density, 0.0),
        top_fraud_clusters=max(args.top_fraud_clusters, 0),
    )

    summary = run_pipeline(
        spark_cfg=spark_cfg,
        pipeline_cfg=pipeline_cfg,
        input_path=args.input_path,
        input_format=args.input_format,
        source_col=args.source_col,
        destination_col=args.destination_col,
        weight_col=args.weight_col or None,
        label_col=args.label_col or None,
        time_col=args.time_col or None,
        time_min=args.time_min,
        time_max=args.time_max,
        input_has_header=args.input_has_header,
        input_delimiter=args.input_delimiter,
        visualize=args.visualize,
        output_dir=args.output_dir,
    )
    print(format_summary(summary))
