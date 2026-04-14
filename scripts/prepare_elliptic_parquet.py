#!/usr/bin/env python3
"""Prepare Elliptic Bitcoin CSV files into a parquet edge table.

Output schema:
- src_id (string)
- dst_id (string)
- amount (double)
- timestamp (int)
- true_label (string)
"""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession, functions as F


def build_spark(master: str, app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Elliptic CSV files to parquet edge table")
    parser.add_argument("--edges-path", required=True, help="Path to elliptic_txs_edgelist.csv")
    parser.add_argument("--classes-path", required=True, help="Path to elliptic_txs_classes.csv")
    parser.add_argument("--features-path", required=True, help="Path to elliptic_txs_features.csv")
    parser.add_argument("--output-path", required=True, help="Output parquet path")
    parser.add_argument("--master", default="local[*]", help="Spark master URL (local[*], yarn, etc.)")
    parser.add_argument(
        "--timestamp-mode",
        choices=["src", "dst", "max", "min"],
        default="max",
        help="How to derive edge timestamp from endpoint transaction timestamps.",
    )
    parser.add_argument(
        "--default-amount",
        type=float,
        default=1.0,
        help="Default amount value because raw Elliptic edgelist has no amount column.",
    )
    args = parser.parse_args()

    spark = build_spark(master=args.master, app_name="Prepare_Elliptic_Parquet")

    edges = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .csv(args.edges_path)
        .select(
            F.trim(F.col("txId1").cast("string")).alias("src_id"),
            F.trim(F.col("txId2").cast("string")).alias("dst_id"),
        )
        .filter(F.col("src_id").isNotNull() & F.col("dst_id").isNotNull())
        .filter((F.length(F.col("src_id")) > 0) & (F.length(F.col("dst_id")) > 0))
    )

    classes = (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .csv(args.classes_path)
        .select(
            F.trim(F.col("txId").cast("string")).alias("tx_id"),
            F.trim(F.col("class").cast("string")).alias("class"),
        )
    )

    # features file has no header; _c0 is tx id and _c1 is time step.
    features = (
        spark.read.option("header", "false")
        .option("inferSchema", "false")
        .csv(args.features_path)
        .select(
            F.trim(F.col("_c0").cast("string")).alias("tx_id"),
            F.col("_c1").cast("int").alias("timestamp"),
        )
    )

    src_time = features.select(F.col("tx_id").alias("src_id"), F.col("timestamp").alias("src_timestamp"))
    dst_time = features.select(F.col("tx_id").alias("dst_id"), F.col("timestamp").alias("dst_timestamp"))
    src_class = classes.select(F.col("tx_id").alias("src_id"), F.col("class").alias("src_class"))
    dst_class = classes.select(F.col("tx_id").alias("dst_id"), F.col("class").alias("dst_class"))

    joined = (
        edges.join(src_time, on="src_id", how="left")
        .join(dst_time, on="dst_id", how="left")
        .join(src_class, on="src_id", how="left")
        .join(dst_class, on="dst_id", how="left")
    )

    if args.timestamp_mode == "src":
        ts_col = F.col("src_timestamp")
    elif args.timestamp_mode == "dst":
        ts_col = F.col("dst_timestamp")
    elif args.timestamp_mode == "min":
        ts_col = F.least(F.col("src_timestamp"), F.col("dst_timestamp"))
    else:
        ts_col = F.greatest(F.col("src_timestamp"), F.col("dst_timestamp"))

    output = (
        joined.withColumn("amount", F.lit(float(args.default_amount)).cast("double"))
        .withColumn("timestamp", ts_col.cast("int"))
        .withColumn(
            "true_label",
            F.when((F.col("src_class") == F.lit("1")) | (F.col("dst_class") == F.lit("1")), F.lit("fraud"))
            .when((F.col("src_class") == F.lit("2")) & (F.col("dst_class") == F.lit("2")), F.lit("normal"))
            .otherwise(F.lit("unknown")),
        )
        .select("src_id", "dst_id", "amount", "timestamp", "true_label")
    )

    output.write.mode("overwrite").parquet(args.output_path)

    stats = output.agg(
        F.count("*").alias("edge_count"),
        F.approx_count_distinct("src_id").alias("distinct_src"),
        F.approx_count_distinct("dst_id").alias("distinct_dst"),
        F.min("timestamp").alias("min_timestamp"),
        F.max("timestamp").alias("max_timestamp"),
    ).collect()[0]

    print("Wrote parquet dataset:", args.output_path)
    print(
        "edge_count={edge_count}, distinct_src={distinct_src}, distinct_dst={distinct_dst}, "
        "timestamp_range=({min_timestamp}, {max_timestamp})".format(**stats.asDict())
    )

    spark.stop()


if __name__ == "__main__":
    main()
