from __future__ import annotations

import math
import time
from typing import Any

from pyspark.ml.clustering import PowerIterationClustering
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from .config import PipelineConfig


def create_edge_dataframe(
    spark: SparkSession, raw_edges: list[tuple[int, int, float, str]]
) -> tuple[DataFrame, int, float]:
    edge_schema = StructType(
        [
            StructField("src", LongType(), nullable=False),
            StructField("dst", LongType(), nullable=False),
            StructField("weight", DoubleType(), nullable=False),
            StructField("true_label", StringType(), nullable=True),
        ]
    )

    edge_df = spark.createDataFrame(raw_edges, schema=edge_schema)

    edge_df = (
        edge_df.withColumn("src_norm", F.least(F.col("src"), F.col("dst")))
        .withColumn("dst_norm", F.greatest(F.col("src"), F.col("dst")))
        .groupBy("src_norm", "dst_norm", "true_label")
        .agg(F.max("weight").alias("weight"))
        .withColumnRenamed("src_norm", "src")
        .withColumnRenamed("dst_norm", "dst")
        .filter(F.col("src") != F.col("dst"))
        .select("src", "dst", "weight", "true_label")
    )

    w_max = edge_df.agg(F.max("weight")).collect()[0][0]
    edge_df = edge_df.withColumn("weight_norm", F.round(F.col("weight") / F.lit(w_max), 6))

    edge_df.cache()
    total_edges = edge_df.count()
    return edge_df, total_edges, float(w_max)


def create_edge_dataframe_from_df(
    edge_df: DataFrame,
    source_col: str = "src",
    destination_col: str = "dst",
    weight_col: str | None = "weight",
    label_col: str | None = "true_label",
) -> tuple[DataFrame, int, float]:
    selected = edge_df
    columns = set(selected.columns)

    if source_col not in columns:
        raise ValueError(f"Missing required source column: {source_col}")
    if destination_col not in columns:
        raise ValueError(f"Missing required destination column: {destination_col}")

    selected = selected.select(
        F.trim(F.col(source_col).cast(StringType())).alias("src_key"),
        F.trim(F.col(destination_col).cast(StringType())).alias("dst_key"),
        (
            F.col(weight_col).cast(DoubleType()).alias("weight")
            if weight_col and weight_col in columns
            else F.lit(1.0).cast(DoubleType()).alias("weight")
        ),
        (
            F.col(label_col).cast(StringType()).alias("true_label")
            if label_col and label_col in columns
            else F.lit("unknown").cast(StringType()).alias("true_label")
        ),
    ).filter(
        F.col("src_key").isNotNull()
        & F.col("dst_key").isNotNull()
        & (F.length(F.col("src_key")) > 0)
        & (F.length(F.col("dst_key")) > 0)
    )

    selected = selected.withColumn(
        "weight",
        F.when(F.col("weight").isNull() | (F.col("weight") <= 0.0), F.lit(1.0)).otherwise(F.col("weight")),
    )

    node_index = (
        selected.select(F.col("src_key").alias("node_key"))
        .union(selected.select(F.col("dst_key").alias("node_key")))
        .distinct()
        .withColumn("node_id", F.monotonically_increasing_id().cast(LongType()))
        .cache()
    )
    node_index.count()

    indexed = (
        selected.join(
            node_index.withColumnRenamed("node_key", "src_key_map").withColumnRenamed("node_id", "src"),
            on=selected["src_key"] == F.col("src_key_map"),
            how="inner",
        )
        .join(
            node_index.withColumnRenamed("node_key", "dst_key_map").withColumnRenamed("node_id", "dst"),
            on=selected["dst_key"] == F.col("dst_key_map"),
            how="inner",
        )
        .drop("src_key", "dst_key", "src_key_map", "dst_key_map")
        .select("src", "dst", "weight", "true_label")
    )

    normalized = (
        indexed.withColumn("src_norm", F.least(F.col("src"), F.col("dst")))
        .withColumn("dst_norm", F.greatest(F.col("src"), F.col("dst")))
        .groupBy("src_norm", "dst_norm", "true_label")
        .agg(F.max("weight").alias("weight"))
        .withColumnRenamed("src_norm", "src")
        .withColumnRenamed("dst_norm", "dst")
        .filter(F.col("src") != F.col("dst"))
        .select("src", "dst", "weight", "true_label")
    )

    w_max = normalized.agg(F.max("weight")).collect()[0][0] or 1.0
    w_max = float(w_max) if float(w_max) > 0 else 1.0
    normalized = normalized.withColumn("weight_norm", F.round(F.col("weight") / F.lit(w_max), 6))

    node_index.unpersist()
    normalized.cache()
    total_edges = normalized.count()
    return normalized, total_edges, w_max


def run_pic(edge_df: DataFrame, cfg: PipelineConfig) -> tuple[DataFrame, float]:
    pic = PowerIterationClustering(
        k=cfg.pic_k,
        maxIter=cfg.pic_max_iter,
        weightCol="weight_norm",
        srcCol="src",
        dstCol="dst",
        initMode=cfg.pic_init_mode,
    )

    start = time.time()
    assignments_df = pic.assignClusters(edge_df)
    assignments_df.cache()
    assignments_df.count()
    elapsed = time.time() - start
    return assignments_df, elapsed


def build_cluster_metrics(edge_df: DataFrame, assignments_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    src_cluster = assignments_df.select(
        F.col("id").alias("src"), F.col("cluster").alias("src_cluster")
    )
    dst_cluster = assignments_df.select(
        F.col("id").alias("dst"), F.col("cluster").alias("dst_cluster")
    )

    enriched_edges = (
        edge_df.join(src_cluster, on="src", how="left")
        .join(dst_cluster, on="dst", how="left")
        .withColumn("same_cluster", F.col("src_cluster") == F.col("dst_cluster"))
    )

    intra_cluster_edges = enriched_edges.filter(F.col("same_cluster") == F.lit(True))
    intra_cluster_edges.cache()
    intra_cluster_edges.count()

    node_counts = assignments_df.groupBy("cluster").agg(F.count("*").alias("n_nodes"))

    edge_stats = (
        intra_cluster_edges.groupBy("src_cluster")
        .agg(
            F.count("*").alias("n_edges"),
            F.sum("weight").alias("total_volume"),
            F.avg("weight").alias("avg_weight"),
            F.stddev("weight").alias("std_weight"),
            F.min("weight").alias("min_weight"),
            F.max("weight").alias("max_weight"),
        )
        .withColumnRenamed("src_cluster", "cluster")
    )

    cluster_metrics = (
        node_counts.join(edge_stats, on="cluster", how="left")
        .fillna(
            0.0,
            subset=["n_edges", "total_volume", "avg_weight", "std_weight", "min_weight", "max_weight"],
        )
        .withColumn(
            "max_possible_edges",
            (F.col("n_nodes") * (F.col("n_nodes") - 1) / 2).cast(DoubleType()),
        )
        .withColumn(
            "internal_density",
            F.when(F.col("max_possible_edges") > 0, F.col("n_edges") / F.col("max_possible_edges")).otherwise(0.0),
        )
        .withColumn(
            "coeff_variation",
            F.when(F.col("avg_weight") > 0, F.col("std_weight") / F.col("avg_weight")).otherwise(999.0),
        )
        .withColumn(
            "volume_per_node",
            F.when(F.col("n_nodes") > 0, F.col("total_volume") / F.col("n_nodes")).otherwise(0.0),
        )
    )

    vpn_max = cluster_metrics.agg(F.max("volume_per_node")).collect()[0][0]
    cluster_metrics = cluster_metrics.withColumn(
        "volume_per_node_norm",
        F.when(F.lit(vpn_max) > 0, F.col("volume_per_node") / F.lit(vpn_max)).otherwise(0.0),
    )

    cluster_metrics.cache()
    cluster_metrics.count()
    return intra_cluster_edges, cluster_metrics


def score_clusters(cluster_metrics: DataFrame, cfg: PipelineConfig) -> DataFrame:
    scored_clusters = (
        cluster_metrics.withColumn(
            "uniformity_score", F.lit(1.0) / (F.lit(1.0) + F.col("coeff_variation"))
        )
        .withColumn(
            "fraud_score",
            F.round(
                F.lit(cfg.density_weight) * F.col("internal_density")
                + F.lit(cfg.uniformity_weight) * F.col("uniformity_score")
                + F.lit(cfg.volume_weight) * F.col("volume_per_node_norm"),
                6,
            ),
        )
        .withColumn(
            "is_fraud_ring",
            (F.col("fraud_score") >= F.lit(cfg.fraud_score_threshold))
            & (F.col("n_nodes") <= F.lit(cfg.micro_cluster_max_size))
            & (F.col("internal_density") >= F.lit(cfg.min_internal_density)),
        )
        .orderBy(F.desc("fraud_score"))
    )

    scored_clusters.cache()
    scored_clusters.count()
    return scored_clusters


def evaluate_predictions(
    edge_df: DataFrame, assignments_df: DataFrame, scored_clusters: DataFrame
) -> dict[str, Any]:
    flagged_clusters_df = scored_clusters.select("cluster", "fraud_score", "is_fraud_ring")
    node_results = (
        assignments_df.join(flagged_clusters_df, on="cluster", how="left")
        .withColumnRenamed("id", "node_id")
        .withColumn("is_fraud_ring", F.coalesce(F.col("is_fraud_ring"), F.lit(False)))
    )

    src_labels = edge_df.select(F.col("src").alias("node_id"), F.col("true_label"))
    dst_labels = edge_df.select(F.col("dst").alias("node_id"), F.col("true_label"))
    node_labels = (
        src_labels.union(dst_labels)
        .filter(~F.col("true_label").isin("normal", "unknown"))
        .distinct()
    )

    node_results_gt = (
        node_results.join(node_labels, on="node_id", how="left")
        .withColumn("true_fraud", F.col("true_label").isNotNull() & (F.col("true_label") != "normal"))
    )
    node_results_gt.cache()
    node_results_gt.count()

    tp = node_results_gt.filter(F.col("is_fraud_ring") & F.col("true_fraud")).count()
    fp = node_results_gt.filter(F.col("is_fraud_ring") & ~F.col("true_fraud")).count()
    fn = node_results_gt.filter(~F.col("is_fraud_ring") & F.col("true_fraud")).count()
    tn = node_results_gt.filter(~F.col("is_fraud_ring") & ~F.col("true_fraud")).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    flagged_cluster_ids = [row["cluster"] for row in scored_clusters.filter(F.col("is_fraud_ring")).select("cluster").collect()]

    return {
        "node_results_gt": node_results_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_cluster_ids": flagged_cluster_ids,
    }


def summarize_pipeline(
    assignments_df: DataFrame,
    intra_cluster_edges: DataFrame,
    scored_clusters: DataFrame,
    metrics: dict[str, Any],
    elapsed: float,
    total_edges: int,
) -> dict[str, Any]:
    n_nodes = assignments_df.select(F.col("id").alias("node_id")).distinct().count()
    n_clusters = assignments_df.select("cluster").distinct().count()
    n_fraud_clusters = scored_clusters.filter(F.col("is_fraud_ring")).count()
    total_nodes_flagged = metrics["node_results_gt"].filter(F.col("is_fraud_ring")).count()

    fraud_cluster_ids = metrics["flagged_cluster_ids"]
    total_vol_flagged = (
        intra_cluster_edges.join(
            scored_clusters.filter(F.col("is_fraud_ring")).select(F.col("cluster").alias("src_cluster")),
            on="src_cluster",
            how="inner",
        )
        .agg(F.sum("weight").alias("flagged_volume"))
        .collect()[0][0]
        or 0.0
    )

    density = 2 * total_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

    return {
        "n_nodes": n_nodes,
        "total_edges": total_edges,
        "graph_density": density,
        "n_clusters": n_clusters,
        "n_fraud_clusters": n_fraud_clusters,
        "fraud_cluster_ids": fraud_cluster_ids,
        "total_nodes_flagged": total_nodes_flagged,
        "total_volume_flagged": float(total_vol_flagged),
        "elapsed_seconds": round(elapsed, 2),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def cleanup_dataframes(*dfs: DataFrame) -> None:
    for df in dfs:
        try:
            df.unpersist()
        except Exception:
            pass
