from __future__ import annotations
from typing import Any

from pyspark.sql import DataFrame, functions as F

from .config import PipelineConfig


def _build_tau_schedule(tau_start: float, tau_end: float, steps: int) -> list[float]:
    if steps <= 1:
        return [max(tau_end, 1e-6)]
    step_values = []
    for i in range(steps):
        alpha = i / (steps - 1)
        tau = tau_start + (tau_end - tau_start) * alpha
        step_values.append(max(tau, 1e-6))
    return step_values


def apply_heat_kernel_annealing(edge_df: DataFrame, cfg: PipelineConfig) -> tuple[DataFrame, dict[str, Any]]:
    """Apply heat-kernel cooling on edge affinities and boost edges touching suspicious nodes."""
    tau_schedule = _build_tau_schedule(cfg.heat_tau_start, cfg.heat_tau_end, cfg.heat_steps)

    src_nodes = edge_df.select(
        F.col("src").alias("node_id"),
        F.col("weight").alias("weight"),
    )
    dst_nodes = edge_df.select(
        F.col("dst").alias("node_id"),
        F.col("weight").alias("weight"),
    )
    node_obs = src_nodes.unionByName(dst_nodes)

    node_stats = node_obs.groupBy("node_id").agg(
        F.count("*").alias("degree"),
        F.avg("weight").alias("avg_weight"),
    )

    stats = node_stats.agg(
        F.avg("degree").alias("degree_mean"),
        F.stddev("degree").alias("degree_std"),
        F.avg("avg_weight").alias("weight_mean"),
        F.stddev("avg_weight").alias("weight_std"),
    ).collect()[0]

    degree_mean = float(stats["degree_mean"] or 0.0)
    degree_std = float(stats["degree_std"] or 1.0)
    weight_mean = float(stats["weight_mean"] or 0.0)
    weight_std = float(stats["weight_std"] or 1.0)

    degree_std = degree_std if degree_std > 1e-9 else 1.0
    weight_std = weight_std if weight_std > 1e-9 else 1.0

    # Suspiciousness prior combines unusual connectivity and high spending behavior.
    node_stats = (
        node_stats.withColumn("degree_z", (F.col("degree") - F.lit(degree_mean)) / F.lit(degree_std))
        .withColumn("weight_z", (F.col("avg_weight") - F.lit(weight_mean)) / F.lit(weight_std))
        .withColumn("risk_raw", F.lit(0.5) * F.col("degree_z") + F.lit(0.5) * F.col("weight_z"))
    )

    node_stats = node_stats.withColumn(
        "node_risk",
        F.lit(1.0) / (F.lit(1.0) + F.exp(-F.col("risk_raw"))),
    )

    risk_threshold = node_stats.approxQuantile("node_risk", [cfg.faulty_node_quantile], 0.01)[0]
    node_stats = node_stats.withColumn(
        "is_faulty_node",
        (F.col("node_risk") >= F.lit(risk_threshold)).cast("int"),
    )

    node_snapshot_rows = node_stats.select("node_id", "node_risk", "is_faulty_node").collect()
    node_snapshot = [
        {
            "node_id": int(r["node_id"]),
            "node_risk": float(r["node_risk"]),
            "is_faulty_node": int(r["is_faulty_node"]),
        }
        for r in node_snapshot_rows
    ]

    src_meta = node_stats.select(
        F.col("node_id").alias("src"),
        F.col("node_risk").alias("src_risk"),
        F.col("is_faulty_node").alias("src_faulty"),
    )
    dst_meta = node_stats.select(
        F.col("node_id").alias("dst"),
        F.col("node_risk").alias("dst_risk"),
        F.col("is_faulty_node").alias("dst_faulty"),
    )

    working = (
        edge_df.join(src_meta, on="src", how="left")
        .join(dst_meta, on="dst", how="left")
        .fillna(0.0, subset=["src_risk", "dst_risk"])
        .fillna(0, subset=["src_faulty", "dst_faulty"])
        .withColumn("weight_hk", F.col("weight_norm"))
    )

    total_edges = working.count()
    heat_history: list[dict[str, float | int]] = []
    heat_graph_frames: list[dict[str, Any]] = []
    area_threshold = 0.7

    for iter_idx, tau in enumerate(tau_schedule, start=1):
        working = (
            working.withColumn(
                "risk_dist",
                F.abs(F.col("src_risk") - F.col("dst_risk")) / F.lit(max(cfg.heat_distance_scale, 1e-6)),
            )
            .withColumn(
                "heat_kernel",
                F.exp(-F.pow(F.col("risk_dist"), F.lit(2.0)) / F.lit(tau)),
            )
            .withColumn(
                "faulty_edge",
                F.greatest(F.col("src_faulty"), F.col("dst_faulty")),
            )
            .withColumn(
                "weight_hk",
                F.col("weight_hk")
                * F.col("heat_kernel")
                * (F.lit(1.0) + F.lit(cfg.faulty_edge_boost) * F.col("faulty_edge")),
            )
        )

        hk_max = float(working.agg(F.max("weight_hk").alias("hk_max")).collect()[0]["hk_max"] or 1.0)
        hk_max = hk_max if hk_max > 1e-9 else 1.0
        working = working.withColumn("weight_hk", F.col("weight_hk") / F.lit(hk_max))

        iter_stats = working.agg(
            F.sum("weight_hk").alias("heat_mass"),
            F.avg("weight_hk").alias("heat_mean"),
            F.sum(F.when(F.col("weight_hk") >= F.lit(area_threshold), F.lit(1)).otherwise(F.lit(0))).alias("hot_edges"),
            F.sum(F.when(F.col("weight_hk") >= F.lit(area_threshold), F.col("weight_hk")).otherwise(F.lit(0.0))).alias("hot_area_mass"),
        ).collect()[0]

        hot_edges = int(iter_stats["hot_edges"] or 0)
        heat_history.append(
            {
                "iteration": iter_idx,
                "tau": round(float(tau), 6),
                "heat_mass": round(float(iter_stats["heat_mass"] or 0.0), 6),
                "heat_mean": round(float(iter_stats["heat_mean"] or 0.0), 6),
                "hot_edges": hot_edges,
                "hot_edge_ratio": round(hot_edges / total_edges, 6) if total_edges > 0 else 0.0,
                "hot_area_mass": round(float(iter_stats["hot_area_mass"] or 0.0), 6),
            }
        )

        frame_rows = (
            working.select(
                "src",
                "dst",
                "weight_hk",
                "src_risk",
                "dst_risk",
                "src_faulty",
                "dst_faulty",
            )
            .filter(F.col("weight_hk") >= F.lit(cfg.heat_graph_min_weight))
            .orderBy(F.desc("weight_hk"))
            .limit(cfg.heat_graph_max_edges)
            .collect()
        )

        frame_edges = [
            {
                "src": int(r["src"]),
                "dst": int(r["dst"]),
                "weight": float(r["weight_hk"]),
                "src_risk": float(r["src_risk"]),
                "dst_risk": float(r["dst_risk"]),
                "src_faulty": int(r["src_faulty"]),
                "dst_faulty": int(r["dst_faulty"]),
            }
            for r in frame_rows
        ]

        heat_graph_frames.append(
            {
                "iteration": iter_idx,
                "tau": round(float(tau), 6),
                "edges": frame_edges,
            }
        )

    output_df = (
        working.withColumn("weight_norm", F.round(F.col("weight_hk"), 6))
        .drop(
            "src_risk",
            "dst_risk",
            "src_faulty",
            "dst_faulty",
            "risk_dist",
            "heat_kernel",
            "faulty_edge",
            "weight_hk",
        )
    )

    metadata = {
        "enabled": True,
        "tau_schedule": tau_schedule,
        "faulty_node_quantile": cfg.faulty_node_quantile,
        "faulty_edge_boost": cfg.faulty_edge_boost,
        "risk_threshold": round(float(risk_threshold), 6),
        "hot_area_threshold": area_threshold,
        "heat_history": heat_history,
        "node_snapshot": node_snapshot,
        "heat_graph_frames": heat_graph_frames,
    }
    return output_df, metadata
