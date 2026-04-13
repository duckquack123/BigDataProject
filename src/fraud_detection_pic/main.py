from __future__ import annotations

from typing import Any

from .config import PipelineConfig, SparkConfig
from .data_generation import FraudGraphGenerator
from .heat_kernel import apply_heat_kernel_annealing
from .pipeline import (
    build_cluster_metrics,
    cleanup_dataframes,
    create_edge_dataframe,
    evaluate_predictions,
    run_pic,
    score_clusters,
    summarize_pipeline,
)
from .spark_utils import build_spark_session
from .visualization import generate_visual_artifacts


def run_pipeline(
    spark_cfg: SparkConfig | None = None,
    pipeline_cfg: PipelineConfig | None = None,
    visualize: bool = False,
    output_dir: str = "outputs",
) -> dict[str, Any]:
    spark_cfg = spark_cfg or SparkConfig()
    pipeline_cfg = pipeline_cfg or PipelineConfig()

    spark = build_spark_session(spark_cfg)

    generator = FraudGraphGenerator(
        n_normal_nodes=pipeline_cfg.n_normal_nodes,
        n_normal_edges=pipeline_cfg.n_normal_edges,
        fraud_rings=pipeline_cfg.fraud_rings,
        seed=pipeline_cfg.seed,
    )
    raw_edges = generator.generate()

    edge_df, total_edges, _ = create_edge_dataframe(spark, raw_edges)

    heat_kernel_info: dict[str, Any] | None = None
    if pipeline_cfg.use_heat_kernel:
        edge_df, heat_kernel_info = apply_heat_kernel_annealing(edge_df, pipeline_cfg)
        edge_df.cache()
        edge_df.count()

    assignments_df, elapsed = run_pic(edge_df, pipeline_cfg)
    intra_cluster_edges, cluster_metrics = build_cluster_metrics(edge_df, assignments_df)
    scored_clusters = score_clusters(cluster_metrics, pipeline_cfg)

    eval_metrics = evaluate_predictions(edge_df, assignments_df, scored_clusters)
    summary = summarize_pipeline(
        assignments_df=assignments_df,
        intra_cluster_edges=intra_cluster_edges,
        scored_clusters=scored_clusters,
        metrics=eval_metrics,
        elapsed=elapsed,
        total_edges=total_edges,
    )

    if visualize:
        heat_history = heat_kernel_info.get("heat_history") if heat_kernel_info else None
        heat_graph_frames = heat_kernel_info.get("heat_graph_frames") if heat_kernel_info else None
        node_snapshot = heat_kernel_info.get("node_snapshot") if heat_kernel_info else None
        summary["artifacts"] = generate_visual_artifacts(
            scored_clusters,
            output_dir=output_dir,
            heat_history=heat_history,
            heat_graph_frames=heat_graph_frames,
            node_snapshot=node_snapshot,
        )

    if heat_kernel_info is not None:
        summary["heat_kernel"] = heat_kernel_info

    cleanup_dataframes(
        edge_df,
        assignments_df,
        intra_cluster_edges,
        cluster_metrics,
        scored_clusters,
        eval_metrics["node_results_gt"],
    )
    spark.stop()

    return summary


def format_summary(summary: dict[str, Any]) -> str:
    artifacts_text = ""
    artifacts = summary.get("artifacts")
    if artifacts:
        artifacts_lines = [f"- {key}: {value}" for key, value in artifacts.items()]
        artifacts_text = "\n\nArtifacts:\n" + "\n".join(artifacts_lines)

    heat_text = ""
    hk = summary.get("heat_kernel")
    if hk:
        heat_text = (
            "\n\nHeat Kernel:\n"
            + f"- enabled: {hk.get('enabled')}\n"
            + f"- tau_schedule: {hk.get('tau_schedule')}\n"
            + f"- faulty_node_quantile: {hk.get('faulty_node_quantile')}\n"
            + f"- faulty_edge_boost: {hk.get('faulty_edge_boost')}\n"
            + f"- risk_threshold: {hk.get('risk_threshold')}"
        )

    return (
        "\n" + "=" * 70 + "\n"
        + "FRAUD DETECTION PIPELINE SUMMARY\n"
        + "=" * 70 + "\n"
        + f"Nodes: {summary['n_nodes']:,}\n"
        + f"Edges: {summary['total_edges']:,}\n"
        + f"Graph density: {summary['graph_density']:.6f}\n"
        + f"PIC clusters: {summary['n_clusters']}\n"
        + f"Fraud clusters flagged: {summary['n_fraud_clusters']}\n"
        + f"Fraud cluster IDs: {summary['fraud_cluster_ids']}\n"
        + f"Flagged nodes: {summary['total_nodes_flagged']}\n"
        + f"Flagged transaction volume: ${summary['total_volume_flagged']:,.2f}\n"
        + f"Runtime: {summary['elapsed_seconds']:.2f}s\n"
        + "\n"
        + f"Precision: {summary['precision']:.4f}\n"
        + f"Recall: {summary['recall']:.4f}\n"
        + f"F1 score: {summary['f1']:.4f}\n"
        + f"TP/FP/FN/TN: {summary['tp']}/{summary['fp']}/{summary['fn']}/{summary['tn']}\n"
        + artifacts_text
        + heat_text
    )
