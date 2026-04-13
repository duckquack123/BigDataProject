from __future__ import annotations

from pathlib import Path
from typing import Any


def _safe_matplotlib_import() -> tuple[Any, Any] | tuple[None, None]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        return plt, pd
    except Exception:
        return None, None


def _safe_networkx_import() -> Any | None:
    try:
        import networkx as nx

        return nx
    except Exception:
        return None


def generate_visual_artifacts(
    scored_clusters,
    output_dir: str = "outputs",
    heat_history: list[dict[str, Any]] | None = None,
    heat_graph_frames: list[dict[str, Any]] | None = None,
    node_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Save CSV and optional PNG charts for cluster-level fraud results."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = scored_clusters.select(
        "cluster",
        "n_nodes",
        "n_edges",
        "internal_density",
        "coeff_variation",
        "fraud_score",
        "is_fraud_ring",
        "total_volume",
    ).toPandas()

    csv_path = out_dir / "cluster_scores.csv"
    pdf.to_csv(csv_path, index=False)

    artifacts = {"cluster_scores_csv": str(csv_path)}

    plt, pd = _safe_matplotlib_import()
    if plt is None or pd is None:
        return artifacts

    pdf = pdf.sort_values("fraud_score", ascending=False).reset_index(drop=True)
    pdf["cluster_label"] = pdf["cluster"].astype(str)

    score_fig = out_dir / "fraud_scores_by_cluster.png"
    colors = ["#c0392b" if flag else "#7f8c8d" for flag in pdf["is_fraud_ring"]]
    plt.figure(figsize=(12, 6))
    plt.bar(pdf["cluster_label"], pdf["fraud_score"], color=colors)
    plt.axhline(y=0.55, linestyle="--", color="#2c3e50", linewidth=1.2)
    plt.title("Fraud Score by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Fraud Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(score_fig, dpi=140)
    plt.close()
    artifacts["fraud_score_chart"] = str(score_fig)

    scatter_fig = out_dir / "density_vs_cv.png"
    plt.figure(figsize=(8, 6))
    mask = pdf["is_fraud_ring"] == True
    plt.scatter(pdf.loc[~mask, "coeff_variation"], pdf.loc[~mask, "internal_density"], c="#95a5a6", label="Normal")
    plt.scatter(pdf.loc[mask, "coeff_variation"], pdf.loc[mask, "internal_density"], c="#e74c3c", label="Flagged")
    plt.title("Internal Density vs Coefficient of Variation")
    plt.xlabel("Coefficient of Variation (CV)")
    plt.ylabel("Internal Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_fig, dpi=140)
    plt.close()
    artifacts["density_cv_chart"] = str(scatter_fig)

    if heat_history:
        heat_df = pd.DataFrame(heat_history)
        heat_csv = out_dir / "heat_kernel_iterations.csv"
        heat_df.to_csv(heat_csv, index=False)
        artifacts["heat_iterations_csv"] = str(heat_csv)

        heat_fig = out_dir / "heat_area_reduction.png"
        plt.figure(figsize=(10, 6))
        plt.plot(heat_df["iteration"], heat_df["heat_mass"], marker="o", color="#1f77b4", label="Total heat mass")
        plt.plot(heat_df["iteration"], heat_df["hot_area_mass"], marker="s", color="#d62728", label="Hot-area mass")
        plt.fill_between(heat_df["iteration"], heat_df["hot_area_mass"], color="#ff9896", alpha=0.25)
        plt.title("Heat Area Reduction Across Cooling Iterations")
        plt.xlabel("Cooling iteration (map-reduce pass)")
        plt.ylabel("Heat magnitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(heat_fig, dpi=140)
        plt.close()
        artifacts["heat_area_chart"] = str(heat_fig)

        try:
            from matplotlib.animation import FuncAnimation, PillowWriter

            anim_fig, ax = plt.subplots(figsize=(10, 6))
            x_vals = heat_df["iteration"].tolist()
            total_vals = heat_df["heat_mass"].tolist()
            hot_vals = heat_df["hot_area_mass"].tolist()
            tau_vals = heat_df["tau"].tolist()

            y_max = max(total_vals + hot_vals) * 1.1 if len(total_vals) > 0 else 1.0
            y_max = y_max if y_max > 0 else 1.0

            ax.set_xlim(1, max(x_vals) if len(x_vals) > 0 else 1)
            ax.set_ylim(0, y_max)
            ax.set_title("Heat Cooling Animation")
            ax.set_xlabel("Cooling iteration (map-reduce pass)")
            ax.set_ylabel("Heat magnitude")

            total_line, = ax.plot([], [], color="#1f77b4", lw=2.5, label="Total heat mass")
            hot_line, = ax.plot([], [], color="#d62728", lw=2.0, label="Hot-area mass")
            tau_text = ax.text(0.02, 0.94, "", transform=ax.transAxes)
            ax.legend(loc="upper right")
            fill_poly = [None]

            def _update(frame_idx: int):
                upto = frame_idx + 1
                cur_x = x_vals[:upto]
                cur_total = total_vals[:upto]
                cur_hot = hot_vals[:upto]
                total_line.set_data(cur_x, cur_total)
                hot_line.set_data(cur_x, cur_hot)
                if fill_poly[0] is not None:
                    fill_poly[0].remove()
                fill_poly[0] = ax.fill_between(cur_x, cur_hot, color="#ff9896", alpha=0.35)
                tau_text.set_text(f"tau = {tau_vals[frame_idx]:.3f}")
                return total_line, hot_line, tau_text

            animation = FuncAnimation(
                anim_fig,
                _update,
                frames=len(heat_df),
                interval=700,
                blit=False,
                repeat=True,
            )

            anim_path = out_dir / "heat_cooling_animation.gif"
            animation.save(anim_path, writer=PillowWriter(fps=1.4))
            plt.close(anim_fig)
            artifacts["heat_cooling_gif"] = str(anim_path)
        except Exception:
            pass

    if heat_graph_frames:
        nx = _safe_networkx_import()
        if nx is not None:
            try:
                from matplotlib.animation import FuncAnimation, PillowWriter

                risk_map = {}
                faulty_map = {}
                for n in node_snapshot or []:
                    nid = int(n["node_id"])
                    risk_map[nid] = float(n["node_risk"])
                    faulty_map[nid] = int(n["is_faulty_node"])

                base_graph = nx.Graph()
                for frame in heat_graph_frames:
                    for e in frame.get("edges", []):
                        base_graph.add_edge(int(e["src"]), int(e["dst"]))

                if base_graph.number_of_nodes() > 0:
                    pos = nx.spring_layout(base_graph, seed=42, k=0.55)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    frame_count = len(heat_graph_frames)

                    def _node_size(nid: int) -> float:
                        risk = risk_map.get(nid, 0.1)
                        return 350.0 + 2200.0 * float(risk)

                    def _render(frame_idx: int):
                        ax.clear()
                        frame = heat_graph_frames[frame_idx]
                        graph = nx.Graph()

                        for e in frame.get("edges", []):
                            graph.add_edge(int(e["src"]), int(e["dst"]), weight=float(e["weight"]))

                        nodes = list(graph.nodes())
                        if len(nodes) == 0:
                            return

                        edge_vals = [graph[u][v]["weight"] for u, v in graph.edges()]
                        edge_w = [0.6 + 2.8 * w for w in edge_vals]

                        nx.draw_networkx_edges(
                            graph,
                            pos,
                            ax=ax,
                            edge_color=edge_vals,
                            edge_cmap=plt.cm.inferno,
                            width=edge_w,
                            alpha=0.85,
                        )

                        node_sizes = [_node_size(n) for n in nodes]
                        node_colors = [risk_map.get(n, 0.0) for n in nodes]
                        nx.draw_networkx_nodes(
                            graph,
                            pos,
                            ax=ax,
                            nodelist=nodes,
                            node_size=node_sizes,
                            node_color=node_colors,
                            cmap=plt.cm.YlOrRd,
                            linewidths=1.3,
                            edgecolors=["#b91313" if faulty_map.get(n, 0) == 1 else "#2f3640" for n in nodes],
                        )

                        ax.set_title(
                            f"Heat Cooling on Graph | iteration={frame.get('iteration')} | tau={frame.get('tau')}",
                            fontsize=12,
                        )
                        ax.axis("off")

                    animation = FuncAnimation(
                        fig,
                        _render,
                        frames=frame_count,
                        interval=850,
                        repeat=True,
                    )

                    graph_anim = out_dir / "heat_graph_cooling.gif"
                    animation.save(graph_anim, writer=PillowWriter(fps=1.2))
                    plt.close(fig)
                    artifacts["heat_graph_cooling_gif"] = str(graph_anim)
            except Exception:
                pass

    return artifacts
