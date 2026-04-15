from dataclasses import dataclass, field


@dataclass(frozen=True)
class SparkConfig:
    app_name: str = "FraudDetection_SpectralClustering_PIC"
    master: str = "local[*]"
    driver_memory: str = "2g"
    executor_memory: str = "4g"
    shuffle_partitions: str = "200"
    default_parallelism: str = "200"


@dataclass(frozen=True)
class RingConfig:
    n_nodes: int
    weight_mean: float
    weight_std: float
    id_offset: int


@dataclass(frozen=True)
class PipelineConfig:
    n_normal_nodes: int = 500
    n_normal_edges: int = 3000
    seed: int = 42

    pic_k: int = 200
    pic_max_iter: int = 40
    pic_init_mode: str = "random"

    use_heat_kernel: bool = False
    heat_tau_start: float = 1.2
    heat_tau_end: float = 0.2
    heat_steps: int = 6
    heat_distance_scale: float = 1.0
    faulty_node_quantile: float = 0.95
    faulty_edge_boost: float = 0.35
    heat_graph_max_edges: int = 700
    heat_graph_min_weight: float = 0.08

    fraud_score_threshold: float = 0.35
    micro_cluster_max_size: int = 2000
    min_internal_density: float = 0.01
    top_fraud_clusters: int = 5

    density_weight: float = 0.50
    uniformity_weight: float = 0.30
    volume_weight: float = 0.20

    fraud_rings: list[RingConfig] = field(
        default_factory=lambda: [
            RingConfig(n_nodes=8, weight_mean=950.0, weight_std=15.0, id_offset=10_000),
            RingConfig(n_nodes=6, weight_mean=480.0, weight_std=8.0, id_offset=20_000),
        ]
    )
