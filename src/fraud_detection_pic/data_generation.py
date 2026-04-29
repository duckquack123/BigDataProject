from __future__ import annotations
import itertools
import random
from dataclasses import asdict

import numpy as np

from .config import RingConfig


class FraudGraphGenerator:
    """Generate synthetic edges with embedded fraud-ring cliques."""

    def __init__(
        self,
        n_normal_nodes: int = 500,
        n_normal_edges: int = 3000,
        fraud_rings: list[RingConfig] | None = None,
        seed: int = 42,
    ) -> None:
        self.n_normal_nodes = n_normal_nodes
        self.n_normal_edges = n_normal_edges
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.fraud_rings = fraud_rings or []

    def _generate_normal_edges(self) -> list[tuple[int, int, float, str]]:
        edges: list[tuple[int, int, float, str]] = []
        node_ids = list(range(1, self.n_normal_nodes + 1))

        for _ in range(self.n_normal_edges):
            src = int(random.choice(node_ids))
            dst = int(random.choice(node_ids))
            if src == dst:
                continue
            weight = float(np.random.lognormal(mean=4.5, sigma=1.2))
            weight = max(round(weight, 2), 1.0)
            edges.append((src, dst, weight, "normal"))
        return edges

    def _generate_fraud_ring(self, ring_cfg: RingConfig, ring_label: str) -> list[tuple[int, int, float, str]]:
        cfg = asdict(ring_cfg)
        node_ids = list(range(cfg["id_offset"], cfg["id_offset"] + cfg["n_nodes"]))

        edges: list[tuple[int, int, float, str]] = []
        for src, dst in itertools.combinations(node_ids, 2):
            weight = float(max(np.random.normal(cfg["weight_mean"], cfg["weight_std"]), 1.0))
            edges.append((int(src), int(dst), round(weight, 2), ring_label))
        return edges

    def generate(self) -> list[tuple[int, int, float, str]]:
        all_edges = self._generate_normal_edges()

        for i, ring in enumerate(self.fraud_rings):
            label = f"fraud_ring_{chr(65 + i)}"
            all_edges.extend(self._generate_fraud_ring(ring, label))

        random.shuffle(all_edges)
        return all_edges
