import pytest
from fraud_detection_pic.data_generation import FraudGraphGenerator
from fraud_detection_pic.config import RingConfig

def test_data_generation_creates_edges():
    rings = [
        RingConfig(id_offset=1000, n_nodes=5, weight_mean=50.0, weight_std=10.0),
        RingConfig(id_offset=2000, n_nodes=3, weight_mean=100.0, weight_std=5.0)
    ]
    generator = FraudGraphGenerator(n_normal_nodes=50, n_normal_edges=100, fraud_rings=rings)
    edges = generator.generate()
    
    # Check that edges are created
    assert len(edges) > 100
    
    # Check that the tuple structure is correct (src, dst, weight, label)
    assert len(edges[0]) == 4
    
    # Verify that we have some fraud ring edges
    labels = [edge[3] for edge in edges]
    assert "fraud_ring_A" in labels
    assert "fraud_ring_B" in labels
    assert "normal" in labels
