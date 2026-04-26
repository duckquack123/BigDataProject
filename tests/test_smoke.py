from fraud_detection_pic.main import run_pipeline
from fraud_detection_pic.config import PipelineConfig

def test_pipeline_runs_and_returns_metrics():
    # Use a tiny configuration so the smoke test runs in seconds
    pipeline_cfg = PipelineConfig(
        n_normal_nodes=50,
        n_normal_edges=100,
        pic_k=5,
        pic_max_iter=5
    )
    
    summary = run_pipeline(pipeline_cfg=pipeline_cfg)
    assert "f1" in summary
    assert "n_nodes" in summary
    assert summary["n_nodes"] > 0

