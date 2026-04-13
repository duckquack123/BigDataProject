from fraud_detection_pic.main import run_pipeline


def test_pipeline_runs_and_returns_metrics():
    summary = run_pipeline()
    assert "f1" in summary
    assert "n_nodes" in summary
    assert summary["n_nodes"] > 0
