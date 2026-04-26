import os
import pytest
from fraud_detection_pic.data_generation import generate_elliptic_style_data
from fraud_detection_pic.spark_utils import build_spark_session

@pytest.fixture(scope="session")
def spark():
    spark_session = build_spark_session(appName="testing", master="local[*]", executor_memory="1g")
    yield spark_session
    spark_session.stop()

def test_data_generation_creates_clusters(spark):
    edges_df, classes_df, features_df = generate_elliptic_style_data(
        spark,
        num_fraud_rings=2,
        ring_min_size=10,
        ring_max_size=20,
        num_background_nodes=100
    )
    
    # Check that dataframes are created and not empty
    assert edges_df.count() > 0
    assert classes_df.count() > 0
    assert features_df.count() > 0
    
    # Check schema
    assert "src_id" in edges_df.columns
    assert "dst_id" in edges_df.columns
    assert "amount" in edges_df.columns
    
    assert "node_id" in classes_df.columns
    assert "true_label" in classes_df.columns
