import os
import sys
from pyspark.sql import SparkSession

from .config import SparkConfig


def build_spark_session(cfg: SparkConfig) -> SparkSession:
    """Create a SparkSession tuned for graph-heavy shuffle workloads."""
    # Use the cluster's default environment or spark-submit flags.

    spark = (
        SparkSession.builder.appName(cfg.app_name)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.unsafe", "true")
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.shuffle.partitions", cfg.shuffle_partitions)
        .config("spark.default.parallelism", cfg.default_parallelism)
        .config("spark.shuffle.compress", "true")
        .config("spark.shuffle.spill.compress", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.ml.powerIterationClustering.convergenceTol", "1e-5")
        .master(cfg.master)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
