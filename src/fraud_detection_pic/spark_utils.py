from __future__ import annotations
import os
import sys
from pyspark.sql import SparkSession

from .config import SparkConfig


def build_spark_session(cfg: SparkConfig) -> SparkSession:
    """Create a SparkSession tuned for graph-heavy shuffle workloads."""
    # Preserve Hadoop/YARN configuration so spark-submit can run on a real cluster.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


    builder = (
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
    )

    # Only add RAPIDS configs if the plugin jar is present and not in CI
    jar_path = "rapids-4-spark.jar"
    if not os.environ.get("CI") and os.path.exists(jar_path):
        builder = (
            builder
            .config("spark.kryo.registrator", "com.nvidia.spark.rapids.GpuKryoRegistrator")
            .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
            .config("spark.rapids.sql.enabled", "true")
            .config("spark.rapids.sql.explain", "ALL")
            .config("spark.jars", jar_path)
        )

    if "local" in cfg.master:
        builder = (
            builder.config("spark.driver.resource.gpu.amount", "1")
            .config("spark.driver.resource.gpu.discoveryScript", "./get-gpu-resources.sh")
        )
    elif not cfg.master.startswith("spark://"):
        # Only request executor GPU when running on YARN/Mesos, not in
        # standalone Docker cluster mode where workers self-manage resources.
        builder = builder.config("spark.executor.resource.gpu.amount", "1")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
