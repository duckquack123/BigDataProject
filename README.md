# Fraud Detection PIC Project

This project provides a high-performance, scalable pipeline for large-scale fraud-ring detection using **Spark Power Iteration Clustering (PIC)**. It is specifically designed to handle massive transaction networks, such as the Elliptic Bitcoin dataset, using both CPU and GPU acceleration.

## 🚀 Key Features

*   **Distributed Clustering:** Uses Spark ML's Power Iteration Clustering for unsupervised fraud detection.
*   **GPU Acceleration:** Integrated with **NVIDIA RAPIDS** to offload heavy ETL, joins, and matrix math to NVIDIA GPUs (e.g., RTX 3050, A100).
*   **Heat-Kernel Laplacian Cooling:** Advanced graph-annealing technique to highlight suspicious regions in the network.
*   **Real-world Dataset Support:** Built-in scripts for processing the Elliptic Bitcoin Dataset.
*   **Visual Analytics:** Generates cluster density charts, fraud score distributions, and animated GIFs of the heat-cooling process.
*   **Cloud & Container Ready:** Includes Docker and Kubernetes deployment strategies.

---

## 📂 Project Structure

```text
big_data_project/
├── src/fraud_detection_pic/    # Core Python package
│   ├── config.py               # Pipeline & Spark parameters
│   ├── data_generation.py      # Synthetic fraud-ring generator
│   ├── main.py                 # Pipeline orchestration
│   ├── pipeline.py             # Clustering & evaluation logic
│   ├── spark_utils.py          # GPU/CPU Spark session factory
│   └── visualization.py        # Matplotlib/GIF logic
├── scripts/
│   ├── run_pipeline.py         # Main CLI entrypoint
│   └── prepare_elliptic_parquet.py # Dataset preprocessing
├── notebooks/                  # Experimental notebooks
├── outputs/                    # Generated reports & charts
├── Dockerfile                  # Containerization config
├── get-gpu-resources.sh        # GPU discovery script for Spark
├── rapids-4-spark.jar          # NVIDIA RAPIDS plugin (downloaded)
└── pyproject.toml              # Build & dependency config
```

---

## ⚙️ Setup & GPU Acceleration

### 1. Local Environment (Conda)
```bash
conda activate spark_env
python -m pip install -e .
```

### 2. Enabling NVIDIA RAPIDS (GPU)
This project is configured to automatically offload heavy workloads to your GPU if the RAPIDS plugin is present.

#### Prerequisites:
1. **NVIDIA Drivers:** Ensure you have drivers installed (e.g., version 535+). Check with `nvidia-smi`.
2. **Download the Plugin:** The project looks for a file named `rapids-4-spark.jar` in the root directory.
   ```bash
   wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.04.1/rapids-4-spark_2.12-24.04.1.jar -O rapids-4-spark.jar
   ```
3. **GPU Discovery Script:** Spark requires a script to locate your GPU. We use `get-gpu-resources.sh`:
   ```bash
   # Create the script if missing
   echo -e '#!/bin/bash\necho "{\\"name\\": \\"gpu\\", \\"addresses\\": [\\"0\\"]}"' > get-gpu-resources.sh
   chmod +x get-gpu-resources.sh
   ```

#### Verification:
When you run the pipeline, check for these indicators of success:
- **Console Logs:** Look for `WARN RapidsPluginUtils: RAPIDS Accelerator 24.04.1 using cudf ...`
- **Spark UI:** Visit `http://localhost:4040` (or 4041). In the **SQL** tab, GPU-accelerated operators will be prefixed with **`Gpu`** (e.g., `GpuHashAggregate`).

#### Disabling GPU Fallback:
If you want to force the pipeline to run on CPU even if a GPU is present, modify `src/fraud_detection_pic/spark_utils.py` and set:
```python
.config("spark.rapids.sql.enabled", "false")
```

---

## 🏃 Running the Pipeline

### Synthetic Data (Quick Test)
Run a local run with default synthetic data and visualization:
```bash
python scripts/run_pipeline.py --visualize --output-dir outputs
```

### Elliptic Bitcoin Dataset
1. **Prepare the data:**
   ```bash
   python scripts/prepare_elliptic_parquet.py \
       --edges-path elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv \
       --classes-path elliptic_bitcoin_dataset/elliptic_txs_classes.csv \
       --features-path elliptic_bitcoin_dataset/elliptic_txs_features.csv \
       --output-path outputs/elliptic_edges_prepared.parquet
   ```
2. **Run detection:**
   ```bash
   python scripts/run_pipeline.py \
       --input-path outputs/elliptic_edges_prepared.parquet \
       --input-format parquet \
       --source-col src_id \
       --destination-col dst_id \
       --weight-col amount \
       --label-col true_label \
       --pic-k 200 \
       --visualize
   ```

---

## 🐳 Dockerization

Build the image:
```bash
docker build -t fraud-detection-pic:latest .
```

Run tests inside the container:
```bash
docker run --rm fraud-detection-pic:latest python3 -m pytest
```

---

## 🐳 Distributed Docker & CI/CD

### CI/CD with GitHub Actions
- Automated build, test, and Docker image creation on every push or pull request to `main`.
- See `.github/workflows/ci-cd.yml` for details.

### Distributed Spark with Docker Compose
- Use `docker-compose.yml` to launch a Spark master and two worker containers for distributed computation.
- The `entrypoint.sh` script configures each container as either a master or worker based on the `ROLE` environment variable.

#### Quick Start

```bash
# Build and start the cluster
sudo docker compose up --build

# Master node will be available at spark://master:7077
# You can scale workers:
sudo docker compose up --scale worker=4 -d
```

- By default, `docker-compose.yml` defines a single `worker` service, but you can scale to any number of workers:

```bash
sudo docker compose up --scale worker=4 -d  # Launch 4 workers
```

- The master will auto-discover all workers on the `fraud-net` network.
- Adjust the number as needed for your workload and hardware.

---

## ☸️ Kubernetes Deployment

The project supports deployment via the **Spark Operator**. 

1. **Push image:** `docker push <registry>/fraud-detection-pic:latest`
2. **Apply RBAC:** `kubectl apply -f k8s/rbac.yaml`
3. **Submit Job:** `kubectl apply -f k8s/spark-operator-app.yaml`

---

## 📊 Visualizing Results

The pipeline generates several files in the `outputs/` directory:
- `cluster_scores.csv`: Detailed metrics for every cluster.
- `fraud_scores_by_cluster.png`: Distribution of risk across the network.
- `heat_graph_cooling.gif`: Animation of the spectral cooling process (if enabled).

Access the **Spark UI** during execution at **`http://localhost:4040`** (or 4041) to see the GPU-accelerated DAGs and query plans.

---

## 🛠️ Refactoring & Modernization
This project was refactored from a monolithic notebook into a structured Python package with the following improvements:
- **Modularity:** Separate modules for data generation, pipeline logic, and configuration.
- **Performance:** Integrated Kryo serialization and RAPIDS GPU offloading.
- **Observability:** Added timestamped logging and comprehensive summary reports.
