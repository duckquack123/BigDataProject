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


## 🐳 Distributed Docker Cluster

Each Spark worker runs in **its own Docker container**, forming a true distributed cluster.

### Quick Start (Distributed)
```bash
# Build and launch master + 2 workers
sudo docker compose up --build --scale worker=2

# Scale to more workers on the fly
sudo docker compose up --scale worker=4 -d
```

**What happens:**
1. **Master container** starts the Spark master, waits for workers, then runs the pipeline via `spark-submit`
2. **Worker containers** each start a Spark worker that connects to `spark://master:7077`
3. Results are saved to the `outputs/` directory on your host (volume-mounted)

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROLE` | `standalone` | Container role: `master`, `worker`, or `standalone` |
| `WORKER_CORES` | `2` | CPU cores per worker |
| `WORKER_MEMORY` | `2g` | Memory per worker |
| `WAIT_FOR_WORKERS` | `30` | Seconds master waits for workers before submitting |
| `PIPELINE_ARGS` | `--visualize --output-dir /app/outputs` | CLI args for the pipeline |

### Spark UI Access
- **Master UI:** [http://localhost:8080](http://localhost:8080)
- **Application UI:** [http://localhost:4040](http://localhost:4040)

---

## 🌐 Multi-Machine Deployment

Run the master on your machine and workers on **other physical machines** on the same network.

### Option A: Docker Swarm (Recommended)

Docker Swarm makes multi-machine deployment feel just like `docker compose`.

#### 1. Initialize Swarm on the Master machine (your machine):
```bash
docker swarm init --advertise-addr <YOUR_LAN_IP>
# Example: docker swarm init --advertise-addr 192.168.1.10
```
This prints a `docker swarm join` command with a token.

#### 2. Join each Worker machine to the Swarm:
Run the join command on every other machine:
```bash
docker swarm join --token <TOKEN> <MASTER_IP>:2377
```

#### 3. Deploy the cluster:
Back on the master machine:
```bash
docker stack deploy -c docker-compose.swarm.yml fraud
```

#### 4. Monitor and Scale:
```bash
# Check services
docker service ls

# Scale to 5 workers across all machines
docker service scale fraud_worker=5

# View master logs
docker service logs fraud_master -f
```

#### 5. Tear down:
```bash
docker stack rm fraud
docker swarm leave --force  # on each machine
```

### Option B: Manual `docker run` (No Swarm)

If you can't use Swarm, you can manually start containers on each machine.

#### On your machine (Master):
```bash
docker run -d --name spark-master \
  -e ROLE=master \
  -e SPARK_MASTER_URL=spark://<YOUR_IP>:7077 \
  -e WAIT_FOR_WORKERS=60 \
  -e "PIPELINE_ARGS=--visualize --output-dir /app/outputs --master spark://<YOUR_IP>:7077 --executor-memory 2g --executor-cores 2" \
  -p 7077:7077 -p 8080:8080 -p 4040:4040 \
  -v "$PWD/outputs:/app/outputs" \
  --network host \
  ghcr.io/duckquack123/bigdataproject/fraud-detection-pic:latest
```

#### On each Worker machine:
```bash
docker run -d --name spark-worker \
  -e ROLE=worker \
  -e SPARK_MASTER_URL=spark://<MASTER_IP>:7077 \
  -e WORKER_CORES=4 \
  -e WORKER_MEMORY=4g \
  --network host \
  ghcr.io/duckquack123/bigdataproject/fraud-detection-pic:latest
```

> **Note:** `--network host` is required so workers and master can communicate directly over the LAN.

### Network Requirements

| Port | Purpose | Must be open on |
|------|---------|-----------------|
| `7077` | Spark Master | Master machine |
| `8080` | Master Web UI | Master machine |
| `4040` | Application UI | Master machine |
| `2377` | Swarm management (Swarm only) | Master machine |

---

### 🐳 Standalone Mode (All-in-One, No Compose Needed)

Run everything in a single container (backward compatible):

```bash
# Pull from GHCR
docker pull ghcr.io/duckquack123/bigdataproject/fraud-detection-pic:latest

# Run all-in-one
docker run --rm \
   -v "$PWD/outputs:/app/outputs" \
   -e ROLE=standalone \
   -e NUM_WORKERS=2 \
   ghcr.io/duckquack123/bigdataproject/fraud-detection-pic:latest
```

---

## 🐳 CI/CD with GitHub Actions
- Automated build and Docker image push on every push or pull request.
- See `.github/workflows/ci-cd.yml` for details.

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
