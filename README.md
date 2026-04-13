# Fraud Detection PIC Project

This project converts the original notebook workflow into a structured Python package for large-scale fraud-ring detection with Spark Power Iteration Clustering (PIC).

## Project Structure

```text
big_data_project/
├── notebooks/
│   └── fraud_detection_pic.ipynb
├── scripts/
│   └── run_pipeline.py
├── src/
│   └── fraud_detection_pic/
│       ├── __init__.py
│       ├── config.py
│       ├── data_generation.py
│       ├── main.py
│       ├── pipeline.py
│       └── spark_utils.py
├── tests/
│   └── test_smoke.py
├── pyproject.toml
└── requirements.txt
```

## Setup (Existing spark_env)

```bash
conda activate spark_env
python -m pip install -U pip
python -m pip install -e .
```

If dependencies are already installed in `spark_env`, you can skip installation and run directly.

## Run

```bash
conda activate spark_env
python scripts/run_pipeline.py
```

## Visualize Results

```bash
conda activate spark_env
python scripts/run_pipeline.py --visualize --output-dir outputs
```

## Heat-Kernel Laplacian Cooling

Use heat-kernel annealing to progressively cool the graph affinity and emphasize suspicious/faulty-node regions:

```bash
conda activate spark_env
python scripts/run_pipeline.py \
	--heat-kernel \
	--tau-start 1.2 \
	--tau-end 0.2 \
	--cooling-steps 6 \
	--faulty-node-quantile 0.95 \
	--faulty-edge-boost 0.35 \
	--visualize --output-dir outputs
```

Notes:
- `tau-start -> tau-end` controls temperature decay (higher to lower).
- `faulty-node-quantile` picks top-risk nodes to treat as suspicious.
- `faulty-edge-boost` increases affinity for edges touching suspicious nodes.

This writes at least one file:
- `outputs/cluster_scores.csv`

If `matplotlib` is available in your environment, it also writes:
- `outputs/fraud_scores_by_cluster.png`
- `outputs/density_vs_cv.png`

When heat-kernel mode is enabled, it also writes:
- `outputs/heat_kernel_iterations.csv`
- `outputs/heat_area_reduction.png`
- `outputs/heat_cooling_animation.gif`
- `outputs/heat_graph_cooling.gif`

Install plotting support if needed:

```bash
conda activate spark_env
python -m pip install matplotlib
```

## Optional Test

```bash
conda activate spark_env
python -m pytest -q
```

## What Was Refactored

- Spark session creation was moved to `spark_utils.py`.
- Synthetic graph generation was moved to `data_generation.py`.
- Clustering, scoring, and evaluation steps were moved to `pipeline.py`.
- Orchestration and summary formatting were moved to `main.py`.
- Notebook remains available under `notebooks/` for experimentation.
