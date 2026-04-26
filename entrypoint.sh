#!/bin/bash
set -e

# Fix permissions on mounted volumes (host mounts override Dockerfile chown)
if [ -d /app/outputs ]; then
  chown -R spark:spark /app/outputs 2>/dev/null || true
fi


ROLE=${ROLE:-standalone}
SPARK_MASTER_URL=${SPARK_MASTER_URL:-spark://master:7077}
WORKER_CORES=${WORKER_CORES:-2}
WORKER_MEMORY=${WORKER_MEMORY:-2g}
NUM_WORKERS=${NUM_WORKERS:-2}
WAIT_FOR_WORKERS=${WAIT_FOR_WORKERS:-30}
PIPELINE_ARGS=${PIPELINE_ARGS:---visualize --output-dir /app/outputs}

# For multi-machine setups: advertise the correct hostname/IP
export SPARK_PUBLIC_DNS=${SPARK_PUBLIC_DNS:-$(hostname)}
export SPARK_LOCAL_IP=${SPARK_LOCAL_IP:-0.0.0.0}

echo "================================================"
echo "  Role        : $ROLE"
echo "  Master URL  : $SPARK_MASTER_URL"
echo "  Public DNS  : $SPARK_PUBLIC_DNS"
echo "================================================"

case "$ROLE" in
  master)
    echo "[Master] Starting Spark master..."
    /opt/spark/sbin/start-master.sh --host 0.0.0.0
    sleep 5

    echo "[Master] Waiting ${WAIT_FOR_WORKERS}s for workers to register..."
    sleep "$WAIT_FOR_WORKERS"

    echo "[Master] Submitting fraud detection pipeline..."
    /opt/spark/bin/spark-submit \
      --master "$SPARK_MASTER_URL" \
      --deploy-mode client \
      --driver-memory 2g \
      --executor-memory "$WORKER_MEMORY" \
      --executor-cores "$WORKER_CORES" \
      /app/scripts/run_pipeline.py $PIPELINE_ARGS

    echo "[Master] Pipeline complete. Tailing logs (Ctrl+C to exit)..."
    tail -f /opt/spark/logs/* 2>/dev/null || sleep infinity
    ;;

  worker)
    echo "[Worker] Waiting for master to be available..."
    sleep 10

    echo "[Worker] Starting Spark worker -> $SPARK_MASTER_URL"
    /opt/spark/sbin/start-worker.sh \
      "$SPARK_MASTER_URL" \
      --cores "$WORKER_CORES" \
      --memory "$WORKER_MEMORY"

    echo "[Worker] Worker started. Tailing logs (Ctrl+C to exit)..."
    tail -f /opt/spark/logs/* 2>/dev/null || sleep infinity
    ;;

  standalone)
    echo "[Standalone] Starting Spark master..."
    /opt/spark/sbin/start-master.sh
    sleep 5

    MASTER_URL="spark://localhost:7077"

    echo "[Standalone] Starting $NUM_WORKERS local Spark workers..."
    for i in $(seq 1 $NUM_WORKERS); do
        /opt/spark/sbin/start-worker.sh $MASTER_URL &
    done
    sleep 5

    echo "[Standalone] Running fraud detection pipeline..."
    python3 scripts/run_pipeline.py --output-dir outputs

    echo "[Standalone] Tailing Spark logs (Ctrl+C to exit)..."
    tail -f /opt/spark/logs/* 2>/dev/null || sleep infinity
    ;;

  *)
    echo "ERROR: Unknown ROLE '$ROLE'. Use 'master', 'worker', or 'standalone'."
    exit 1
    ;;
esac
