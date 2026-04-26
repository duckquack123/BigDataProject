#!/bin/bash
set -e

ROLE=${ROLE:-worker}

if [ "$ROLE" = "master" ]; then
    echo "[Master] Starting Spark master node..."
    # Example: Start a Spark master (customize as needed)
    /opt/spark/sbin/start-master.sh && tail -f /opt/spark/logs/*
else
    echo "[Worker] Starting Spark worker node..."
    # Example: Start a Spark worker and connect to master (customize as needed)
    /opt/spark/sbin/start-worker.sh spark://master:7077 && tail -f /opt/spark/logs/*
fi
