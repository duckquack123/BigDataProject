# Use the official Spark image for Python
FROM apache/spark-py:v3.5.0

USER root

# Install system dependencies if needed (e.g., for matplotlib)
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt


# Copy the source code and installation files
COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY entrypoint.sh /opt/entrypoint.sh

# Install the project in editable mode (or standard mode)
RUN pip3 install --no-cache-dir -e .

# Spark standard environment variables
ENV PYTHONPATH=$PYTHONPATH:/app/src

# Make entrypoint executable
RUN chmod +x /opt/entrypoint.sh

# Set the default user back to spark for security
USER spark

# Default command (can be overridden by spark-submit)
ENTRYPOINT [ "/opt/entrypoint.sh" ]
