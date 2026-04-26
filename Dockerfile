# Use the official Spark image
FROM apache/spark:3.5.0

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

# Install the project in standard mode (PEP 517/518 compatible)
RUN pip3 install --no-cache-dir .

# Spark standard environment variables
ENV PYTHONPATH=$PYTHONPATH:/app/src

# Create outputs directory and give spark user ownership
RUN mkdir -p /app/outputs && chown spark:spark /app/outputs

# Make entrypoint executable
RUN chmod +x /opt/entrypoint.sh

# Expose Spark ports
# 7077 = Master port, 8080 = Master Web UI, 8081 = Worker Web UI, 4040 = Application UI
EXPOSE 7077 8080 8081 4040

# Note: Running as root so entrypoint can fix mounted volume permissions.
# For production, consider using gosu to drop privileges after setup.

# Default command (can be overridden by spark-submit)
ENTRYPOINT [ "/opt/entrypoint.sh" ]
