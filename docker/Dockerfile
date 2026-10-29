# ==============================================================================
# Nevrine: Cloud-Native Neural Acceleration Engine Multi-Stage Dockerfile
# ==============================================================================

# Stage 1: Build C++ and CUDA Core Libraries & Python Extension
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-dev \
    python3-pip \
    libcublas-dev-12-4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install build dependencies
RUN pip3 install --no-cache-dir pybind11 numpy

# Copy source code
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/
COPY python/ ./python/

# Configure and compile using Ninja
RUN cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;89" \
    && cmake --build build --target _nevrine nevrine_core -j$(nproc)

# ==============================================================================
# Stage 2: Minimal Runtime Serving Image
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/python:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libcublas-12-4 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install serving dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    fastapi \
    uvicorn \
    pydantic \
    httpx

# Copy runtime library and python package from builder
COPY --from=builder /workspace/python/nevrine ./python/nevrine
COPY serving/ ./serving/

# Expose FastAPI port
EXPOSE 8000

# Health check to ensure GPU context and server are responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn server
ENTRYPOINT ["python3", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
