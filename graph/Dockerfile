FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    systemtap-sdt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY libtrace_cuda_cupti.so /usr/local/lib/

RUN ldconfig

# Copy graph
COPY graph.cu .

# Compile saxpy
RUN nvcc graph.cu -o graph

# Set library path (if needed)
ENV CUDA_INJECTION64_PATH=/usr/local/lib/libtrace_cuda_cupti.so

CMD ["./graph"]
