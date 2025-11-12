FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
  git build-essential cmake  libssl-dev

WORKDIR /workspace

RUN git clone https://github.com/xmrig/xmrig-cuda.git

CMD ["bash"]

