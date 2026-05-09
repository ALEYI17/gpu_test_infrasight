import torch

device = "cuda"

x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)

# Warmup (required)
for _ in range(3):
    z = x @ y

g = torch.cuda.CUDAGraph()

# Static memory (important!)
static_x = torch.randn_like(x)
static_y = torch.randn_like(y)

torch.cuda.synchronize()

with torch.cuda.graph(g):
    static_z = static_x @ static_y

# Replay graph multiple times
for _ in range(1000):
    g.replay()

torch.cuda.synchronize()
