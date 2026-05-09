#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h> // for usleep

#define N (1 << 20)

__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main()
{
    float *x, *y;
    float *d_x, *d_y;

    // Allocate host memory
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ----------------------------
    // Build CUDA Graph
    // ----------------------------
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Kernel launch inside graph
    saxpy<<<(N + 255) / 256, 256, 0, stream>>>(N, 2.0f, d_x, d_y);

    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // ----------------------------
    // Replay graph in bursts
    // ----------------------------
    for (int burst = 0; burst < 10; burst++) {
        printf("=== Burst %d ===\n", burst);

        for (int i = 0; i < 50; i++) {
            cudaGraphLaunch(graphExec, stream);
        }

        cudaStreamSynchronize(stream);

        // 🔴 IMPORTANT: idle gap → allows your cleanup logic to trigger
        usleep(100000); // 100 ms
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}
