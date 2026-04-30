#include <iostream>
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define T 16

// Naive transpose kernel
__global__ void transposeNaive(float* A, float* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}

// Tiled transpose kernel with shared memory
__global__ void transposeTiled(float* A, float* B, int M, int N) {
    __shared__ float tile[T][T + 1]; // +1 avoids bank conflicts

    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;

    // Load into shared memory (coalesced read)
    if (row < M && col < N) {
        tile[threadIdx.y][threadIdx.x] = A[row * N + col];
    }

    __syncthreads();

    // Transpose coordinates
    int transposedRow = blockIdx.x * T + threadIdx.y;
    int transposedCol = blockIdx.y * T + threadIdx.x;

    // Write from shared memory (coalesced write)
    if (transposedRow < N && transposedCol < M) {
        B[transposedRow * M + transposedCol] =
            tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    size_t size = M * N * sizeof(float);

    // Host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = i * N + j;
        }
    }

    // Device memory
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(T, T);
    dim3 grid((N + T - 1) / T, (M + T - 1) / T);

    cudaEvent_t start, stop;
    float timeNaive, timeTiled;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---- Naive timing ----
    cudaEventRecord(start);
    transposeNaive<<<grid, block>>>(d_A, d_B, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeNaive, start, stop);

    // ---- Tiled timing ----
    cudaEventRecord(start);
    transposeTiled<<<grid, block>>>(d_A, d_B, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeTiled, start, stop);

    // Copy back result
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < M && correct; i++) {
        for (int j = 0; j < N; j++) {
            if (h_B[j * M + i] != h_A[i * N + j]) {
                correct = false;
                break;
            }
        }
    }

    std::cout << "Correct: " << (correct ? "Yes" : "No") << std::endl;
    std::cout << "Naive time (ms): " << timeNaive << std::endl;
    std::cout << "Tiled time (ms): " << timeTiled << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
