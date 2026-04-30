#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 1024
#define N 1024
#define T 16

__global__ void naiveTranspose(const float *A, float *B) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    B[col*M + row] = A[row*N + col];
}

__global__ void tiledTranspose(const float *A, float *B) {
    __shared__ float tile[T][T+1];

    int col = blockIdx.x*T + threadIdx.x;
    int row = blockIdx.y*T + threadIdx.y;

    if (row < M && col < N)
        tile[threadIdx.y][threadIdx.x] = A[row*N + col];
    __syncthreads();

    int out_col = blockIdx.y*T + threadIdx.x;
    int out_row = blockIdx.x*T + threadIdx.y;

    if (out_row < N && out_col < M)
        B[out_row*M + out_col] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char *argv[]) {
    int nsize = M * N;

    // Allocate host memory
    float *A = (float *)malloc(nsize * sizeof(float));
    float *B = (float *)malloc(nsize * sizeof(float));

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            A[i*N + j] = (float)(i*N + j);

    // Allocate device memory
    float *A_d, *B_d;
    cudaMalloc(&A_d, nsize * sizeof(float));
    cudaMalloc(&B_d, nsize * sizeof(float));

    cudaMemcpy(A_d, A, nsize*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksize(T, T);
    dim3 gridsize((N + T-1)/T, (M + T-1)/T);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms_naive, ms_tiled;

    // Naive transpose
    cudaEventRecord(start);
    naiveTranspose<<<gridsize, blocksize>>>(A_d, B_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_naive, start, stop);

    // Tiled transpose
    cudaEventRecord(start);
    tiledTranspose<<<gridsize, blocksize>>>(A_d, B_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms_tiled, start, stop);

    cudaMemcpy(B, B_d, nsize*sizeof(float), cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < M && errors < 10; i++)
        for (int j = 0; j < N && errors < 10; j++)
            if (B[j*M + i] != A[i*N + j]) errors++;

    printf("Verification: %s\n", errors == 0 ? "PASSED" : "FAILED");
    printf("Naive  time : %.4f ms\n", ms_naive);
    printf("Tiled  time : %.4f ms\n", ms_tiled);
    printf("Speedup     : %.2fx\n",   ms_naive / ms_tiled);

    cudaFree(A_d); cudaFree(B_d);
    free(A); free(B);
    return 0;
}
