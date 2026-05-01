// Vibe Coded Using Gemini 3
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

// CUDA Kernel for row-wise addition: A[row][col] += B[col]
__global__ void rowWiseAddKernel(double* A_local, double* B, int local_rows, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < local_rows && col < N) {
        // Flat-index mapping: row * N + col
        A_local[row * N + col] += B[col];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int M = 4096;
    const int N = 256;
    const int local_rows = M / size;

    // Set CUDA device based on rank (shared among available GPUs)
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    cudaSetDevice(rank % numGPUs);

    std::vector<double> A_full;
    std::vector<double> B(N);
    std::vector<double> A_local(local_rows * N);

    // Rank 0 initializes data
    if (rank == 0) {
        A_full.resize(M * N);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                A_full[i * N + j] = (double)i * N + j;
            }
        }
    }
    for (int j = 0; j < N; ++j) B[j] = (double)j;

    // 1. MPI Distribution
    MPI_Scatter(A_full.data(), local_rows * N, MPI_DOUBLE, 
                A_local.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Host-to-Device Copy
    double *d_A, *d_B;
    cudaMalloc(&d_A, local_rows * N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));

    cudaMemcpy(d_A, A_local.data(), local_rows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // 3. Kernel Launch
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (local_rows + 15) / 16);
    rowWiseAddKernel<<<grid, block>>>(d_A, d_B, local_rows, N);

    // 4. Synchronize
    cudaDeviceSynchronize();

    // 5. Device-to-Host Copy
    cudaMemcpy(A_local.data(), d_A, local_rows * N * sizeof(double), cudaMemcpyDeviceToHost);

    // 6. MPI Gather
    std::vector<double> A_final;
    if (rank == 0) A_final.resize(M * N);

    MPI_Gather(A_local.data(), local_rows * N, MPI_DOUBLE, 
               A_final.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Correctness Check
    if (rank == 0) {
        bool success = true;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double expected = (double)i * N + 2 * j;
                if (A_final[i * N + j] != expected) {
                    success = false;
                    break;
                }
            }
        }
        if (success) std::cout << "Verification PASSED!" << std::endl;
        else std::cout << "Verification FAILED!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    MPI_Finalize();
    return 0;
}
