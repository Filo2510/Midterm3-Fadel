#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define M 4096
#define N 256

__global__ void addBtoRows(double* A_local, const double* B, int totalElems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElems)
        A_local[idx] += B[idx % N];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // GPU based on rank
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    cudaSetDevice(rank % numGPUs);

    const int localRows  = M / size;          // rows per rank
    const int localElems = localRows * N;     // elements per rank

    std::vector<double> A, B(N, 0.0);
    if (rank == 0) {
        A.resize(M * N);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                A[i * N + j] = static_cast<double>(i * N + j);
        for (int j = 0; j < N; j++)
            B[j] = static_cast<double>(j);
    }

    std::vector<double> A_local(localElems);
    MPI_Scatter(rank == 0 ? A.data() : nullptr, localElems, MPI_DOUBLE,
                A_local.data(),                 localElems, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *d_A, *d_B;
    cudaMalloc(&d_A, localElems * sizeof(double));
    cudaMalloc(&d_B, N          * sizeof(double));

    cudaMemcpy(d_A, A_local.data(), localElems * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(),       N          * sizeof(double), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize  = (localElems + blockSize - 1) / blockSize;
    addBtoRows<<<gridSize, blockSize>>>(d_A, d_B, localElems);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(A_local.data(), d_A, localElems * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);

    if (rank == 0) A.resize(M * N);
    MPI_Gather(A_local.data(),                 localElems, MPI_DOUBLE,
               rank == 0 ? A.data() : nullptr, localElems, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        int errors = 0;
        for (int i = 0; i < M && errors < 10; i++)
            for (int j = 0; j < N && errors < 10; j++) {
                double expected = static_cast<double>(i * N + 2 * j);
                if (A[i * N + j] != expected) {
                    std::cerr << "FAIL at [" << i << "][" << j << "]: "
                              << "got " << A[i * N + j]
                              << ", expected " << expected << "\n";
                    ++errors;
                }
            }
        if (errors == 0)
            std::cout << "Verification PASSED: all A[i][j] == i*N + 2*j\n";
    }

    MPI_Finalize();
    return 0;
}
