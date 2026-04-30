// Vibe-coded using AI assistant Model: ChatGPT (GPT-5.3)

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1024; // can change for runs
    if (argc > 1) N = atoi(argv[1]);

    // Start timing
    double t0 = MPI_Wtime();

    // Compute local range
    int start = (rank * N) / size;
    int end   = ((rank + 1) * N) / size;
    int local_n = end - start;

    // Allocate local vectors
    std::vector<double> u(local_n), v(local_n);

    // Initialize
    for (int i = 0; i < local_n; i++) {
        int global_i = start + i;
        u[i] = global_i + 1;
        v[i] = 1.0 / (global_i + 1);
    }

    // Local dot product
    double local_dot = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_dot += u[i] * v[i];
    }

    // Binary tree reduction (manual)
    double sum = local_dot;

    int step = 1;
    while (step < size) {
        if (rank % (2 * step) == 0) {
            // Receiver
            if (rank + step < size) {
                double recv_val;
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, rank + step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sum += recv_val;
            }
        } else {
            // Sender
            int target = rank - step;
            MPI_Send(&sum, 1, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            break; // become inactive
        }
        step *= 2;
    }

    double t1 = MPI_Wtime();
    double elapsed = t1 - t0;

    // Get max time across processes
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "N = " << N << ", P = " << size << std::endl;
        std::cout << "Dot product = " << sum << std::endl;
        std::cout << "Max time = " << max_time << " seconds\n" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
