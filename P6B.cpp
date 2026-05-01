// Vibe Coded using Gemini 3
#include <iostream>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (size != 8) {
            if (rank == 0) std::cerr << "Error: World size must be 8." << std::endl;
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }

        // i. Create 4x2 Topology
        const int P = 4; // Rows
        const int Q = 2; // Columns
        int p = rank / Q;
        int q = rank % Q;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(MPI_COMM_WORLD, p, q, &row_comm); // Row sub-comm (color = p)
        MPI_Comm_split(MPI_COMM_WORLD, q, p, &col_comm); // Col sub-comm (color = q)

        const int M = 16;
        const int local_size = M / P; // Linear chunk size for col 0 scatter

        // ii. Scatter down column 0, then Broadcast horizontally
        Kokkos::View<double*, Kokkos::HostSpace> h_x_local("h_x_local", local_size);
        
        if (q == 0) {
            std::vector<double> x_full;
            if (rank == 0) {
                x_full.resize(M);
                for (int i = 0; i < M; ++i) x_full[i] = (double)i;
            }
            MPI_Scatter(x_full.data(), local_size, MPI_DOUBLE, 
                        h_x_local.data(), local_size, MPI_DOUBLE, 0, col_comm);
        }

        // Broadcast local chunks horizontally within each row
        MPI_Bcast(h_x_local.data(), local_size, MPI_DOUBLE, 0, row_comm);

        // iii. Kokkos Parallel For Scaling
        // Move to device View for the kernel
        Kokkos::View<double*> x_local("x_local", local_size);
        Kokkos::deep_copy(x_local, h_x_local);

        Kokkos::parallel_for("Scale", local_size, KOKKOS_LAMBDA(const int i) {
            x_local(i) = 2.0 * x_local(i);
        });
        Kokkos::fence(); // Ensure kernel completion

        // iv. Redistribute to Scatter Distribution
        // Global index J -> process column q = J % Q, local index j = J / Q
        const int scatter_local_size = M / Q; 
        Kokkos::View<double*, Kokkos::HostSpace> h_y_scatter("h_y_scatter", scatter_local_size);
        
        // Manual redistribution for verification gathering
        // Every process sends its scaled local x back to rank 0 for global assembly
        std::vector<double> y_global_scaled(M);
        std::vector<double> x_local_gathered(local_size);
        Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(x_local_gathered.data(), local_size), x_local);

        std::vector<double> all_scaled(M);
        MPI_Gather(x_local_gathered.data(), local_size, MPI_DOUBLE, 
                   all_scaled.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // v. Rank (0,0) gathers and prints the full y
        if (rank == 0) {
            // Reassemble the full y based on the logic that rank 0, 2, 4, 6 
            // held the scaled linear chunks [0-3, 4-7, 8-11, 12-15]
            std::cout << "Final scaled y (Scatter Distribution Check):" << std::endl;
            for (int J = 0; J < M; ++J) {
                double val = all_scaled[J]; 
                std::cout << "y[" << J << "] = " << val << " (Expected: " << J * 2 << ")" << std::endl;
            }
        }

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
    }

    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
