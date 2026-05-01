#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        const int P = 4, Q = 2, M = 16;
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        const int proc_row = world_rank / Q;
        const int proc_col = world_rank % Q;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(MPI_COMM_WORLD, proc_row, proc_col, &row_comm);
        MPI_Comm_split(MPI_COMM_WORLD, proc_col, proc_row, &col_comm);

        const int chunk    = M / P;
        const int loc_size = chunk / Q;

        std::vector<double> x_full(M, 0.0);
        if (world_rank == 0)
            for (int j = 0; j < M; j++) x_full[j] = static_cast<double>(j);

        std::vector<double> x_chunk(chunk, 0.0);
        if (proc_col == 0)
            MPI_Scatter(world_rank == 0 ? x_full.data() : nullptr,
                        chunk, MPI_DOUBLE,
                        x_chunk.data(), chunk, MPI_DOUBLE, 0, col_comm);

        MPI_Bcast(x_chunk.data(), chunk, MPI_DOUBLE, 0, row_comm);

        // Load into Kokkos View
        Kokkos::View<double*> x_dev("x", chunk);
        {
            auto hx = Kokkos::create_mirror_view(x_dev);
            for (int i = 0; i < chunk; i++) hx(i) = x_chunk[i];
            Kokkos::deep_copy(x_dev, hx);
        }

        Kokkos::View<double*> y_dev("y", chunk);
        Kokkos::parallel_for("scale", chunk, KOKKOS_LAMBDA(int i) {
            y_dev(i) = 2.0 * x_dev(i);
        });
        Kokkos::fence();

        // Scatter distribution
        auto hy = Kokkos::create_mirror_view(y_dev);
        Kokkos::deep_copy(hy, y_dev);

        std::vector<double> y_scat(loc_size);
        for (int j = 0; j < loc_size; j++)
            y_scat[j] = hy(proc_col + j * Q);

        std::vector<double> row_buf(chunk, 0.0);
        MPI_Gather(y_scat.data(), loc_size, MPI_DOUBLE,
                   row_buf.data(), loc_size, MPI_DOUBLE, 0, row_comm);

        if (proc_col == 0) {
            // Restore linear order
            std::vector<double> row_ord(chunk);
            for (int idx = 0; idx < chunk; idx++)
                row_ord[idx] = row_buf[(idx % Q) * loc_size + idx / Q];

            // Gather all row-ordered chunks
            std::vector<double> y_full_out(M, 0.0);
            MPI_Gather(row_ord.data(), chunk, MPI_DOUBLE,
                       world_rank == 0 ? y_full_out.data() : nullptr,
                       chunk, MPI_DOUBLE, 0, col_comm);

            if (world_rank == 0) {
                bool ok = true;
                for (int J = 0; J < M && ok; J++)
                    if (y_full_out[J] != 2.0 * J) ok = false;
                std::cout << (ok ? "PASSED: y[J] == 2*x[J] for all J\n"
                                 : "FAILED verification\n");
                for (int J = 0; J < M; J++)
                    std::cout << "  y[" << J << "] = " << y_full_out[J] << "\n";
            }
        }

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
