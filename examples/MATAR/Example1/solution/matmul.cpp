#include <stdio.h>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "matar.h"
#include "timer.hpp"

using namespace mtr;

#define MATRIX_SIZE 1024

double calculate_flops(int size, double time_ms) {
    double total_ops = static_cast<double>(size) * size * (2.0 * size);
    return total_ops / (time_ms / 1000.0);
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    { // kokkos scope
    printf("Starting MATAR Matrix Multiplication (fixed)\n");
    printf("Matrix size: %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);

    CArrayKokkos<int> A(MATRIX_SIZE, MATRIX_SIZE);
    CArrayKokkos<int> B(MATRIX_SIZE, MATRIX_SIZE);
    CArrayKokkos<int> C(MATRIX_SIZE, MATRIX_SIZE);

    A.set_values(2);
    B.set_values(2);
    C.set_values(0);

    Timer timer;
    timer.start();

    // FIX: use a 2D FOR_ALL (parallel over i,j) with a serial loop over k.
    // Each (i,j) thread owns exactly one output element — no shared writes,
    // no race condition.
    FOR_ALL(i, 0, MATRIX_SIZE,
            j, 0, MATRIX_SIZE, {
        int sum = 0;
        for (int k = 0; k < MATRIX_SIZE; k++)
            sum += A(i,k) * B(k,j);
        C(i,j) = sum;
    });

    Kokkos::fence();
    double time_ms = timer.stop();

    auto C_view = C.get_kokkos_view();
    auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C_view);

    const int expected = 2 * 2 * MATRIX_SIZE;
    int n_wrong = 0;
    for (int i = 0; i < MATRIX_SIZE; i++)
        for (int j = 0; j < MATRIX_SIZE; j++)
            if (C_host(i + j * MATRIX_SIZE) != expected) n_wrong++;

    printf("Execution time: %.2f ms\n", time_ms);
    printf("Performance:    %.2f GFLOPS\n",
           2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE / (time_ms * 1e6));
    printf("Verification:   %d / %d elements correct (expected each = %d)\n",
           MATRIX_SIZE * MATRIX_SIZE - n_wrong, MATRIX_SIZE * MATRIX_SIZE, expected);
    printf("C corner (3x3):\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            printf("  %6d", C_host(i + j * MATRIX_SIZE));
        printf("\n");
    }

    } // end kokkos scope
    Kokkos::finalize();
    return 0;
}
