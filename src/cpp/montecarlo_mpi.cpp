#include <mpi.h>
#include <iostream>
#include <random>
#include <chrono>
using namespace std;

const int NUM_SIMULACIONES = 1000000;
const double PLAZO_LIMITE = 30.0;

// Función normal
double sample_normal(std::mt19937_64& gen, double mean, double sd) {
    std::normal_distribution<double> dist(mean, sd);
    return dist(gen);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto t0 = chrono::high_resolution_clock::now();

    int simulaciones_por_proceso = NUM_SIMULACIONES / size;
    int exitos_local = 0;

    std::mt19937_64 rng(std::random_device{}() + rank);

    for (int i = 0; i < simulaciones_por_proceso; i++) {
        double mean = 20.0;
        double sd = 5.0;
        double plazo = sample_normal(rng, mean, sd);
        if (plazo <= PLAZO_LIMITE)
            exitos_local++;
    }

    int exitos_global = 0;

    MPI_Reduce(&exitos_local, &exitos_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double prob = (double)exitos_global / NUM_SIMULACIONES;

        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t1 - t0).count();

        cout << "=== MONTECARLO MPI ===" << endl;
        cout << "Procesos utilizados: " << size << endl;
        cout << "Simulaciones totales: " << NUM_SIMULACIONES << endl;
        cout << "Probabilidad (<= 30 días): " << prob << endl;
        cout << "Tiempo de Ejecución: " << ms << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}