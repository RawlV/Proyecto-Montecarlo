#include <mpi.h>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
using namespace std;

int NUM_SIMULACIONES = 1000000;
double PLAZO_LIMITE = 30.0;

double beta_pert_sample(std::mt19937_64 &gen, double a, double m, double b) {
    double alpha = 1 + 4 * ((m - a) / (b - a));
    double beta  = 1 + 4 * ((b - m) / (b - a));

    std::gamma_distribution<double> dist_alpha(alpha, 1.0);
    std::gamma_distribution<double> dist_beta(beta, 1.0);

    double x = dist_alpha(gen);
    double y = dist_beta(gen);

    return a + (x / (x + y)) * (b - a);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    if (argc >= 2) NUM_SIMULACIONES = atoi(argv[1]);
    if (argc >= 3) PLAZO_LIMITE = atof(argv[2]);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto t0 = chrono::high_resolution_clock::now();

    long simulaciones_por_proceso = NUM_SIMULACIONES / size;
    int exitos_local = 0;

    std::mt19937_64 rng(std::random_device{}() + rank);

    for (long i = 0; i < simulaciones_por_proceso; i++) {
        double t1 = beta_pert_sample(rng, 10, 12, 18);
        double t2 = beta_pert_sample(rng, 8, 10, 14);
        double t3 = beta_pert_sample(rng, 12, 14, 22);
        double total = t1 + t2 + t3;

        if (total <= PLAZO_LIMITE)
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
        cout << fixed << setprecision(4);
    cout << "Probabilidad <= " << PLAZO_LIMITE << " dias: " << (prob * 100.0) << "%\n";
        cout << "Tiempo de Ejecución: " << ms << " ms" << endl;
    }

    MPI_Finalize();
    return 0;
}