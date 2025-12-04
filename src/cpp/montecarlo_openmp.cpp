#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>

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

    if (argc >= 2) NUM_SIMULACIONES = atoi(argv[1]);
    if (argc >= 3) PLAZO_LIMITE = atof(argv[2]);

    auto t0 = chrono::high_resolution_clock::now();

    long exitos_global = 0;

    // Paralelización con reducción (suma segura)
    #pragma omp parallel
    {
        std::mt19937_64 rng( std::random_device{}() + omp_get_thread_num() );

        long exitos_local = 0;

        #pragma omp for
        for (long i = 0; i < NUM_SIMULACIONES; i++) {
            double t1 = beta_pert_sample(rng, 10, 12, 18);
            double t2 = beta_pert_sample(rng, 8, 10, 14);
            double t3 = beta_pert_sample(rng, 12, 14, 22);
            double total = t1 + t2 + t3;

            if (total <= PLAZO_LIMITE)
                exitos_local++;
        }

        #pragma omp atomic
        exitos_global += exitos_local;
    }

    double prob = (double)exitos_global / NUM_SIMULACIONES;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1 - t0).count();

    cout << "=== MONTECARLO OPENMP ===" << endl;
    cout << "Simulaciones: " << NUM_SIMULACIONES << endl;
    cout << fixed << setprecision(4);
    cout << "Probabilidad <= " << PLAZO_LIMITE << " dias: " << (prob * 100.0) << "%\n";
    cout << "Tiempo de Ejecución: " << ms << " ms" << endl;

    return 0;
}