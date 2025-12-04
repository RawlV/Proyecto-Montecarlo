#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>
using namespace std;

double beta_pert_sample(std::mt19937_64 &gen, double a, double m, double b) {
    double alpha = 1 + 4 * ((m - a) / (b - a));
    double beta  = 1 + 4 * ((b - m) / (b - a));

    std::gamma_distribution<double> dist_alpha(alpha, 1.0);
    std::gamma_distribution<double> dist_beta(beta, 1.0);

    double x = dist_alpha(gen);
    double y = dist_beta(gen);

    double beta_raw = x / (x + y);
    double result = a + beta_raw * (b - a);

    return result;
}

struct Tarea {
    double optimo; 
    double probable;
    double pesimo;
};

int main(int argc, char** argv) {

    long NUM_SIMULACIONES = (argc >= 2) ? atol(argv[1]) : 1000000;
    double PLAZO_LIMITE   = (argc >= 3) ? atof(argv[2]) : 30.0;

    vector<Tarea> tareas = {
        {10, 12, 18},
        {8, 10, 14},
        {12, 14, 22}
    };


    long exitos = 0;

    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::mt19937_64 rng(std::random_device{}() + omp_get_thread_num());
        int exitos_local = 0;

        #pragma omp for
        for (int i = 0; i < NUM_SIMULACIONES; i++) {
            double total = 0.0;
            for (auto &t : tareas)
                total += beta_pert_sample(rng, t.optimo, t.probable, t.pesimo);

            if (total <= PLAZO_LIMITE)
                exitos_local++;
        }

        #pragma omp atomic
        exitos += exitos_local;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1 - t0).count();

    double prob = (double)exitos / NUM_SIMULACIONES;

    cout << "=== MONTECARLO PERT (OpenMP) ===" << endl;
    cout << "Simulaciones: " << NUM_SIMULACIONES << endl;
    cout << fixed << setprecision(4);
    cout << "Probabilidad <= " << PLAZO_LIMITE << " dias: " << (prob * 100.0) << "%\n";
    cout << "Tiempo de Ejecución: " << ms << " ms" << endl;

    return 0;
}