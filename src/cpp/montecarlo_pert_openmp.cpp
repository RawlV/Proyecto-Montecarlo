#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <omp.h>
using namespace std;

/* ================================
   Función Beta-PERT
   ================================ */
double beta_pert_sample(std::mt19937_64 &gen, double a, double m, double b) {
    double alpha = 1 + 4 * ((m - a) / (b - a));
    double beta  = 1 + 4 * ((b - m) / (b - a));

    std::gamma_distribution<double> dist_alpha(alpha, 1.0);
    std::gamma_distribution<double> dist_beta(beta, 1.0);

    double x = dist_alpha(gen);
    double y = dist_beta(gen);

    double beta_raw = x / (x + y);
    return a + beta_raw * (b - a);
}

/* Estructura de cada tarea */
struct Tarea {
    double optimo;   // a
    double probable; // m
    double pesimo;   // b
};

/* ================================
   MAIN (OpenMP + Beta-PERT)
   ================================ */
int main() {
    const int NUM_SIMULACIONES = 1000000;

    // Definir tareas del proyecto
    vector<Tarea> tareas = {
        {5, 7, 10},   // Tarea 1
        {4, 6, 12},   // Tarea 2
        {8, 10, 15},  // Tarea 3
        {3, 4, 7}     // Tarea 4
    };

    int exitos = 0;
    const double PLAZO_LIMITE = 30.0;

    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::mt19937_64 rng(std::random_device{}() + omp_get_thread_num());
        int exitos_local = 0;

        #pragma omp for
        for (int i = 0; i < NUM_SIMULACIONES; i++) {
            double total = 0.0;
            for (const auto &t : tareas) {
                total += beta_pert_sample(rng, t.optimo, t.probable, t.pesimo);
            }
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
    cout << "Probabilidad de completar <= 30 días: "
         << prob << endl;
    cout << "Tiempo de Ejecución: " << ms << " ms" << endl;

    return 0;
}