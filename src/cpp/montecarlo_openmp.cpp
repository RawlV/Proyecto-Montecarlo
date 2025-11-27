#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

const int NUM_SIMULACIONES = 1000000;
const double PLAZO_LIMITE = 30.0;

// Función que genera una normal
double sample_normal(std::mt19937_64& gen, double mean, double sd) {
    std::normal_distribution<double> dist(mean, sd);
    return dist(gen);
}

int main() {
    auto t0 = chrono::high_resolution_clock::now();

    int exitos_global = 0;

    // Paralelización con reducción (suma segura)
    #pragma omp parallel
    {
        std::mt19937_64 rng( std::random_device{}() + omp_get_thread_num() );

        int exitos_local = 0;

        #pragma omp for
        for(int i = 0; i < NUM_SIMULACIONES; i++){
            double mean = 20.0;
            double sd   = 5.0;
            double plazo = sample_normal(rng, mean, sd);

            if(plazo <= PLAZO_LIMITE){
                exitos_local++;
            }
        }

        #pragma omp atomic
        exitos_global += exitos_local;
    }

    double prob = (double)exitos_global / NUM_SIMULACIONES;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1 - t0).count();

    cout << "=== MONTECARLO OPENMP ===" << endl;
    cout << "Simulaciones: " << NUM_SIMULACIONES << endl;
    cout << "Probabilidad (<= 30 días): " << prob << endl;
    cout << "Tiempo de Ejecución: " << ms << " ms" << endl;

    return 0;
}