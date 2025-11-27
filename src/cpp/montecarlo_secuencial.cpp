#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
using namespace std;
using namespace chrono;

const int NUM_SIMULACIONES = 1000000; 
const double PLAZO_LIMITE = 30.0;

double sample_normal(std::mt19937_64& gen, double mean, double sd){
    std::normal_distribution<double> d(mean, sd);
    return d(gen);
}

int main(){
    cout << "=== VERSIÓN SECUENCIAL ===" << endl;

    auto t0 = high_resolution_clock::now();

    std::mt19937_64 gen(42);
    int exitos = 0;

    for(int i = 0; i < NUM_SIMULACIONES; i++){
        double t1 = sample_normal(gen, 10.0, 2.0);
        double t2 = sample_normal(gen, 8.0, 1.5);
        double t3 = sample_normal(gen, 12.0, 3.0);

        if ((t1 + t2 + t3) <= PLAZO_LIMITE)
            exitos++;
    }

    auto t1 = high_resolution_clock::now();
    long long ms = duration_cast<milliseconds>(t1 - t0).count();

    double prob = 100.0 * double(exitos) / NUM_SIMULACIONES;

    cout << "Probabilidad Éxito Conjunto: " 
         << fixed << setprecision(4) << prob << "%" << endl;

    cout << "Tiempo de Ejecución: " << ms << " ms" << endl;

    return 0;
}