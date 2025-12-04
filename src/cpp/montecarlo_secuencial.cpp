#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
using namespace std;
using namespace chrono;

int NUM_SIMULACIONES = 1000000; 
double PLAZO_LIMITE = 30.0;

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

int main(int argc, char* argv[]){

    if (argc >= 2) NUM_SIMULACIONES = atoi(argv[1]);
    if (argc >= 3) PLAZO_LIMITE     = atof(argv[2]);

    cout << "=== VERSIÓN SECUENCIAL ===" << endl;
    cout << "Simulaciones: " << NUM_SIMULACIONES
         << " | Plazo límite: " << PLAZO_LIMITE << " días" << endl;

    auto t0 = chrono::high_resolution_clock::now();

    std::mt19937_64 rng(std::random_device{}());
    int exitos = 0;

    for(int i = 0; i < NUM_SIMULACIONES; i++){
        double t1 = beta_pert_sample(rng, 10, 12, 18);
        double t2 = beta_pert_sample(rng, 8, 10, 14);
        double t3 = beta_pert_sample(rng, 12, 14, 22);

        double total = t1 + t2 + t3;

        if ((total) <= PLAZO_LIMITE)
            exitos++;
    }

    double prob = (double)exitos / NUM_SIMULACIONES;

    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, std::milli>(t1 - t0).count();

    cout << "=== MONTECARLO SECUENCIAL ===\n";
    cout << "Simulaciones: " << NUM_SIMULACIONES << endl;
    cout << fixed << setprecision(4);
    cout << "Probabilidad <= " << PLAZO_LIMITE << " dias: " << (prob * 100.0) << "%\n";
    cout << "Tiempo: " << ms << " ms\n";

    return 0;
}