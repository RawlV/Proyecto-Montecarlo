import subprocess, json, os, re, time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
os.makedirs('../../results', exist_ok=True)

print("=== CONFIGURACIÓN DE LA SIMULACIÓN ===")

num_sim = input("Número de simulaciones (default 1,000,000): ")
num_sim = int(num_sim) if num_sim.strip() else 1_000_000

plazo_lim = input("Plazo límite en días (default 30): ")
plazo_lim = float(plazo_lim) if plazo_lim.strip() else 30.0

mpi_procs = input("Número de procesos MPI (default 4): ")
mpi_procs = int(mpi_procs) if mpi_procs.strip() else 4

py_workers = input("Workers Python (default 12): ")
py_workers = int(py_workers) if py_workers.strip() else 12

py_chunk = input("Chunk por worker Python (default 100000): ")
py_chunk = int(py_chunk) if py_chunk.strip() else 100000

print("\n=== INICIANDO BENCHMARK ===\n")

def run_cmd(cmd, shell=False):
    print("Ejecutándose:", cmd)
    start = time.time()
    try:
        out = subprocess.check_output(cmd, shell=shell, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    end = time.time()
    elapsed_ms = (end - start) * 1000.0
    return out, elapsed_ms

# 1) C++ Secuencial
out_seq, t_seq = run_cmd([f"../../src/cpp/secuencial", str(num_sim), str(plazo_lim)])
with open('../../results/secuencial.txt', 'w') as f: f.write(out_seq)

# 2) C++ OpenMP
out_omp, t_omp = run_cmd([f"../../src/cpp/openmp", str(num_sim), str(plazo_lim)])
with open('../../results/openmp.txt', 'w') as f: f.write(out_omp)

# 3) MPI
mpi_cmd = f"mpirun -np {mpi_procs} ../../src/cpp/mpi {num_sim} {plazo_lim}"
out_mpi, t_mpi = run_cmd(mpi_cmd, shell=True)
with open('../../results/mpi.txt', 'w') as f: f.write(out_mpi)

# 4) C++ OpenMP PERT
out_pert_cpp, t_pert_cpp = run_cmd([f"../../src/cpp/pert", str(num_sim), str(plazo_lim)])
with open('../../results/pert_cpp.txt', 'w') as f: f.write(out_pert_cpp)

# 5) Python Parallel PERT
py_cmd = f"python3 pert_parallel.py --n {num_sim} --workers {py_workers} --chunk {py_chunk}"
out_py, t_py = run_cmd(py_cmd, shell=True)
with open('../../results/pert_parallel.txt', 'w') as f: f.write(out_py)

def extract_reported_ms(text):
    m = re.search(r'Tiempo.*?:\s*(\d+\.?\d*)', text)
    if m:
        return float(m.group(1))
    try:
        j = json.loads(text)
        if 'time_ms' in j:
            return float(j['time_ms'])
    except:
        pass
    return None

reported_seq = extract_reported_ms(out_seq) or t_seq
reported_omp = extract_reported_ms(out_omp) or t_omp
reported_mpi = extract_reported_ms(out_mpi) or t_mpi
reported_pert_cpp = extract_reported_ms(out_pert_cpp) or t_pert_cpp
reported_py  = extract_reported_ms(out_py)  or t_py

summary_path = '../../results/benchmark_summary.csv'

new_col_name = time.strftime("run_%Y%m%d_%H%M%S")

# Los nuevos tiempos
new_results = {
    'Secuencial (C++)': reported_seq,
    'OpenMP (C++)': reported_omp,
    'MPI (C++)': reported_mpi,
    'PERT (C++)': reported_pert_cpp, 
    'PERT Parallel (Python)': reported_py,
}

# Si NO existe el CSV -> crear nuevo DataFrame
if not os.path.exists(summary_path):
    df = pd.DataFrame({
        'impl': list(new_results.keys()),
        new_col_name: list(new_results.values()),
    })

else:
    df = pd.read_csv(summary_path)

    # Validar consistencia (por si el orden cambia)
    df = df.set_index('impl')

    # Agregar columna con los nuevos tiempos
    df[new_col_name] = pd.Series(new_results).reindex(df.index)

    df = df.reset_index()

print("\n=== RESULTADOS ===")
print(df)

df.to_csv(summary_path, index=False)

latest_run = new_col_name 

df['time_ms'] = df[latest_run]
df['speedup'] = df['time_ms'].iloc[0] / df['time_ms']

print("\n=== RESULTADOS (corrida actual) ===")
print(df[['impl', 'time_ms', 'speedup']])

plt.figure(figsize=(10,5))
sns.barplot(data=df, x='impl', y='time_ms')
plt.xticks(rotation=12)
plt.ylabel('Tiempo (ms)')
plt.tight_layout()
plt.savefig('../../results/benchmark_time.png')
plt.close()

plt.figure(figsize=(10,5))
sns.barplot(data=df, x='impl', y='speedup')
plt.xticks(rotation=12)
plt.ylabel('Speedup')
plt.tight_layout()
plt.savefig('../../results/benchmark_speedup.png')
plt.close()

print("\n=== CONCLUSIÓN AUTOMÁTICA ===")

fastest = df.loc[df['time_ms'].idxmin()]
slowest = df.loc[df['time_ms'].idxmax()]

print(f"- La implementación más rápida fue: **{fastest['impl']}** con {fastest['time_ms']:.2f} ms.")
print(f"- La más lenta fue: **{slowest['impl']}** con {slowest['time_ms']:.2f} ms.")

print("\nSpeedups respecto a la versión secuencial:")
for _, row in df.iterrows():
    print(f"• {row['impl']}: ×{row['speedup']:.2f}")

print("""
En general:
- OpenMP acelera la simulación al paralelizar en memoria compartida.
- MPI ofrece escalamiento distribuido, pero con sobrecosto de comunicación.
- La versión Python PERT, aunque paralela, depende del GIL, overhead de procesos
  y menor velocidad del intérprete.
- C++ sigue siendo la opción más eficiente para Montecarlo intensivo.

Los resultados concretos dependen del hardware (núcleos, RAM, red MPI), pero el
comportamiento relativo suele mantenerse de forma consistente.
""")

print("\nResultados guardados en ../../results/.")