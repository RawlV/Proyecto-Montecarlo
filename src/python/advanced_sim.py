import subprocess, json, os, re, time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
os.makedirs('../../results', exist_ok=True)

def run_cmd(cmd, shell=False):
    print("Running:", cmd)
    start = time.time()
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        out = e.output
    end = time.time()
    elapsed_ms = (end - start) * 1000.0
    return out, elapsed_ms

# 1) C++ secuencial
out_seq, t_seq = run_cmd('../../results/secuencial')
with open('../../results/secuencial.txt','w') as f: f.write(out_seq)

# 2) OpenMP
out_omp, t_omp = run_cmd('../../results/openmp')
with open('../../results/openmp.txt','w') as f: f.write(out_omp)

# 3) MPI (4 procs)
out_mpi, t_mpi = run_cmd('mpirun -np 4 ../../results/mpi', shell=True)
with open('../../results/mpi.txt','w') as f: f.write(out_mpi)

# 4) Python PERT parallel
out_py, t_py = run_cmd('python3 pert_parallel.py --n 2000000 --workers 12 --chunk 100000', shell=True)
with open('../../results/pert_parallel.txt','w') as f: f.write(out_py)

# Try to extract reported timing from outputs (fallback to measured)
def extract_reported_ms(text):
    m = re.search(r'Tiempo.*?:\s*(\d+\.?\d*)', text)
    if m:
        return float(m.group(1))
    # check JSON from pert summary
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
reported_py  = extract_reported_ms(out_py)  or t_py

df = pd.DataFrame([
    {'impl':'Secuencial (C++)', 'time_ms': reported_seq},
    {'impl':'OpenMP (C++)', 'time_ms': reported_omp},
    {'impl':'MPI (C++)', 'time_ms': reported_mpi},
    {'impl':'PERT Parallel (Python)', 'time_ms': reported_py},
])

df['speedup'] = df['time_ms'].iloc[0] / df['time_ms']

print(df)
df.to_csv('../../results/benchmark_summary.csv', index=False)

# Plotting
plt.figure(figsize=(10,5))
sns.barplot(data=df, x='impl', y='time_ms')
plt.xticks(rotation=15)
plt.ylabel('Time (ms)')
plt.tight_layout()
plt.savefig('../../results/benchmark_time.png')
plt.close()

plt.figure(figsize=(10,5))
sns.barplot(data=df, x='impl', y='speedup')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('../../results/benchmark_speedup.png')
plt.close()

print("Saved results to results/.")