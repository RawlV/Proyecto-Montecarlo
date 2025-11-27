import argparse
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

def alpha_beta_from_pert(a, m, b):
    """
    Convert PERT (a,m,b) to Beta distribution alpha,beta parameters using method of moments.
    Uses commonly used mapping:
      mean = (a + 4m + b)/6
      var approx = ((b-a)/6)^2  (approx used earlier) -> turned to alpha/beta via mean/var.
    Safer approach: use formula from PERT -> Beta parameters:
      alpha = ((mean - a) * (2*mean - a - b)) / var
      beta  = alpha * (b - mean) / (mean - a)
    We'll compute a robust small regularization to avoid negative values.
    """
    mean = (a + 4.0*m + b) / 6.0
    # Use approximate variance (b-a)^2 / 36 as PERT approximation
    var = ((b - a) ** 2) / 36.0
    # avoid degenerate
    eps = 1e-6
    if var <= eps:
        return 2.0, 2.0
    num = (mean - a) * (2 * mean - a - b)
    if num <= 0:
        # fallback to symmetric beta
        return 2.0, 2.0
    alpha = num / var
    beta = alpha * (b - mean) / max(mean - a, eps)
    # clip
    alpha = max(alpha, 0.5)
    beta  = max(beta, 0.5)
    return alpha, beta

def simulate_chunk(seed, n_sim, tareas, plazo_limite, presupuesto_max=None):
    """
    Simulate n_sim MonteCarlo trials using numpy RNG seeded by 'seed'.
    tareas: list of tuples (a,m,b,costo_por_dia)
    Returns: dict {'successes': int, 'n': n_sim}
    """
    rng = np.random.default_rng(seed)
    # Precompute alpha/beta for each task
    alphas = []
    betas  = []
    ranges = []
    costos = []
    for (a,m,b,c) in tareas:
        alpha, beta = alpha_beta_from_pert(a,m,b)
        alphas.append(alpha)
        betas.append(beta)
        ranges.append((a,b))
        costos.append(c)
    alphas = np.array(alphas)
    betas  = np.array(betas)
    ranges = np.array(ranges)
    costos = np.array(costos)

    # For each task generate an array of size (n_sim,) per task using Beta then scale to [a,b]
    # We'll generate per-task arrays and compute totals.
    # Create a 2D array tasks x n_sim
    num_tasks = len(tareas)
    # draw beta variates for each task
    samples = np.empty((num_tasks, n_sim), dtype=np.float64)
    for idx in range(num_tasks):
        a,b = ranges[idx]
        alpha = alphas[idx]
        beta  = betas[idx]
        # numpy beta draws in (0,1) -> scale to [a,b]
        v = rng.beta(alpha, beta, size=n_sim)
        samples[idx, :] = a + v * (b - a)

    # Example topology: tasks 0 and 1 run in parallel; others serial after -> total = max(t0,t1) + sum(t2..)
    # To be general, let's compute: if first two are parallel:
    if num_tasks >= 2:
        parallel_part = np.maximum(samples[0, :], samples[1, :])
        if num_tasks > 2:
            serial_part = samples[2:, :].sum(axis=0)
            total_times = parallel_part + serial_part
        else:
            total_times = parallel_part
    else:
        total_times = samples.sum(axis=0)

    # cost: example costs: cost = sum(task_time * costo_por_dia) for tasks that are time-dependent,
    # plus fixed costs (we assume costos array is per-day multiplier)
    # Here: cost_i = samples[i,:] * costos[i]
    costs = (samples.T * costos).sum(axis=1)  # length n_sim

    # successes = both time and (optionally) budget constraints
    if presupuesto_max is None:
        successes = np.count_nonzero(total_times <= plazo_limite)
    else:
        successes = np.count_nonzero( (total_times <= plazo_limite) & (costs <= presupuesto_max) )

    return {'successes': int(successes), 'n': int(n_sim)}

def worker_args(idx, chunk, base_seed):
    return base_seed + idx * 7919  # arbitrary jump to de-correlate

def run_parallel(total_sim, tareas, plazo, presupuesto, workers, chunk):
    # compute number of chunks per worker
    per_chunk = chunk
    chunks = (total_sim + per_chunk - 1) // per_chunk

    # prepare tasks distribution
    args = []
    remaining = total_sim
    for i in range(chunks):
        n = per_chunk if remaining >= per_chunk else remaining
        remaining -= n
        args.append(n)

    successes_total = 0
    n_total = 0
    base_seed = int(time.time()) & 0xffffffff

    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = []
        for i, n in enumerate(args):
            seed = worker_args(i, n, base_seed)
            futures.append(exe.submit(simulate_chunk, seed, n, tareas, plazo, presupuesto))
        for f in as_completed(futures):
            res = f.result()
            successes_total += res['successes']
            n_total += res['n']
    stop = time.time()
    elapsed_ms = (stop - start) * 1000.0

    prob = successes_total / n_total
    return {
        'successes': successes_total,
        'n': n_total,
        'prob': prob,
        'time_ms': elapsed_ms
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=2000000, help='Total simulations')
    p.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of worker processes')
    p.add_argument('--chunk', type=int, default=100000, help='Chunk size per task')
    p.add_argument('--plazo', type=float, default=30.0, help='Plazo limite (dias)')
    p.add_argument('--presupuesto', type=float, default=None, help='Presupuesto maximo (opcional)')
    return p.parse_args()

def main():
    args = parse_args()
    # define tasks: (a,m,b,costo_por_dia)
    tareas = [
        (8.0, 10.0, 15.0, 1000.0),   # A
        (5.0, 7.0, 12.0, 5000.0),    # B
        (10.0, 14.0, 25.0, 1500.0),  # C
        # add more if needed
    ]

    print("Parallel PERT Monte Carlo")
    print(f"Simulations: {args.n}, workers: {args.workers}, chunk: {args.chunk}")
    res = run_parallel(args.n, tareas, args.plazo, args.presupuesto, workers=args.workers, chunk=args.chunk)
    print("Results:", json.dumps(res, indent=2))

    # Save summary
    out = {
        'config': {
            'n': args.n, 'workers': args.workers, 'chunk': args.chunk, 'plazo': args.plazo, 'presupuesto': args.presupuesto
        },
        'result': res
    }
    os.makedirs('../../results', exist_ok=True)
    with open('../../results/pert_parallel_summary.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()