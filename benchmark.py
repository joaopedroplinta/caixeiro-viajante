"""Roda várias seeds por instância em paralelo e mostra melhor/média/pior.

Uso: python benchmark.py [tempo_s] [n_seeds] [instancia ...]
Sem instâncias: roda as TSPLIB com ótimo conhecido.
"""
import contextlib
import io
import logging
import multiprocessing as mp
import os
import statistics
import sys

import tsp

for kv in filter(None, os.environ.get('TSP_CFG', '').split(',')):
    k, v = kv.split('=')
    tsp.CONFIG[k] = type(tsp.CONFIG[k])(v)

ROOT = os.path.dirname(os.path.abspath(__file__))
OPT = {'eil51': 426, 'berlin52': 7542, 'kroA100': 21282, 'eil101': 629,
       'ch150': 6528, 'kroA200': 29368, 'a280': 2579, 'pcb442': 50778,
       'rat783': 8806, 'pr1002': 259045, 'pr2392': 378032}


def run(args):
    path, limit, seed = args
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()):
        return tsp.solve_tsp(path, limit, seed)[1]


if __name__ == "__main__":
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    paths = sys.argv[3:] or [os.path.join(ROOT, 'tsplib', f'{k}.tsp') for k in list(OPT)[:8]]
    # com workers > 1 o próprio solver já usa todos os núcleos: seeds em sequência
    serial = tsp.CONFIG['workers'] > 1
    with (contextlib.nullcontext() if serial else mp.Pool(min(seeds, mp.cpu_count()))) as pool:
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            jobs = [(path, limit, s) for s in range(seeds)]
            res = [int(x) for x in (map(run, jobs) if serial else pool.map(run, jobs))]
            ref = OPT.get(name, min(res))  # sem ótimo conhecido: gap vs melhor visto
            gap = lambda v: (v - ref) / ref * 100
            print(f"{name:9s} melhor {min(res):6d} ({gap(min(res)):5.2f}%)  média {statistics.mean(res):9.1f} "
                  f"({gap(statistics.mean(res)):5.2f}%)  pior {max(res):6d} ({gap(max(res)):5.2f}%)  "
                  f"ótimo {sum(r == ref for r in res)}/{seeds}", flush=True)
