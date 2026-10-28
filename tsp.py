import ctypes
import multiprocessing as mp
import subprocess
import sys
import math
import random
import time
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# NÚCLEO EM C
# Lin-Kernighan + ILS em tsp_core.c, compilado sob demanda. Sem compilador, cai no Python puro.
def _load_core():
    here = os.path.dirname(os.path.abspath(__file__))
    src, lib = os.path.join(here, 'tsp_core.c'), os.path.join(here, 'libtspcore.so')
    try:
        if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
            subprocess.run(['cc', '-O3', '-march=native', '-shared', '-fPIC', src, '-o', lib],
                           check=True, capture_output=True)
        core = ctypes.CDLL(lib)
    except (OSError, subprocess.CalledProcessError):
        return None
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    vp = ctypes.c_void_p  # D (int32 n*n) ou xy (float64 n*2); um deles é None
    core.tsp_knn.restype = None
    core.tsp_knn.argtypes = [ctypes.c_int, vp, ctypes.c_int, ip]
    core.tsp_alpha_candidates.restype = None
    core.tsp_alpha_candidates.argtypes = [ctypes.c_int, vp, vp, ctypes.c_int, ip, ctypes.c_int,
                                          ctypes.c_long, ctypes.c_int, ip]
    core.tsp_solve.restype = ctypes.c_long
    core.tsp_solve.argtypes = [ctypes.c_int, vp, vp, ctypes.c_int, ip, ip, ctypes.c_double,
                               ctypes.c_ulonglong, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_long)]
    return core

CORE = _load_core()

# CONFIGURAÇÕES
CONFIG = {
    'neighbors': 8,         # candidatos por cidade quando alpha = 0
    'pool': 16,             # vizinhos mais próximos entre os quais o alpha escolhe
    'alpha': 6,             # >0: K candidatos por alpha-nearness no lugar dos vizinhos mais próximos
    'alpha_iters': 100,     # iterações do subgradiente (Held-Karp)
    'time_limit': 30.0,     # segundos de ILS (sobrescrito pelo 2º argumento da CLI)
    'kick_window': 50,      # tamanho máximo dos segmentos trocados no double-bridge local
    'workers': mp.cpu_count(),  # processos paralelos que compartilham o melhor tour (1 = serial)
    'sync_chunks': 20,      # nº de rodadas de sincronização por execução
    'accept_temp': 0.1,     # temperatura do aceite do ILS, em fração da aresta média (0 = só não piora)
    'max_segment': 3,       # tamanho máximo do segmento movido pelo Or-opt
}

# RASTREAMENTO DE PROGRESSO
# Monitora a evolução da solução ao longo do tempo.
class ProgressTracker:
    def __init__(self):
        # Inicializa histórico e melhor comprimento.
        self.history = []
        self.best_length = float('inf')
        self.iteration = 0
        
    def update(self, length, stage=""):
        # Registra nova medição e atualiza o melhor comprimento se necessário.
        self.iteration += 1
        if length < self.best_length:
            self.best_length = length
        self.history.append({
            'iteration': self.iteration,
            'length': length,
            'best': self.best_length,
            'stage': stage
        })
        
    def get_gap_history(self, optimal_length):
        # Calcula histórico de gap de otimalidade em relação ao ótimo. 
        if not self.history or optimal_length is None or optimal_length == 0:
            return []
        
        gaps = []
        for entry in self.history:
            gap = ((entry['best'] - optimal_length) / optimal_length) * 100
            gaps.append({
                'iteration': entry['iteration'],
                'gap': max(0.1, gap)
            })
        return gaps

# PARSER TSPLIB
# Utilitário para ler arquivos de instâncias TSP.
class TSPReader:
    @staticmethod
    def _read_tsplib(lines):
        # Formato TSPLIB (EUC_2D): cabeçalho + NODE_COORD_SECTION "id x y".
        nodes, coords, in_coords = [], {}, False
        for line in lines:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
            elif line.startswith("EOF"):
                break
            elif in_coords and line:
                idx, x, y = line.split()[:3]
                nodes.append(int(idx))
                coords[int(idx)] = (float(x), float(y))
        return nodes, coords, "EUC_2D", None, None

    @staticmethod
    def read(filepath):
        # Lê arquivo, extrai nós, coordenadas, pesos e a seed.
        nodes = []
        coords = {}
        weights_flat = []
        seed_value = None
        
        reading_coords = False
        reading_matrix = False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()

        if any(l.strip().startswith("NODE_COORD_SECTION") for l in lines):
            return TSPReader._read_tsplib(lines)

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Coordenadas:"):
                reading_coords = True
                reading_matrix = False
                continue
            elif line.startswith("Matriz de Distâncias:") or line.startswith("Matriz de distâncias:"):
                reading_coords = False
                reading_matrix = True
                continue
            elif line.startswith("Seed"):
                parts = line.split(':')
                if len(parts) > 1:
                    seed_value = parts[1].strip()
                continue

            if reading_coords:
                try:
                    parts = line.split(':')
                    if len(parts) < 2: continue
                    
                    idx = int(parts[0].strip())
                    coord_parts = parts[1].split(',')
                    x = float(coord_parts[0].strip())
                    y = float(coord_parts[1].strip())
                    
                    nodes.append(idx)
                    coords[idx] = (x, y)
                except ValueError:
                    continue

            elif reading_matrix:
                tokens = line.split()
                for token in tokens:
                    try:
                        weights_flat.append(int(float(token)))
                    except ValueError:
                        pass

        n = len(nodes)
        if n == 0:
            raise ValueError("Nenhum nó encontrado no arquivo.")
            
        matrix = [[0]*n for _ in range(n)]
        
        if len(weights_flat) >= n * n:
            for i in range(n):
                for j in range(n):
                    matrix[i][j] = weights_flat[i * n + j]
        else:
            print("Aviso: Matriz incompleta. Usando distâncias euclidianas.")
            weight_type = "EUC_2D"
            return nodes, coords, weight_type, None, seed_value

        return nodes, coords, "EXPLICIT", matrix, seed_value

# CÁLCULO DE DISTÂNCIAS
# Operações matemáticas de distância.
class DistanceCalculator:
    @staticmethod
    def euclidean(p1, p2):
        # Calcula distância Euclidiana 2D arredondada entre dois pontos.
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return int(math.hypot(dx, dy) + 0.5)

    @staticmethod
    def matrix_from_xy(xy):
        diff = xy[:, None, :] - xy[None, :, :]
        return np.floor(np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2) + 0.5).astype(np.int64)

    @staticmethod
    def build_matrix(nodes, coords, weight_type, explicit_weights=None):
        # Constrói matriz de distâncias NxN (explícita ou calculada).
        if weight_type == "EXPLICIT" and explicit_weights:
            return explicit_weights
        return DistanceCalculator.matrix_from_xy(np.array([coords[n] for n in nodes]))

_D = _XY = _NEIGH = None

def _init_worker(D, xy, neigh):
    global _D, _XY, _NEIGH
    _D, _XY, _NEIGH = D, xy, neigh

def _run_chunk(tour, seconds, seed, init):
    # Roda o núcleo C a partir de `tour`; devolve (comprimento, tour, iterações).
    tour = tour.copy()
    iters = ctypes.c_long(0)
    D = None if _D is None else _D.ctypes.data
    xy = _XY.ctypes.data if _D is None else None
    cur = CORE.tsp_solve(len(tour), D, xy, _NEIGH.shape[1], _NEIGH, tour, seconds, seed,
                         CONFIG['kick_window'], CONFIG['accept_temp'], init, ctypes.byref(iters))
    return cur, tour, iters.value

# BUSCA LOCAL
# 2-opt + Or-opt com listas de vizinhos e fila de cidades "sujas" (don't-look bits).
# Trabalha em índices 0..n-1; delta acumulado evita recalcular o comprimento do tour.
class LocalSearch:
    def __init__(self, D, neigh):
        self.D = D          # lista de listas (acesso escalar mais rápido que numpy)
        self.neigh = neigh  # neigh[a] = vizinhos de a por distância crescente
        self.n = len(D)

    @staticmethod
    def build_pos(tour, pos):
        for i, c in enumerate(tour):
            pos[c] = i

    def _reverse(self, tour, pos, i, j):
        # Inverte o caminho de posições i..j (cíclico); inverte o complemento se der a volta.
        if i > j:
            i, j = j + 1, i - 1
        tour[i:j + 1] = tour[i:j + 1][::-1]
        for k in range(i, j + 1):
            pos[tour[k]] = k

    def _try_two_opt(self, a, tour, pos):
        D, n = self.D, self.n
        Da = D[a]
        for forward in (True, False):
            step = 1 if forward else -1
            b = tour[(pos[a] + step) % n]
            dab = Da[b]
            for c in self.neigh[a]:
                if dab - Da[c] <= 0:
                    break
                d = tour[(pos[c] + step) % n]
                if c == b or d == a:
                    continue
                delta = Da[c] + D[b][d] - dab - D[c][d]
                if delta < 0:
                    if forward:
                        self._reverse(tour, pos, pos[b], pos[c])
                    else:
                        self._reverse(tour, pos, pos[a], pos[d])
                    return delta, (a, b, c, d)
        return None

    def _try_or_opt(self, a, tour, pos):
        D, n = self.D, self.n
        Da = D[a]
        i = pos[a]
        for L in range(1, CONFIG['max_segment'] + 1):
            for forward in (True, False):
                step = 1 if forward else -1
                seg = [tour[(i + step * k) % n] for k in range(L)]  # seg[0] == a
                z = seg[-1]
                p = tour[(i - step) % n]
                q = tour[(i + step * L) % n]
                gain = D[p][a] + D[z][q] - D[p][q]
                if gain <= 0:
                    continue
                segset = set(seg)
                for c in self.neigh[a]:
                    if Da[c] >= gain:
                        break
                    if c in segset:
                        continue
                    pc = pos[c]
                    for e in (tour[(pc + 1) % n], tour[pc - 1]):
                        if e in segset:
                            continue
                        delta = Da[c] + D[z][e] - D[c][e] - gain
                        if delta < 0:
                            self._move_segment(tour, pos, seg, forward, c, e)
                            return delta, (p, q, a, z, c, e)
        return None

    def _move_segment(self, tour, pos, seg, forward, c, e):
        # Reinsere seg entre c e e, com seg[0] colado em c.
        n, L = self.n, len(seg)
        s = pos[seg[0]] if forward else pos[seg[-1]]  # posição inicial do segmento no tour
        rest = tour[s + L:] + tour[:s] if s + L <= n else tour[(s + L) % n:s]
        k = rest.index(c)
        if k + 1 < len(rest) and rest[k + 1] == e:
            ins, piece = k + 1, seg
        else:
            ins, piece = k, seg[::-1]
        tour[:] = rest[:ins] + piece + rest[ins:]
        self.build_pos(tour, pos)

    def run(self, tour, pos, dirty):
        # Melhora o tour in-place a partir das cidades em `dirty`; devolve o delta total.
        inq = [False] * self.n
        stack = []
        for c in dirty:
            if not inq[c]:
                inq[c] = True
                stack.append(c)
        total = 0
        while stack:
            a = stack.pop()
            inq[a] = False
            while True:
                move = self._try_two_opt(a, tour, pos) or self._try_or_opt(a, tour, pos)
                if not move:
                    break
                delta, touched = move
                total += delta
                for c in touched:
                    if not inq[c]:
                        inq[c] = True
                        stack.append(c)
        return total

# CONSTRUÇÃO E BUSCA ITERADA
class Solver:
    def __init__(self, rng, xy, D=None):
        # D: matriz de distâncias; None = o núcleo C calcula EUC_2D direto das coordenadas.
        self.xy = np.ascontiguousarray(xy, dtype=np.float64)
        self.n = len(self.xy)
        self.rng = rng
        if D is None and not CORE:
            D = DistanceCalculator.matrix_from_xy(self.xy)
        self.D = None if D is None else np.asarray(D)
        self.D32 = None if D is None else np.ascontiguousarray(self.D, dtype=np.int32)
        self.start_tour = None  # tour já passado pela busca local (preenchido ao calcular o alpha)
        self.neigh = self._candidates()
        if not CORE:  # a lista de listas só serve ao fallback Python (e custa ~n² objetos)
            self.ls = LocalSearch(self.D.tolist(), self.neigh.tolist())

    def _ptrs(self):
        return (None if self.D32 is None else self.D32.ctypes.data), (self.xy.ctypes.data if self.D32 is None else None)

    def _candidates(self):
        n = self.n
        pool = min(CONFIG['pool'], n - 1)
        if self.D32 is None:
            cand = np.empty((n, pool), dtype=np.int32)
            CORE.tsp_knn(n, self.xy.ctypes.data, pool, cand)
        else:
            masked = self.D + np.diag(np.full(n, np.iinfo(self.D.dtype).max // 2))
            cand = np.ascontiguousarray(np.argsort(masked, axis=1, kind='stable')[:, :pool], dtype=np.int32)
        if CORE and CONFIG['alpha'] > 0 and n >= 8:
            return self._alpha_neighbors(cand, min(CONFIG['alpha'], pool))
        return np.ascontiguousarray(cand[:, :min(CONFIG['neighbors'], pool)])

    def _alpha_neighbors(self, cand, k):
        # Limite superior = tour já melhorado pela busca local; guia o passo do subgradiente.
        D, xy = self._ptrs()
        tour = self.initial_tour()
        near = np.ascontiguousarray(cand[:, :CONFIG['neighbors']])
        ub = CORE.tsp_solve(self.n, D, xy, near.shape[1], near, tour, 0.0, 1,
                            CONFIG['kick_window'], 0.0, 1, ctypes.byref(ctypes.c_long(0)))
        self.start_tour = tour
        out = np.zeros((self.n, k), dtype=np.int32)
        CORE.tsp_alpha_candidates(self.n, D, xy, cand.shape[1], cand, k, ub, CONFIG['alpha_iters'], out)
        return out

    def initial_tour(self):
        # Curva de Hilbert: O(n log n); a busca local corrige o resto.
        xy = self.xy
        span = max(float(np.ptp(xy, axis=0).max()), 1e-9)
        g = ((xy - xy.min(axis=0)) / span * 65535).astype(np.int64)
        x, y = g[:, 0].copy(), g[:, 1].copy()
        d = np.zeros(self.n, dtype=np.int64)
        s = 1 << 15
        while s:
            rx, ry = (x & s) > 0, (y & s) > 0
            d += s * s * ((3 * rx) ^ ry)
            refl = rx & ~ry
            x, y = np.where(refl, 65535 - x, x), np.where(refl, 65535 - y, y)
            x, y = np.where(~ry, y, x), np.where(~ry, x, y)
            s >>= 1
        return np.argsort(d, kind='stable').astype(np.int32)

    def length(self, tour):
        D = self.ls.D
        return sum(D[tour[i - 1]][tour[i]] for i in range(self.n))

    def _kick(self, tour):
        # Double-bridge local: troca dois segmentos adjacentes numa janela pequena.
        n, D = self.n, self.ls.D
        w = max(1, min(CONFIG['kick_window'], (n - 2) // 3))
        r = self.rng.randrange(n)
        l1, l2 = self.rng.randint(1, w), self.rng.randint(1, w)
        t = tour[r:] + tour[:r]
        a, b1, b2, c1, c2, d = t[0], t[1], t[l1], t[l1 + 1], t[l1 + l2], t[(l1 + l2 + 1) % n]
        delta = (D[a][c1] + D[c2][b1] + D[b2][d]) - (D[a][b1] + D[b2][c1] + D[c2][d])
        tour[:] = [a] + t[l1 + 1:l1 + l2 + 1] + t[1:l1 + 1] + t[l1 + l2 + 1:]
        return delta, (a, b1, b2, c1, c2, d)

    def solve(self, time_limit, tracker=None):
        return self._solve_c(time_limit, tracker) if CORE else self._solve_python(time_limit, tracker)

    def _solve_c(self, time_limit, tracker=None):
        # Rodadas curtas: em cada uma, os workers partem do melhor tour com seeds diferentes.
        chunks, workers = CONFIG['sync_chunks'], max(1, CONFIG['workers'])
        neigh = np.ascontiguousarray(self.neigh, dtype=np.int32)
        _init_worker(self.D32, self.xy, neigh)
        if self.start_tour is not None:
            tour = self.start_tour
            cur = self.tour_length_c(tour)
        else:  # busca local completa uma vez, no processo principal
            cur, tour, _ = _run_chunk(self.initial_tour(), 0.0, 0, 1)
        if tracker:
            tracker.update(cur, "LK")
        pool = mp.get_context('fork').Pool(workers) if workers > 1 and 'fork' in mp.get_all_start_methods() else None
        total = 0
        try:
            for _ in range(chunks):
                jobs = [(tour, time_limit / chunks, self.rng.getrandbits(32), 0) for _ in range(workers)]
                results = pool.starmap(_run_chunk, jobs) if pool else [_run_chunk(*jobs[0])]
                cur, tour, _ = min(results, key=lambda r: r[0])
                total += sum(r[2] for r in results)
                if tracker:
                    tracker.update(cur, "LK + ILS")
        finally:
            if pool:
                pool.terminate()
        logger.info(f"LK + ILS (C, {workers} workers): {total} iterações, comprimento {cur}")
        return tour.tolist(), cur

    def tour_length_c(self, tour):
        # Comprimento via núcleo (time_limit 0, sem busca local): mesma função de distância do LK.
        return _run_chunk(tour, 0.0, 0, 0)[0]

    def _solve_python(self, time_limit, tracker=None):
        tour = self.initial_tour().tolist()
        pos = [0] * self.n
        LocalSearch.build_pos(tour, pos)
        cur = self.length(tour)
        logger.info(f"Vizinho mais próximo: {cur}")
        cur += self.ls.run(tour, pos, tour[:])
        logger.info(f"2-opt + Or-opt: {cur}")
        if tracker:
            tracker.update(cur, "busca local")

        deadline = time.time() + time_limit
        iters = 0
        while time.time() < deadline and self.n >= 8:
            iters += 1
            old_tour, old_pos = tour[:], pos[:]
            delta, touched = self._kick(tour)
            LocalSearch.build_pos(tour, pos)
            delta += self.ls.run(tour, pos, touched)
            if delta <= 0:
                cur += delta
                if delta < 0 and tracker:
                    tracker.update(cur, "ILS")
            else:
                tour[:], pos[:] = old_tour, old_pos
        logger.info(f"ILS: {iters} iterações, comprimento {cur}")
        return tour, cur


# VISUALIZAÇÃO
# Plota o tour e o gráfico de convergência.
def plot_final_result(tour, coords, final_length, gap_history, filename="tsp_result.png"):
    fig = plt.figure(figsize=(16, 7))
    
    # Gráfico 1: Rota do TSP
    ax1 = plt.subplot(1, 2, 1)
    
    # Plota o tour (já fechado: primeiro == último) numa única linha; nós só em instâncias pequenas
    pts = np.array([coords[n] for n in tour if n in coords])
    ax1.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1, alpha=0.6)
    if len(pts) <= 3000:
        ax1.scatter(pts[:, 0], pts[:, 1], c='blue', s=20, zorder=5)

    ax1.set_title(f'Melhor rota encontrada: {final_length:.2f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coordenada X', fontsize=12)
    ax1.set_ylabel('Coordenada Y', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Gráfico 2: Gap de Otimalidade
    ax2 = plt.subplot(1, 2, 2)
    
    if gap_history:
        iterations = [entry['iteration'] for entry in gap_history]
        gaps = [entry['gap'] for entry in gap_history]
        
        ax2.plot(iterations, gaps, 'g-', linewidth=2, label='Gap')
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='1% Gap')
        ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='0.1% Gap')
        
        final_gap = gaps[-1]
        ax2.set_yscale('log')
        ax2.set_title(f'Gap em relação à melhor solução (Final: {final_gap:.2f}%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteração', fontsize=12)
        ax2.set_ylabel('Gap (%)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, which='both')
    else:
        ax2.text(0.5, 0.5, 'Sem histórico de gap disponível', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Gap em relação à melhor solução', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo em: {filename}")
    plt.show()


# EXECUÇÃO PRINCIPAL
def solve_tsp(filepath, time_limit=None, seed=None):
    nodes, coords, weight_type, explicit_weights, file_seed = TSPReader.read(filepath)
    xy = np.array([coords[n] for n in nodes], dtype=np.float64)
    # Matriz só se vier no arquivo; senão o núcleo C calcula EUC_2D direto das coordenadas.
    dist_matrix = explicit_weights

    # seed explícita > seed do arquivo > 0
    try:
        rng = random.Random(int(seed if seed is not None else file_seed))
    except (TypeError, ValueError):
        rng = random.Random(0)

    tracker = ProgressTracker()
    solver = Solver(rng, xy, dist_matrix)
    tour, final_length = solver.solve(time_limit or CONFIG['time_limit'], tracker)

    # Converte índices de volta para rótulos e fecha o ciclo
    best_tour = [nodes[i] for i in tour]
    best_tour.append(best_tour[0])

    print(f"\n{'='*60}")
    print(f"FINAL LENGTH: {int(final_length)}")
    print(f"{'='*60}")

    return best_tour, final_length, nodes, coords, dist_matrix, tracker, file_seed


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Uso: python tsp.py instancia.txt [tempo_ils_segundos]")
        sys.exit(1)

    start = time.time()
    
    best_tour, final_length, nodes, coords, dist_matrix, tracker, seed = solve_tsp(sys.argv[1], float(sys.argv[2]) if len(sys.argv) == 3 else None)
    
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    print(f"Tempo de execução: {time_str}")

    # Gera arquivo de saída
    base_name = os.path.splitext(sys.argv[1])[0]
    output_filename = base_name + "_solucao.txt"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            seed_str = seed if seed is not None else "None"
            f.write(f"Seed utilizada:{seed_str}\n\n")
            
            f.write("Coordenadas:\n")
            for n in nodes:
                if n in coords:
                    x, y = coords[n]
                    f.write(f"{n}: {x:.4f}, {y:.4f}\n")
            
            f.write("\n\n")
            
            f.write("MELHOR SOLUÇÃO ENCONTRADA\n")
            f.write("==========================\n")
            f.write(f"Distância total: {final_length:.4f}\n")
            f.write(f"Tempo de Execução: {time_str}\n\n")
            
            f.write("Rota (arestas no formato i -> j):\n")
            if best_tour:
                for i in range(len(best_tour) - 1):
                    u = best_tour[i]
                    v = best_tour[i+1]
                    f.write(f"{u} -> {v}\n")
        
        print(f"\nArquivo de saída gerado: {output_filename}")
    except Exception as e:
        print(f"\nErro ao gerar arquivo de saída: {e}")

    # Gera visualização
    gap_history = tracker.get_gap_history(final_length)
    plot_filename = base_name + "_grafico.png"
    plot_final_result(best_tour, coords, final_length, gap_history, plot_filename)