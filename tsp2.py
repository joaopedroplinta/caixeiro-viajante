import sys
import math
import heapq
import random
import multiprocessing as mp
import time
from itertools import combinations
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURAÇÕES OTIMIZADAS
CONFIG = {
    'max_clusters': 500000,
    'nodes_per_cluster': 50,  # Aumentado de 30 para 50 - menos clusters
    'infinity': 1e9,
    'tabu': {
        'max_iterations': 100,  # Reduzido de 200 para 100
        'tenure': 5,
        'no_improve_limit': 30,  # Reduzido de 500 para 30
        'log_frequency': 50,
        'min_tenure': 1,
        'max_tenure': 50,  # Reduzido de 500 para 50
        'tenure_update_freq': 5
    },
    'opt3': {
        'rounds': 3,  # Reduzido de 10 para 3
        'max_segment': 50,  # Reduzido de 75 para 50
        'workers': max(4, mp.cpu_count() - 1)  # Mais eficiente
    },
    'perturbation': {
        'count': 20,  # Reduzido de 100 para 20
        'types': [0, 1, 2],  # Removido tipos 4 e 5 (não existiam)
        'cost_guided': True,
        'between_rounds': False,  # Desabilitado para economizar tempo
        'rounds_count': 10  # Reduzido de 100 para 10
    },
    'optimization_rounds': 1,  # Reduzido de 2 para 1
}

# RASTREAMENTO DE PROGRESSO
class ProgressTracker:
    def __init__(self):
        self.history = []
        self.best_length = float('inf')
        self.iteration = 0
        
    def update(self, length, stage=""):
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
class TSPReader:
    @staticmethod
    def read(filepath):
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
class DistanceCalculator:
    @staticmethod
    def euclidean(p1, p2):
        dx, dy = p1[0] - p2[0], p1[1] - p2[1]
        return int(round(math.hypot(dx, dy)))

    @staticmethod
    def build_matrix(nodes, coords, weight_type, explicit_weights=None):
        if weight_type == "EXPLICIT" and explicit_weights:
            return explicit_weights

        n = len(nodes)
        # Usar numpy para cálculo mais rápido
        node_coords = np.array([coords[node] for node in nodes])
        
        # Cálculo vetorizado de distâncias
        diff = node_coords[:, np.newaxis, :] - node_coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        matrix = np.round(distances).astype(int).tolist()
        
        return matrix

# CLUSTERIZAÇÃO
class Clustering:
    @staticmethod
    def create_clusters(nodes, coords):
        valid = {n: c for n, c in coords.items()
                 if c[0] != float('inf') and c[1] != float('inf')}

        if not valid:
            return Clustering._simple_split(nodes)

        xs = [c[0] for c in valid.values()]
        ys = [c[1] for c in valid.values()]
        clusters = []

        def quad_divide(node_list, x_min, x_max, y_min, y_max, depth=0):
            if len(node_list) <= CONFIG['nodes_per_cluster'] or depth > 8:  # Reduzida profundidade máxima
                if node_list and len(clusters) < CONFIG['max_clusters']:
                    clusters.append(node_list)
                return

            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            quadrants = [[], [], [], []]

            for n in node_list:
                x, y = valid.get(n, (x_min, y_min))
                quad = (0 if x < x_mid else 1) + (0 if y < y_mid else 2)
                quadrants[quad].append(n)

            ranges = [
                (x_min, x_mid, y_min, y_mid),
                (x_mid, x_max, y_min, y_mid),
                (x_min, x_mid, y_mid, y_max),
                (x_mid, x_max, y_mid, y_max)
            ]

            for quad, (xmin, xmax, ymin, ymax) in zip(quadrants, ranges):
                if quad:
                    quad_divide(quad, xmin, xmax, ymin, ymax, depth+1)

        quad_divide(nodes, min(xs), max(xs), min(ys), max(ys))
        return clusters[:CONFIG['max_clusters']]

    @staticmethod
    def _simple_split(nodes):
        size = CONFIG['nodes_per_cluster']
        return [nodes[i:i+size] for i in range(0, len(nodes), size)]

    @staticmethod
    def find_border_nodes(cluster, coords, index_map):
        valid = [n for n in cluster if n in index_map and n in coords]
        if not valid:
            return cluster[0], cluster[0]

        valid_coords = {n: coords[n] for n in valid
                        if coords[n][0] != float('inf')}
        if not valid_coords:
            return valid[0], valid[0]

        start = min(valid_coords, key=lambda n: (valid_coords[n][1], valid_coords[n][0]))
        end = max(valid_coords, key=lambda n: (-valid_coords[n][1], valid_coords[n][0]))
        return start, end

# CONSTRUÇÃO DE TOUR
class TourBuilder:
    @staticmethod
    def nearest_neighbor(cluster, start, dist_matrix, index_map):
        """Heurística nearest neighbor - MUITO mais rápida que Dijkstra"""
        if len(cluster) <= 1 or start not in index_map:
            return cluster + [cluster[0]] if cluster else []
        
        unvisited = set(cluster)
        tour = [start]
        unvisited.remove(start)
        
        current = start
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda x: dist_matrix[index_map[current]][index_map[x]] if x in index_map else float('inf'))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        tour.append(start)
        return tour

    @staticmethod
    def calculate_length(tour, dist_matrix, index_map):
        if not tour or len(tour) < 2:
            return CONFIG['infinity']

        total = 0
        for i in range(len(tour)):
            u, v = tour[i], tour[(i+1) % len(tour)]
            if u not in index_map or v not in index_map:
                return CONFIG['infinity']
            total += dist_matrix[index_map[u]][index_map[v]]
        return total

# OTIMIZAÇÕES LOCAIS
class LocalOptimizer:
    @staticmethod
    def two_opt(tour, dist_matrix, idx_map, tracker=None, label="2-opt", max_iter=1000):
        """2-opt com limite de iterações e early stopping"""
        if len(tour) < 4:
            return tour[:]

        best = tour[:]
        n = len(tour)
        iteration = 0
        no_improve = 0
        
        while iteration < max_iter and no_improve < 50:  # Early stopping
            iteration += 1
            improved = False
            
            for i in range(n-1):
                if no_improve >= 50:
                    break
                    
                for j in range(i+2, min(i+50, n if i > 0 else n-1)):  # Limitar janela de busca
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]

                    if all(x in idx_map for x in [a,b,c,d]):
                        delta = (dist_matrix[idx_map[a]][idx_map[c]] +
                                 dist_matrix[idx_map[b]][idx_map[d]] -
                                 dist_matrix[idx_map[a]][idx_map[b]] -
                                 dist_matrix[idx_map[c]][idx_map[d]])

                        if delta < -0.001:  # Threshold para evitar trocas triviais
                            tour[i+1:j+1] = reversed(tour[i+1:j+1])
                            improved = True
                            no_improve = 0
                            if tracker:
                                length = TourBuilder.calculate_length(tour, dist_matrix, idx_map)
                                tracker.update(length, label)
                            best = tour[:]
            
            if not improved:
                no_improve += 1

        return best

    @staticmethod
    def tabu_search(tour, dist_matrix, idx_map, tracker, label="tabu"):
        """Busca Tabu simplificada e mais rápida"""
        best = tour[:]
        best_len = TourBuilder.calculate_length(best, dist_matrix, idx_map)
        tabu_list = {}
        no_improve = 0
        n = len(tour)
        tenure = CONFIG['tabu']['tenure']

        for iteration in range(1, CONFIG['tabu']['max_iterations']+1):
            best_move = None
            best_delta = CONFIG['infinity']

            # Amostragem aleatória para acelerar
            candidates = random.sample(range(n-1), min(n-1, 100))
            
            for i in candidates:
                for j in range(i+2, min(i+30, n if i > 0 else n-1)):  # Janela reduzida
                    a, b = tour[i], tour[(i+1) % n]
                    c, d = tour[j], tour[(j+1) % n]

                    if all(x in idx_map for x in [a,b,c,d]):
                        if (a,c) in tabu_list and tabu_list[(a,c)] > iteration:
                            continue

                        delta = (dist_matrix[idx_map[a]][idx_map[c]] +
                                 dist_matrix[idx_map[b]][idx_map[d]] -
                                 dist_matrix[idx_map[a]][idx_map[b]] -
                                 dist_matrix[idx_map[c]][idx_map[d]])

                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)

            if not best_move:
                break

            i, j = best_move
            tour[i+1:j+1] = reversed(tour[i+1:j+1])
            tabu_list[(tour[i], tour[i+1])] = iteration + tenure

            current_len = TourBuilder.calculate_length(tour, dist_matrix, idx_map)
            if current_len < best_len:
                best = tour[:]
                best_len = current_len
                no_improve = 0
                if tracker:
                    tracker.update(best_len, label)
            else:
                no_improve += 1

            if no_improve >= CONFIG['tabu']['no_improve_limit']:
                break

        return best

# 3-OPT OTIMIZADO
class ThreeOpt:
    @staticmethod
    def _calculate_delta(tour, dist_matrix, idx_map, i, j, k):
        try:
            A, B = tour[i-1], tour[i]
            C, D = tour[j-1], tour[j]
            E, F = tour[k-1], tour[k]

            if any(n not in idx_map for n in [A, B, C, D, E, F]):
                return 0, 0

            cost_original = (dist_matrix[idx_map[A]][idx_map[B]] +
                             dist_matrix[idx_map[C]][idx_map[D]] +
                             dist_matrix[idx_map[E]][idx_map[F]])

            best_delta = 0
            best_option = 0

            # Apenas as 2 recombinações mais promissoras
            options = [
                (A, C, B, D, E, F),
                (A, B, C, E, D, F),
            ]

            for opt_num, (n1, n2, n3, n4, n5, n6) in enumerate(options, 1):
                cost = (dist_matrix[idx_map[n1]][idx_map[n2]] +
                        dist_matrix[idx_map[n3]][idx_map[n4]] +
                        dist_matrix[idx_map[n5]][idx_map[n6]])
                delta = cost - cost_original
                if delta < best_delta:
                    best_delta = delta
                    best_option = opt_num

            return best_delta, best_option
        except:
            return 0, 0

    @staticmethod
    def _apply_swap(tour, i, j, k, option):
        segment1 = tour[:i]
        segment2 = tour[i:j]
        segment3 = tour[j:k]
        segment4 = tour[k:]

        if option == 0:
            return tour
        elif option == 1:
            return segment1 + segment2[::-1] + segment3 + segment4
        elif option == 2:
            return segment1 + segment2 + segment3[::-1] + segment4
        return tour

    @staticmethod
    def _evaluate_triplet_chunk(args):
        tour, dist_matrix, idx_map, triplets = args
        base_len = TourBuilder.calculate_length(tour, dist_matrix, idx_map)

        for i, j, k in triplets:
            best_delta, best_option = ThreeOpt._calculate_delta(tour, dist_matrix, idx_map, i, j, k)
            if best_delta < -0.001:  # Threshold
                new_tour = ThreeOpt._apply_swap(tour, i, j, k, best_option)
                new_len = TourBuilder.calculate_length(new_tour, dist_matrix, idx_map)
                if new_len < base_len:
                    return new_tour, new_len

        return None, None

    @staticmethod
    def optimize(tour, dist_matrix, idx_map, tracker, label="3-opt"):
        best_tour = tour[:]
        best_length = TourBuilder.calculate_length(best_tour, dist_matrix, idx_map)
        n = len(best_tour)

        for round_num in range(1, CONFIG['opt3']['rounds'] + 1):
            # Amostragem aleatória de triplets
            all_triplets = []
            step = max(1, n // 100)  # Amostragem mais esparsa
            
            for i in range(1, n-2, step):
                for j in range(i+1, min(i+CONFIG['opt3']['max_segment'], n-1), step):
                    for k in range(j+1, min(j+CONFIG['opt3']['max_segment'], n), step):
                        all_triplets.append((i, j, k))
            
            if not all_triplets:
                break
            
            # Limitar número total de triplets
            if len(all_triplets) > 1000:
                all_triplets = random.sample(all_triplets, 1000)

            chunk_size = max(1, len(all_triplets) // CONFIG['opt3']['workers'])
            chunks = [all_triplets[i:i+chunk_size] for i in range(0, len(all_triplets), chunk_size)]
            args_list = [(best_tour[:], dist_matrix, idx_map, chunk) for chunk in chunks]

            try:
                with mp.Pool(CONFIG['opt3']['workers']) as pool:
                    results = pool.map(ThreeOpt._evaluate_triplet_chunk, args_list)
            except:
                return best_tour

            found_improvement = None
            best_in_round = best_length
            for new_tour, new_len in results:
                if new_tour is not None and new_len < best_in_round:
                    best_in_round = new_len
                    found_improvement = new_tour

            if found_improvement and best_in_round < best_length:
                best_tour = found_improvement[:]
                best_length = best_in_round
                if tracker:
                    tracker.update(best_length, label)
            else:
                break

        return best_tour

# PERTURBAÇÕES
class Perturbation:
    @staticmethod
    def perturb(tour, strategy, idx_map, high_cost_edges=None):
        perturbed = tour[:]
        n = len(tour)
        if n < 5:
            return perturbed

        if strategy == 0:  # Double-Bridge
            indices = sorted(random.sample(range(n-1), min(4, n-1)))
            if len(indices) == 4:
                i1, i2, i3, i4 = indices
                if i2-i1 >= 1 and i4-i3 >= 1 and i3-i2 >= 1:
                    perturbed = (tour[:i1+1] + tour[i3+1:i4+1] +
                               tour[i2+1:i3+1] + tour[i1+1:i2+1] + tour[i4+1:])

        elif strategy == 1:  # Multi 2-opt
            temp = tour[:]
            for _ in range(random.randint(2, 3)):  # Reduzido
                i = random.randint(0, n - 2)
                j = random.randint(i + 2, n - 1)
                if j - i >= 2:
                    temp[i+1:j+1] = temp[i+1:j+1][::-1]
            perturbed = temp

        elif strategy == 2:  # Embaralhamento
            segment_size = max(2, n // 20)
            start = random.randint(0, max(0, n - segment_size - 1))
            segment = perturbed[start:start+segment_size]
            random.shuffle(segment)
            perturbed = perturbed[:start] + segment + perturbed[start+segment_size:]

        return perturbed

    @staticmethod
    def evaluate_perturbation_worker(args):
        tour, dist_matrix, idx_map, coords, strategy, iteration, high_cost_edges = args
        try:
            perturbed = Perturbation.perturb(tour, strategy, idx_map, high_cost_edges)
            perturbed = TourUtils.remove_subtours(perturbed, dist_matrix, idx_map, coords)
            perturbed = LocalOptimizer.two_opt(perturbed, dist_matrix, idx_map, None, "", max_iter=200)
            length_optimized = TourBuilder.calculate_length(perturbed, dist_matrix, idx_map)
            return perturbed, length_optimized, iteration, strategy
        except:
            return tour[:], CONFIG['infinity'], iteration, strategy

    @staticmethod
    def apply_multiple(tour, dist_matrix, idx_map, coords, tracker, num_perturbations):
        original_len = TourBuilder.calculate_length(tour, dist_matrix, idx_map)
        best_tour = tour[:]
        best_len = original_len

        tasks = []
        for iter_num in range(num_perturbations):
            strategy = random.choice(CONFIG['perturbation']['types'])
            tasks.append((tour[:], dist_matrix, idx_map, coords, strategy, iter_num, []))

        try:
            with mp.Pool(CONFIG['opt3']['workers']) as pool:
                results = pool.map(Perturbation.evaluate_perturbation_worker, tasks)
        except:
            results = [(tour[:], original_len, i, 0) for i in range(num_perturbations)]

        for perturbed_tour, length, iter_num, strategy in results:
            if length < best_len and length != CONFIG['infinity']:
                best_tour = perturbed_tour[:]
                best_len = length
                if tracker:
                    tracker.update(best_len, "perturbation")

        if best_len >= original_len:
            best_tour = tour[:]

        return best_tour

# UTILITÁRIOS
class TourUtils:
    @staticmethod
    def remove_subtours(tour, dist_matrix, idx_map, coords):
        n = len(tour)
        if n <= 1:
            return tour

        adjacency = {}
        for i in range(n):
            u, v = tour[i], tour[(i+1) % n]
            if u in idx_map and v in idx_map:
                adjacency.setdefault(u, []).append(v)
                adjacency.setdefault(v, []).append(u)

        visited = set()
        components = []

        def dfs_component(start):
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    component.append(node)
                    for neighbor in adjacency.get(node, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            return component

        for node in tour:
            if node not in visited and node in idx_map:
                comp = dfs_component(node)
                if comp:
                    components.append(comp)

        if len(components) <= 1:
            return tour

        merged = components[0]
        for comp in components[1:]:
            merged = merged[:-1] + comp + [merged[0]]

        return merged

    @staticmethod
    def merge_tours(tour1, tour2, dist_matrix, idx_map):
        if not tour1 or not tour2:
            return tour1 if tour1 else tour2 if tour2 else []

        t1, t2 = tour1[:-1], tour2[:-1]
        best_gain = -CONFIG['infinity']
        best_connection = None

        # Amostragem para acelerar merge
        sample_size = min(20, len(t1), len(t2))
        indices1 = random.sample(range(len(t1)), min(sample_size, len(t1)))
        indices2 = random.sample(range(len(t2)), min(sample_size, len(t2)))

        for i in indices1:
            u1, v1 = t1[i], t1[(i+1) % len(t1)]
            if u1 not in idx_map or v1 not in idx_map:
                continue

            for j in indices2:
                u2, v2 = t2[j], t2[(j+1) % len(t2)]
                if u2 not in idx_map or v2 not in idx_map:
                    continue

                current_cost = (dist_matrix[idx_map[u1]][idx_map[v1]] +
                              dist_matrix[idx_map[u2]][idx_map[v2]])

                gain1 = current_cost - (dist_matrix[idx_map[u1]][idx_map[u2]] +
                                     dist_matrix[idx_map[v1]][idx_map[v2]])
                if gain1 > best_gain:
                    best_gain = gain1
                    best_connection = (i, j, False)

        if best_connection is None:
            return t1 + t2 + [t1[0]]

        i, j, reverse = best_connection
        t2_reordered = t2[j+1:] + t2[:j+1]
        if reverse:
            t2_reordered.reverse()

        return t1[:i+1] + t2_reordered + t1[i+1:] + [t1[0]]

# VISUALIZAÇÃO
def plot_final_result(tour, coords, final_length, gap_history, filename="tsp_result.png"):
    fig = plt.figure(figsize=(16, 7))
    
    ax1 = plt.subplot(1, 2, 1)
    
    for i in range(len(tour)):
        u, v = tour[i], tour[(i+1) % len(tour)]
        if u in coords and v in coords:
            x1, y1 = coords[u]
            x2, y2 = coords[v]
            ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.6)
    
    xs = [coords[node][0] for node in tour if node in coords]
    ys = [coords[node][1] for node in tour if node in coords]
    ax1.scatter(xs, ys, c='blue', s=20, zorder=5)
    
    ax1.set_title(f'Melhor rota encontrada: {final_length:.2f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coordenada X', fontsize=12)
    ax1.set_ylabel('Coordenada Y', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    ax2 = plt.subplot(1, 2, 2)
    
    if gap_history:
        iterations = [entry['iteration'] for entry in gap_history]
        gaps = [entry['gap'] for entry in gap_history]
        
        ax2.plot(iterations, gaps, 'g-', linewidth=2, label='Gap')
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='1% Gap')
        ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='0.1% Gap')
        
        final_gap = gaps[-1]
        ax2.set_yscale('log')
        ax2.set_title(f'Gap de Otimalidade (Final: {final_gap:.2f}%)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteração', fontsize=12)
        ax2.set_ylabel('Gap (%)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, which='both')
    else:
        ax2.text(0.5, 0.5, 'Sem histórico de gap disponível', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Gap de Otimalidade', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo em: {filename}")
    plt.show()

# EXECUÇÃO PRINCIPAL
def solve_tsp(filepath):
    nodes, coords, weight_type, explicit_weights, seed = TSPReader.read(filepath)
    dist_matrix = DistanceCalculator.build_matrix(nodes, coords, weight_type, explicit_weights)
    idx_map = {n: i for i, n in enumerate(nodes)}

    tracker = ProgressTracker()
    
    clusters = Clustering.create_clusters(nodes, coords)
    logger.info(f"Created {len(clusters)} clusters")

    cluster_tours = []
    border_nodes = {}

    # MUDANÇA CRÍTICA: Usar Nearest Neighbor em vez de Dijkstra
    for cid, cluster in enumerate(clusters):
        if not cluster:
            continue

        start, end = Clustering.find_border_nodes(cluster, coords, idx_map)
        if start is None or start not in idx_map:
            continue

        border_nodes[cid] = (start, end)

        # Nearest Neighbor é 10-20x mais rápido que Dijkstra+DFS
        tour = TourBuilder.nearest_neighbor(cluster, start, dist_matrix, idx_map)

        # 2-opt mais rápido com limite de iterações
        tour = LocalOptimizer.two_opt(tour, dist_matrix, idx_map, tracker, f"2-opt cluster {cid+1}", max_iter=300)
        
        if len(tour) >= 2 and tour[-1] != tour[0]:
            tour.append(tour[0])

        cluster_tours.append(tour)

    if not cluster_tours:
        logger.warning("No valid tours created")
        tour = list(nodes) + [nodes[0]] if nodes else []
        cluster_tours = [tour]

    logger.info(f"Starting merge with {len(cluster_tours)} tours")

    # Merge de clusters
    current_tours = [t for t in cluster_tours if t and len(t) >= 2]
    tour_indices = list(range(len(current_tours)))
    iteration = 1

    while len(current_tours) > 1:
        logger.info(f"Merge iteration {iteration}, {len(current_tours)} tours remaining")

        min_dist = CONFIG['infinity']
        pair = (0, 1)

        for i in range(len(current_tours)):
            for j in range(i+1, len(current_tours)):
                ti, tj = tour_indices[i], tour_indices[j]
                if ti not in border_nodes or tj not in border_nodes:
                    continue

                u = border_nodes[ti][1]
                v = border_nodes[tj][0]
                if u in idx_map and v in idx_map:
                    d = dist_matrix[idx_map[u]][idx_map[v]]
                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)

        i, j = pair
        ti, tj = tour_indices[i], tour_indices[j]
        logger.info(f"Merging clusters {ti} and {tj}")

        tour1, tour2 = current_tours[i], current_tours[j]
        merged_tour = TourUtils.merge_tours(tour1, tour2, dist_matrix, idx_map)
        merged_tour = TourUtils.remove_subtours(merged_tour, dist_matrix, idx_map, coords)

        merged_tour = LocalOptimizer.two_opt(merged_tour, dist_matrix, idx_map, tracker, f"2-opt iter {iteration}", max_iter=500)
        if len(merged_tour) >= 2 and merged_tour[-1] != merged_tour[0]:
            merged_tour.append(merged_tour[0])

        combined_nodes = list(set(tour1[:-1] + tour2[:-1]))
        if combined_nodes:
            start_node, end_node = Clustering.find_border_nodes(combined_nodes, coords, idx_map)
            if start_node and end_node:
                new_index = len(current_tours)
                border_nodes[new_index] = (start_node, end_node)

        current_tours.append(merged_tour)
        current_tours.pop(max(i, j))
        current_tours.pop(min(i, j))
        tour_indices = list(range(len(current_tours)))

        iteration += 1

    combined_tour = current_tours[0] if current_tours else []
    
    # Otimização final
    tour = LocalOptimizer.two_opt(combined_tour, dist_matrix, idx_map, tracker, "2-opt final", max_iter=1000)
    original_len = TourBuilder.calculate_length(tour, dist_matrix, idx_map)

    logger.info("Applying perturbations...")
    tour = Perturbation.apply_multiple(tour, dist_matrix, idx_map, coords, tracker,
                                     CONFIG['perturbation']['count'])

    # Rodadas de otimização
    best_tour = tour
    for round_num in range(1, CONFIG['optimization_rounds'] + 1):
        logger.info(f"=== Optimization Round {round_num}/{CONFIG['optimization_rounds']} ===")

        best_tour = ThreeOpt.optimize(best_tour, dist_matrix, idx_map, tracker, f"3-opt round {round_num}")
        best_tour = LocalOptimizer.tabu_search(best_tour, dist_matrix, idx_map, tracker, f"tabu round {round_num}")

    best_tour = TourUtils.remove_subtours(best_tour, dist_matrix, idx_map, coords)
    final_length = TourBuilder.calculate_length(best_tour, dist_matrix, idx_map)

    if final_length == CONFIG['infinity']:
        logger.error("Final tour has infinite length")
        final_length = 0

    print(f"\n{'='*60}")
    print(f"FINAL LENGTH: {int(final_length)}")
    print(f"{'='*60}")

    return best_tour, final_length, nodes, coords, dist_matrix, tracker, seed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python script.py instancia.txt")
        sys.exit(1)

    mp.freeze_support()
    start = time.time()
    
    best_tour, final_length, nodes, coords, dist_matrix, tracker, seed = solve_tsp(sys.argv[1])
    
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    print(f"Tempo de execução: {time_str}")

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
            
            f.write("Matriz de Distâncias:\n")
            for i in range(len(nodes)):
                linha_str = []
                for j in range(len(nodes)):
                    linha_str.append(str(dist_matrix[i][j]))
                f.write("   " + "   ".join(linha_str) + "\n")
                
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

    gap_history = tracker.get_gap_history(final_length)
    plot_filename = base_name + "_grafico.png"
    plot_final_result(best_tour, coords, final_length, gap_history, plot_filename)