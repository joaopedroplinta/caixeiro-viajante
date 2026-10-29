import math
import os
import tempfile

import tsp

ROOT = os.path.dirname(os.path.abspath(__file__))


def tour_length(tour, coords):
    # mesma regra do núcleo e da TSPLIB: EUC_2D arredondado
    d = lambda a, b: int(math.sqrt((coords[a][0] - coords[b][0]) ** 2 + (coords[a][1] - coords[b][1]) ** 2) + 0.5)
    return sum(d(tour[i], tour[i + 1]) for i in range(len(tour) - 1))


def check(path, seconds, optimum=None, slack=1.0):
    tour, length, nodes, coords, _, _, _ = tsp.solve_tsp(path, seconds)
    assert tour[0] == tour[-1] and sorted(tour[:-1]) == sorted(nodes), "tour inválido"
    assert length == tour_length(tour, coords), "comprimento diverge do recalculado"
    if optimum:
        assert length <= optimum * slack, f"{length} > {optimum} * {slack}"


def test_eil51():
    check(os.path.join(ROOT, "tsplib", "eil51.tsp"), 2, 426, 1.03)


def test_instancias_minusculas():
    # n < 8 não roda ILS; n pequeno não pode quebrar o LK nem o alpha
    for n in (1, 2, 3, 4, 5, 7, 8, 9):
        with tempfile.NamedTemporaryFile("w", suffix=".tsp", delete=False) as f:
            f.write(f"DIMENSION : {n}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
            for i in range(n):
                f.write(f"{i + 1} {(i * 37) % 11} {(i * 53) % 17}\n")
            f.write("EOF\n")
        try:
            check(f.name, 0.2)
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    test_eil51()
    test_instancias_minusculas()
    print("ok")
