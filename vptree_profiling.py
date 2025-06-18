import time
import numpy as np
import matplotlib.pyplot as plt
from vptree import build_vptree, search_knn, region_search, euclidean_distance
import heapq

NUM_TRIALS = 5
POINT_COUNTS = [100, 500, 1000, 2000, 4000]
DIM = 100
K = 5
RADIUS = 4

build_times = []
knn_times = []
region_times = []

for n_points in POINT_COUNTS:
    print(f"Probando con {n_points} puntos...")

    total_build = 0.0
    total_knn = 0.0
    total_region = 0.0

    points = [np.random.rand(DIM) for _ in range(n_points)]
    tree = build_vptree(points)
    
    for _ in range(NUM_TRIALS):
        query = np.full(DIM, 0.5)

         # Tiempo de construcción
        start = time.perf_counter()
        
        total_build += time.perf_counter() - start

        # Tiempo de knn
        heap = []
        start = time.perf_counter()
        search_knn(query, tree, K, heap)
        total_knn += time.perf_counter() - start

        # Tiempo de region search
        results = []
        start = time.perf_counter()
        region_search(tree, query, RADIUS, results)
        total_region += time.perf_counter() - start

        points = [np.random.rand(DIM) for _ in range(n_points)]

    build_times.append(total_build / NUM_TRIALS)
    knn_times.append(total_knn / NUM_TRIALS)
    region_times.append(total_region / NUM_TRIALS)

plt.figure(figsize=(10, 6))
plt.plot(POINT_COUNTS, build_times, label='build_vptree')
plt.plot(POINT_COUNTS, knn_times, label='search_knn')
plt.plot(POINT_COUNTS, region_times, label='region_search')
plt.xlabel('Número de puntos')
plt.ylabel('Tiempo promedio (segundos)')
plt.title(f'VP-Tree: Tiempo promedio vs número de puntos (dim={DIM})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
