from vptree import build_vptree, search_knn, region_search, euclidean_distance
import numpy as np


def brute_force_knn(points, query, k):
    dists = [(euclidean_distance(query, p), p) for p in points]
    dists.sort()
    return dists[:k]


def brute_force_region_search(points, query, radius):
    return [p for p in points if euclidean_distance(query, p) <= radius]


def test_knn():
    np.random.seed(0)
    dim = 10
    target = np.full(dim, 0.5)
    points = [np.random.rand(dim) for _ in range(1000)]
    tree = build_vptree(points)
    k = 5
    heap = []
    search_knn(target, tree, k, heap)
    heap_results = sorted([(-d, p) for d, p in heap])
    bf_results = brute_force_knn(points, target, k)

    for (d1, p1), (d2, p2) in zip(heap_results, bf_results):
        assert np.isclose(d1, d2, atol=1e-6)
        assert np.allclose(p1, p2, atol=1e-6)


def test_region_search():
    np.random.seed(1)
    dim = 8
    target = np.full(dim, 0.5)
    points = [np.random.rand(dim) for _ in range(1000)]
    tree = build_vptree(points)
    radius = 0.5
    results = []
    region_search(tree, target, radius, results)
    bf_results = brute_force_region_search(points, target, radius)

    assert len(results) == len(bf_results)
    for p in results:
        assert any(np.allclose(p, bf, atol=1e-6) for bf in bf_results)