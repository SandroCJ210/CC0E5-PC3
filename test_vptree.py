from vptree import VPTree, EuclideanSpace
import numpy as np


def brute_force_knn(points, query, metric_space, k):
    dists = [(metric_space.distance(query, p), p) for p in points]
    dists.sort()
    return dists[:k]


def brute_force_region_search(points, query, radius, metric_space):
    return [p for p in points if metric_space.distance(query, p) <= radius]


def test_knn():
    np.random.seed(0)
    dim = 10
    target = np.full(dim, 0.5)
    points = [np.random.rand(dim) for _ in range(1000)]
    metric = EuclideanSpace()
    tree = VPTree(points, metric)
    k = 5
    k_nearest_neighbours = tree.search_knn_aux(target, k)
    bf_results = brute_force_knn(points, target, metric, k)

    for (d1, p1), (d2, p2) in zip(k_nearest_neighbours, bf_results):
        assert np.isclose(d1, d2, atol=1e-6)
        assert np.allclose(p1, p2, atol=1e-6)


def test_region_search():
    np.random.seed(1)
    dim = 8
    target = np.full(dim, 0.5)
    points = [np.random.rand(dim) for _ in range(1000)]
    metric = EuclideanSpace()
    tree = VPTree(points, metric)
    radius = 0.5

    region_search = tree.region_search_aux(target, radius)
    bf_results = brute_force_region_search(points, target, radius, metric)

    assert len(region_search) == len(bf_results)
    for p in region_search:
        assert any(np.allclose(p, bf, atol=1e-6) for bf in bf_results)

def test_no_duplicate_points():
    points = [np.array([i, i]) for i in range(10)]
    tree = VPTree(points, EuclideanSpace())

    checked = set()
    def check(node):
        if node is None:
            return
        pt_tuple = tuple(node.point)
        assert pt_tuple not in checked
        checked.add(pt_tuple)
        check(node.left)
        check(node.right)
    check(tree.root)

def test_euclidean_in_metric_space():
    metric_space = EuclideanSpace()
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([1, 1])
    metric_space.validate(a, b, c)