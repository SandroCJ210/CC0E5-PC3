import numpy as np
import random

class VPNode:
	def __init__(self, point, threshold, left, right):
		self.point = point          # Punto pivote
		self.threshold = threshold  # Radio de corte (distancia mediana)
		self.left = left            # Subárbol interior (más cercano)
		self.right = right          # Subárbol exterior (más lejano)


def euclidean_distance(a, b):
	return np.linalg.norm(a - b)

def select_pivot_max_diversity_approx(points, sample_size=20):
	n = len(points)
	if n <= sample_size:
		sample = points  # si hay pocos puntos, usa todos
	else:
		sample = random.sample(points, sample_size)

	max_avg_dist = -1
	best_point = None

	for p in points:
		distances = [euclidean_distance(p, s) for s in sample if not np.array_equal(p, s)]
		avg_dist = np.mean(distances)
		if avg_dist > max_avg_dist:
			max_avg_dist = avg_dist
			best_point = p

	return best_point

def build_vptree(points, sample_size=20):
	if len(points) == 0:
		return None

	if len(points) == 1:
		return VPNode(point=points[0], threshold=0, left=None, right=None)

	# Usar versión aproximada
	pivot = select_pivot_max_diversity_approx(points, sample_size=sample_size)

	distances = [(p, euclidean_distance(pivot, p)) for p in points if not np.array_equal(p, pivot)]
	dist_vals = [d for _, d in distances]
	mu = np.median(dist_vals)

	inner = [p for p, d in distances if d <= mu]
	outer = [p for p, d in distances if d > mu]

	left = build_vptree(inner, sample_size)
	right = build_vptree(outer, sample_size)

	return VPNode(point=pivot, threshold=mu, left=left, right=right)
