import numpy as np
import random
import heapq

class MetricSpace:
	def distance(self, a, b):
		raise NotImplementedError("Subclases deben implementar 'distance'")

	def validate(self, a, b, c):
		d_ab = self.distance(a, b)
		d_ba = self.distance(b, a)
		assert d_ab >= 0, "La distancia no puede ser negativa"
		assert d_ab == d_ba, "La distancia debe de ser simétrica"
		
		d_ac = self.distance(a, c)
		d_bc = self.distance(b, c)
		assert d_ac <= d_ab + d_bc, "Debe de cumplir la desigualdad triangular"

class EuclideanSpace(MetricSpace):
	def distance(self, a, b):
		return np.linalg.norm(a - b)


class VPNode:
	def __init__(self, point, threshold, left, right):
		self.point = point          # Punto pivote
		self.threshold = threshold  # Radio de corte (distancia mediana)
		self.left = left            # Subárbol interior (más cercano)
		self.right = right          # Subárbol exterior (más lejano)

class VPTree:
	def __init__(self, points, metric_space, sample_size=20):
		self.metric_space = metric_space
		self.sample_size = sample_size
		self.root = self._build(points)

	# Construcción del árbol
	def _build(self, points):
		if not points:
			return None
		if len(points) == 1:
			return VPNode(points[0], 0, None, None)

		pivot = select_pivot_max_diversity_approx(points, self.metric_space, self.sample_size)
		distances = [(p, self.metric_space.distance(pivot, p)) for p in points if not self.metric_space.distance(pivot, p) == 0]
		dist_vals = [d for _, d in distances]
		mu = np.median(dist_vals)

		inner = [p for p, d in distances if d <= mu]
		outer = [p for p, d in distances if d > mu]

		left = self._build(inner)
		right = self._build(outer)
		return VPNode(pivot, mu, left, right)

	# Búsqueda de los k vecinos más cercanos
	def search_knn_aux(self, target, k):
		heap = []
		self.search_knn(self.root, target, k, heap)
		return sorted([(-d, p) for d, p in heap])

	def search_knn(self, node, target, k, heap):
		if node is None:
			return

		d = self.metric_space.distance(target, node.point)

		if len(heap) < k:
			heapq.heappush(heap, (-d, node.point))
		elif d < -heap[0][0]:
			heapq.heappushpop(heap, (-d, node.point))

		if d < node.threshold:
			self.search_knn(node.left, target, k, heap)
			if d + (-heap[0][0]) >= node.threshold:
				self.search_knn(node.right, target, k, heap)
		else:
			self.search_knn(node.right, target, k, heap)
			if d - (-heap[0][0]) <= node.threshold:
				self.search_knn(node.left, target, k, heap)

	# Búsqueda de puntos dentro de una región dada
	def region_search_aux(self, target, radius):
		results = []
		self.region_search(self.root, target, radius, results)
		return results

	def region_search(self, node, target, radius, results):
		if node is None:
			return

		d = self.metric_space.distance(target, node.point)
		if d <= radius:
			results.append(node.point)

		if d < node.threshold:
			self.region_search(node.left, target, radius, results)
			if d + radius >= node.threshold:
				self.region_search(node.right, target, radius, results)
		else:
			self.region_search(node.right, target, radius, results)
			if d - radius <= node.threshold:
				self.region_search(node.left, target, radius, results)

def select_pivot_max_diversity_approx(points, metric_space, sample_size=20):
	n = len(points)
	if n <= sample_size:
		sample = points  # si hay pocos puntos, usa todos
	else:
		sample = random.sample(points, sample_size)

	max_avg_dist = -1
	best_point = None
	
	for p in points:
		d = metric_space.distance(p, p)
		distances = [metric_space.distance(p, s) for s in sample if not metric_space.distance(p, s) == 0]
		avg_dist = np.mean(distances)
		if avg_dist > max_avg_dist:
			max_avg_dist = avg_dist
			best_point = p

	return best_point
