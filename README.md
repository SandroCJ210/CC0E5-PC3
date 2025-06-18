# CC0E5-PC3

## VP-Tree y KD-Tree en Python

Este proyecto implementa dos estructuras de datos para búsqueda espacial eficiente: **VP-Tree (Vantage-Point Tree)** y **KD-Tree (K-Dimensional Tree)**. Está orientado a resolver problemas de búsqueda de vecinos más cercanos (k-NN), búsqueda por región, y comparación entre estructuras.

### Archivos

#### `vptree.py`
Contiene una implementación completa de un **VP-Tree** optimizado con una heurística de pivote basada en **máxima diversidad aproximada**.

##### Características:
- Construcción recursiva del árbol con partición basada en la distancia mediana.
- Función de selección de pivote que maximiza la diversidad dentro de una muestra aleatoria.
- Algoritmos de:
  - Búsqueda de k vecinos más cercanos (`search_knn`)
  - Búsqueda en región (`region_search`)

##### Uso:
```python
from vptree import VPTree
```
---

#### `test_vptree.py`

Contiene pruebas automatizadas para validar la implementación del **VP-Tree**, comparando los resultados contra métodos de fuerza bruta.

#### Informe.ipynb
Contiene una explicación detallada de la implementación hecha, además de contener los scripts de profiling para poder analizar el tiempo de ejecución del árbol y sus métodos.

##### Pruebas implementadas:

* `test_knn()`: Valida el resultado de `search_knn` contra una búsqueda lineal.
* `test_region_search()`: Valida la búsqueda por radio comparando con fuerza bruta.

##### Ejecución:

```bash
python test_vptree.py
```

---

#### `kdtree.py`

Implementa desde cero un **KD-Tree** con API de alto nivel (`KdTree`) y estructuras auxiliares como `Point` y `Cube`.

##### Características:

* Inserción, eliminación y verificación de pertenencia de puntos.
* Búsqueda de vecino más cercano.
* Búsqueda por distancia (radio) y por región (hipercubo).
* Soporte para espacios de dimensión arbitraria.
* Validaciones estrictas y mensajes de error descriptivos.

##### Clases principales:

* `Point`: Representa un punto inmutable en un espacio k-dimensional.
* `Cube`: Representa un hipercubo que puede contener puntos o intersectar con regiones.
* `KdTree`: Estructura principal que encapsula la lógica de inserción, búsqueda y eliminación.
* `_Node`: Nodo interno del árbol, gestionado por `KdTree`.

##### Ejemplo de uso:

```python
from kdtree import Point, Cube, KdTree

points = [Point([2, 3]), Point([5, 4]), Point([9, 6])]
tree = KdTree(points)
print(tree.contains(Point([5, 4])))  # True
print(tree.nearestNeighbour(Point([8, 5])))  # Vecino más cercano
```

---

### Comparación de estructuras

Este proyecto permite comparar dos estructuras espaciales:

| Característica     | VP-Tree                            | KD-Tree                                 |
| ------------------ | ---------------------------------- | --------------------------------------- |
| Tipo de división   | Por distancia al pivote            | Por coordenada en una dimensión         |
| Búsqueda NN        | Eficiente en espacios métricos     | Eficiente en espacios de baja dimensión |
| Región de búsqueda | Esferas                            | Hipercubos                              |
| Aplicación típica  | Indexación métrica (e.g. imágenes) | Datos estructurados (e.g. GIS)          |

---

### Requisitos

* Python 3.8+
* NumPy

Instalación de dependencias:

```bash
pip install numpy
```

---
