# Optimización de Hiperparámetros con Algoritmos Genéticos

Este repositorio contiene ejemplos prácticos de cómo usar **algoritmos genéticos** para optimizar hiperparámetros de modelos de Machine Learning, utilizando las librerías [jMetal](https://github.com/jMetal/jMetalPy) y [scikit-learn](https://scikit-learn.org/).

---

## ¿Por qué usar algoritmos genéticos en Machine Learning?

El espacio de búsqueda de hiperparámetros es grande, no lineal y puede tener múltiples óptimos locales. Los algoritmos genéticos permiten explorar eficientemente sin evaluar todas las combinaciones posibles.

---

## Estructura del repositorio

```
.
├── 1_knn_ga_monoobjetivo.ipynb       # KNN: optimización de k (1 variable)
├── 2_knn_ga_multivariable.ipynb      # KNN: optimización de k, weights y metric (3 variables)
├── 3_knn_multiobjetivo.ipynb         # KNN: optimización biobjetivo (accuracy vs tiempo)
├── 4_randomforest.ipynb              # Random Forest: optimización de 4 hiperparámetros
└── 7_randomforest_3.ipynb            # Random Forest: optimización triobjetivo con Frente de Pareto 3D
```

---

## Notebooks

### 1. KNN Monoobjetivo — `1_knn_ga_monoobjetivo.ipynb`

Introducción al uso de algoritmos genéticos con jMetal. Se optimiza **únicamente el valor de `k`** del clasificador KNN sobre el dataset Iris.

- **Dataset:** Iris
- **Variable a optimizar:** `n_neighbors` (k) — rango [1, 20]
- **Evaluación:** Validación cruzada (3 folds)
- **Objetivo:** Maximizar accuracy
- **Resultado típico:** k=3, Accuracy ≈ 0.987

---

### 2. KNN Multivariable — `2_knn_ga_multivariable.ipynb`

Extensión del ejercicio anterior. Se optimizan **3 hiperparámetros simultáneamente** del clasificador KNN.

- **Dataset:** Iris
- **Variables a optimizar:**
  - `n_neighbors` (k) — rango [1, 20]
  - `weights` — `"uniform"` o `"distance"` (codificado como [0, 1])
  - `metric` — `"euclidean"` o `"manhattan"` (codificado como [0, 1])
- **Evaluación:** Validación cruzada (3 folds)
- **Objetivo:** Maximizar accuracy

---

### 3. KNN Multiobjetivo — `3_knn_multiobjetivo.ipynb`

Introduce la **optimización multiobjetivo** usando el algoritmo NSGA-II. Se busca el balance óptimo entre accuracy y tiempo de ejecución, generando un **Frente de Pareto**.

- **Dataset:** Iris
- **Variable a optimizar:** `n_neighbors` (k) — rango [1, 20]
- **Objetivos:**
  - Maximizar accuracy
  - Minimizar tiempo de inferencia
- **Restricción:** accuracy ≥ 0.99
- **Algoritmo:** NSGA-II
- **Salida:** Frente de Pareto (conjunto de soluciones no dominadas)

---

### 4. Random Forest — `4_randomforest.ipynb`

Aplicación más compleja sobre un dataset real. Se optimizan **4 hiperparámetros** del clasificador Random Forest.

- **Dataset:** Breast Cancer
- **Variables a optimizar:**
  - `n_estimators` — rango [10, 200]
  - `max_depth` — rango [1, 20]
  - `min_samples_split` — rango [2, 20]
  - `criterion` — `"gini"` o `"entropy"` (codificado como [0, 1])
- **Evaluación:** Validación cruzada (5 folds)
- **Objetivo:** Maximizar accuracy
- **Resultado típico:** Accuracy ≈ 0.956

---

### 7. Random Forest Triobjetivo — `7_randomforest_3.ipynb`

Extiende el ejercicio 4 a **optimización con 3 objetivos simultáneos** usando NSGA-II, generando un **Frente de Pareto en 3D**.

> **Nota:** Un problema con más de 3 objetivos se denomina *many-objective optimization* y requiere algoritmos específicos (NSGA-III, MOEA/D, etc.). Para 3 objetivos, NSGA-II sigue siendo aplicable aunque algoritmos como NSGA-III suelen preferirse.

- **Dataset:** Breast Cancer
- **Variables a optimizar:**
  - `n_estimators` — rango [10, 200]
  - `max_depth` — rango [1, 20]
  - `min_samples_split` — rango [2, 20]
  - `criterion` — `"gini"` o `"entropy"` (codificado como [0, 1])
- **Evaluación:** Validación cruzada (5 folds)
- **Objetivos:**
  - Maximizar accuracy (invertido: `-acc`)
  - Minimizar `n_estimators` (normalizado a [0, 1])
  - Minimizar `max_depth` (normalizado a [0, 1])
- **Algoritmo:** NSGA-II (`population_size=40`, `max_evaluations=400`)
- **Salida:** Frente de Pareto con 40 soluciones no dominadas, visualizado en un gráfico 3D

**Extremos del Frente de Pareto (resultado de ejemplo):**

| Criterio | Accuracy | n_estimators | max_depth |
|---|---|---|---|
| Mayor accuracy | 0.9561 | 10 | 5 |
| Menor n_estimators | 0.9210 | 10 | 1 |
| Menor max_depth | 0.9210 | 10 | 1 |

---

## Tecnologías utilizadas

| Librería | Uso | Version |
|---|---|---|
| `jMetalPy` | Framework de algoritmos evolutivos (GA, NSGA-II) | 1.9.0 |
| `scikit-learn` | Modelos ML y evaluación (KNN, Random Forest, cross_val_score) | 1.8.0 |
| `matplotlib` | Visualización del Frente de Pareto | 3.10.9 |

---

## Conceptos clave

**FloatProblem / FloatSolution:** Representación del espacio de búsqueda como variables continuas. Las variables categóricas (como `weights` o `criterion`) se codifican mediante *label encoding* en rangos [0, 1].

**jMetal minimiza por defecto**, por lo que el accuracy se invierte (`-score`) para convertir el problema de maximización en minimización.

**Label encoding:** Las variables categóricas se mapean a valores numéricos continuos y se redondean durante la evaluación:
- `0 → "uniform"` / `1 → "distance"`
- `0 → "euclidean"` / `1 → "manhattan"`
- `0 → "gini"` / `1 → "entropy"`

---

## Configuración típica del algoritmo genético

```python
GeneticAlgorithm(
    problem=problem,
    population_size=10,
    offspring_population_size=10,
    mutation=PolynomialMutation(probability=0.2, distribution_index=20),
    crossover=SBXCrossover(probability=0.9, distribution_index=20),
    termination_criterion=StoppingByEvaluations(max_evaluations=50)
)
```

El algoritmo sigue el ciclo:
1. Genera soluciones aleatorias
2. Evalúa cada solución (entrena el modelo con cross-validation)
3. Selecciona las mejores
4. Aplica cruce (SBX) y mutación (Polynomial)
5. Repite hasta alcanzar el criterio de parada
