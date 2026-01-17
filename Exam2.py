import numpy as np
from factor_analyzer import FactorAnalyzer

# Исходная корреляционная матрица
corr_matrix = np.array([
    [1.00, 0.80, 0.70],
    [0.80, 1.00, 0.60],
    [0.70, 0.60, 1.00]
])

print("Исходная корреляционная матрица:")
print(corr_matrix)
print()

# Выполнение факторного анализа с одним фактором
fa = FactorAnalyzer(n_factors=1, method='minres', rotation=None)
fa.fit(corr_matrix)

# Получение результатов
loadings = fa.loadings_  # Факторные нагрузки
communalities = fa.get_communalities()  # Общности (доля общей дисперсии)
uniquenesses = fa.get_uniquenesses()  # Специфические дисперсии

print("\n1. Факторные нагрузки:")
for i, loading in enumerate(loadings, 1):
    print(f"   Переменная {i}: {loading[0]:.4f}")

print("\n2. Доли дисперсии, объясняемые общим фактором:")
for i, comm in enumerate(communalities, 1):
    print(f"   Переменная {i}: {comm:.4f}")

print("\n3. Специфические дисперсии:")
for i, uniq in enumerate(uniquenesses, 1):
    print(f"   Переменная {i}: {uniq:.4f}")

