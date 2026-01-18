import numpy as np

# Заданные данные
A = np.array([
    [0,   0.2, 0.3],
    [0.2, 0,   0.1],
    [0.3, 0.2, 0]
])

Y = np.array([20, 40, 130])

# 1. Найдем матрицу полных затрат (обратная матрица к (E - A))
E = np.eye(3)  # Единичная матрица
B = E - A
B_inv = np.linalg.inv(B)  # Обратная матрица

# Вектор валового выпуска X = (E - A)^(-1) * Y
X = B_inv @ Y
print("Вектор валового выпуска X:")
print(X)

# 2. Межотраслевые поставки: X_ij = A_ij * X_j
X_matrix = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        X_matrix[i, j] = A[i, j] * X[j]

print("\nМежотраслевые поставки (X_ij = A_ij * X_j):")
print(X_matrix)

# 3. Чистая продукция каждой отрасли: N_i = X_i - sum_j(A_ij * X_j)
N = X - np.sum(X_matrix, axis=0)
print("\nЧистая продукция каждой отрасли:")
print(N)