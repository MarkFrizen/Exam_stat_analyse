import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Установить отображение чисел с 3 знаками после запятой
np.set_printoptions(precision=3, suppress=True, linewidth=200)

data = [[-10.2, 7],
        [-11.3, 21],
        [-10.2, 29],
        [-7.4, 100],
        [-13.2, 22],
        [-10.6, 16],
        [-10.4, 24],
        [-10.1, 5]]

# Вычисление матрицы расстояний
dist_matrix = squareform(pdist(data, metric='euclidean'))
print("Матрица расстояний (евклидовы расстояния):")
print(dist_matrix)

Z_result = sch.linkage(data, method='single') # метод "ближайшего соседа"
#Z_result = sch.linkage(data, method='complete') # метод "дальнего соседа"
#Z_result = sch.linkage(data, method='centroid') # метод "центроида"
#Z_result = sch.linkage(data, method='average') # метод "средней связи"
#Z_result = sch.linkage(data, method='median') # метод "медианы"
plt.figure(figsize=(10, 7))
sch.dendrogram(Z_result)
plt.title("Дендрограмма")
plt.show()