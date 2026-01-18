import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer

np.set_printoptions(precision=3, suppress=True, linewidth=200)

def runFactorAnalysis():
    # Создаем пример данных (6 факторов по 30 наблюдений)
    data = {
        'X1': [3, 3, 3, 4, 2, 2, 4, 4, 2, 4, 4, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        'X2': [100, 103, 146, 82, 97, 109, 78, 138, 89, 134, 141, 124, 115, 95, 108, 147, 131, 150, 87, 122, 146, 100, 86, 74, 87, 118, 108, 93, 117, 86],
        'X3': [8.992, 5.892, 0.549, 4.197, 10.220, 10.997, 1.090, 2.656, 6.894, 4.128, 10.004, 6.612, 6.472, 7.461, 3.827, 4.675, 7.081, 4.031, 1.141, 7.705, 6.092, 10.949, 3.545, 7.165, 10.003, 6.927, 0.954, 8.315, 3.637, 8.113],
        'X4': [9.974, 4.203, 6.406, 5.507, 7.361, 10.459, 4.866, 13.911, 9.507, 7.717, 4.978, 8.348, 7.941, 4.348, 12.137, 12.352, 8.231, 6.875, 12.367, 11.616, 8.882, 7.461, 4.162, 10.932, 9.114, 11.286, 4.197, 9.993, 12.516, 4.793],
        'X5': [193, 96, 331, 192, 173, 158, 150, 206, 251, 155, 231, 323, 262, 345, 113, 256, 201, 259, 177, 280, 278, 315, 299, 240, 335, 98, 300, 336, 236, 309],
        'X6': [176, 93, 294, 176, 159, 146, 140, 187, 226, 144, 209, 287, 235, 306, 108, 230, 183, 232, 162, 251, 249, 280, 267, 217, 297, 95, 268, 298, 213, 275]
    }

    df = pd.DataFrame(data)
    print(df)

    # Вычисляем корреляционную матрицу
    corr_matrix = df.corr()
    print("Корреляционная матрица:")
    print(corr_matrix)


    # Вычисляем собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    print("\nСобственные значения:")
    print(eigenvalues)

    print("\nСобственные векторы (столбцы матрицы):")
    print(eigenvectors)

    optimal_factors=1

    fa = FactorAnalyzer(n_factors=optimal_factors, rotation='varimax')
    fa.fit(df)

    #Матрица факторных нагрузок

    loadings = fa.loadings_
    print("\nМатрица факторных нагрузок:")
    loadings_df = pd.DataFrame(
        loadings,
        index=df.columns,
        columns=[f'Factor_{i + 1}' for i in range(optimal_factors)]
    )
    print(loadings_df.round(3))

    # Общности (communalities)
    communalities = fa.get_communalities()
    print("Общности переменных:")
    for i, var in enumerate(df.columns):
        print(f"{var}: {communalities[i]:.3f}")

    # Дисперсия, объясняемая факторами
    variance = fa.get_factor_variance()
    variance_df = pd.DataFrame({
        'SS Loadings': variance[0],
        'Доля объясненной дисперсии': variance[1],
        'Накопительна дисперсия': variance[2]
    }, index=[f'Factor_{i + 1}' for i in range(optimal_factors)])

    print(f"\nОбъясненная дисперсия:")
    print(variance_df.round(3))

    # Получаем факторные оценки
    factor_scores = fa.transform(df)
    factor_scores_df = pd.DataFrame(
        factor_scores,
        columns=[f'Factor_{i+1}_Score' for i in range(optimal_factors)]
    )

    print("Первые 5 строк факторных оценок:")
    print(factor_scores_df.head().round(3))

runFactorAnalysis()