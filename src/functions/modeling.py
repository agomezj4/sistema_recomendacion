from typing import Dict, Any

import pandas as pd
import logging
from sklearn.model_selection import RepeatedStratifiedKFold
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#1. Filtrado Colaborativo
def grid_search_als(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Realiza una búsqueda de hiperparámetros para el modelo ALS.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene los datos de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros necesarios para la búsqueda de hiperparámetros.

    Returns
    -------
    Dict[str, Any]: Diccionario con los mejores hiperparámetros encontrados.
    """
    logger.info("Iniciando la búsqueda de hiperparámetros...")

    rating_col = params['rating_col']
    user_col = params['user_col']
    item_col = params['item_col']
    param_grid = params['param_grid']
    cv_config = params['cv_config']

    # Crear matriz CSR para ALS
    user_item_matrix = csr_matrix((df[rating_col],
                                   (df[user_col].astype('category').cat.codes,
                                    df[item_col].astype('category').cat.codes)))

    cv = RepeatedStratifiedKFold(**cv_config)
    best_score = float('inf')
    best_params = {}

    for factors in param_grid['factors']:
        for regularization in param_grid['regularization']:
            for iterations in param_grid['iterations']:
                scores = []
                for train_index, val_index in cv.split(df, df[rating_col]):
                    train_data = user_item_matrix[train_index]
                    val_data = user_item_matrix[val_index]

                    model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
                    model.fit(train_data)

                    # Evaluar el modelo en el conjunto de validación
                    score = evaluate_model(model, val_data)  # Implementa esta función según tus métricas
                    scores.append(score)

                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = {
                        'factors': factors,
                        'regularization': regularization,
                        'iterations': iterations
                    }

    logger.info(f"Mejores hiperparámetros encontrados: {best_params} con una puntuación de {best_score}")
    return best_params

def evaluate_model(model: AlternatingLeastSquares, val_data: csr_matrix) -> float:
    """
    Evalúa el modelo ALS en el conjunto de validación.

    Parameters
    ----------
    model : AlternatingLeastSquares
        Modelo ALS entrenado.
    val_data : csr_matrix
        Matriz CSR de validación.

    Returns
    -------
    float: Puntuación de evaluación del modelo.
    """
    # Implementa la lógica de evaluación aquí (por ejemplo, RMSE, precisión, recall, etc.)
    # Ejemplo simple:
    user_factors = model.user_factors
    item_factors = model.item_factors
    prediction = user_factors.dot(item_factors.T)
    actual = val_data.toarray()

    mse = np.mean((prediction - actual) ** 2)
    return mse

def train_als(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> AlternatingLeastSquares:
    """
    Entrena un modelo ALS (Alternating Least Squares) por segmento.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene los datos de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros necesarios para entrenar el modelo ALS.

    Returns
    -------
    AlternatingLeastSquares: Modelo ALS entrenado.
    """
    logger.info("Iniciando el entrenamiento del modelo ALS...")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    df_train, df_test = train_test_split(df, test_size=params['test_size'], random_state=params['random_state'])

    # Buscar los mejores hiperparámetros
    best_params = grid_search_als(df_train, params)

    # Entrenar el modelo final con los mejores hiperparámetros
    rating_col = params['rating_col']
    user_col = params['user_col']
    item_col = params['item_col']

    user_item_matrix = csr_matrix((df_train[rating_col],
                                   (df_train[user_col].astype('category').cat.codes,
                                    df_train[item_col].astype('category').cat.codes)))

    model = AlternatingLeastSquares(factors=best_params['factors'],
                                    regularization=best_params['regularization'],
                                    iterations=best_params['iterations'])
    model.fit(user_item_matrix)

    logger.info("Finalizado el entrenamiento del modelo ALS!")

    # Evaluar el modelo en el conjunto de prueba
    test_matrix = csr_matrix((df_test[rating_col],
                              (df_test[user_col].astype('category').cat.codes,
                               df_test[item_col].astype('category').cat.codes)))
    test_score = evaluate_model(model, test_matrix)
    logger.info(f"Puntuación del modelo en el conjunto de prueba: {test_score}")

    return model


