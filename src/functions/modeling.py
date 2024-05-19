from typing import Dict, Any

import pandas as pd
import numpy as np
import logging

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from threadpoolctl import threadpool_limits

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#1. Filtrado Colaborativo

#Paso 1: entrenar el modelo con el conjunto train
def train_als(
    df_train: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, AlternatingLeastSquares]:
    """
    Entrena un modelo ALS (Alternating Least Squares) para cada segmento definido en 'segmento_fc'.

    Parameters
    ----------
    df_train : pandas.DataFrame
        DataFrame de pandas que contiene los datos de entrenamiento de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros modeling.

    Returns
    -------
    Dict[str, AlternatingLeastSquares]: Diccionario de modelos ALS entrenados por segmento.
    """
    logger.info("Iniciando el entrenamiento del modelo ALS por segmento...")

    # Parámetros
    rating_cols = params['train_als']['rating_cols']  # ['recencia', 'frecuencia', 'monto']
    user_col = params['train_als']['user_col']
    item_col = params['train_als']['item_col']
    segment_col = params['train_als']['segmento_fc']  # Columna para segmentar

    models = {}

    # Entrenar un modelo ALS para cada segmento
    for segment in df_train[segment_col].unique():
        logger.info(f"Entrenando modelo ALS para el segmento: {segment}")
        df_segment = df_train[df_train[segment_col] == segment]

        # Convertir las columnas de usuario y producto a categóricas
        df_segment[user_col] = df_segment[user_col].astype('category')
        df_segment[item_col] = df_segment[item_col].astype('category')

        # Crear la matriz de usuario-elemento
        user_item_matrix = csr_matrix((df_segment[rating_cols].mean(axis=1).values,
                                       (df_segment[user_col].cat.codes.values,
                                        df_segment[item_col].cat.codes.values)),
                                      shape=(df_segment[user_col].cat.categories.size,
                                             df_segment[item_col].cat.categories.size))

        # Limitar los hilos de OpenBLAS
        # Entrene el modelo ALS con un solo hilo para evitar problemas de rendimiento
        with threadpool_limits(limits=1, user_api='blas'):
            model = AlternatingLeastSquares(factors=params['train_als']['factors'],
                                            regularization=params['train_als']['regularization'],
                                            iterations=params['train_als']['iterations'])
            model.fit(user_item_matrix)

        # Guardar el modelo en el diccionario
        models[segment] = model
        logger.info(f"Modelo ALS para el segmento {segment} entrenado con éxito.")

    logger.info("Finalizado el entrenamiento de todos los modelos ALS por segmento!")

    return models


# Paso 2: búsqueda de hiperparametros óptimos con gridsearch
def grid_search_als(
    models: Dict[str, AlternatingLeastSquares], 
    df_val: pd.DataFrame, 
    params: Dict[str, Any]
) -> Dict[str, AlternatingLeastSquares]:
    """
    Realiza una búsqueda de hiperparámetros para cada modelo ALS en el diccionario.

    Parameters
    ----------
    models : Dict[str, AlternatingLeastSquares]
        Diccionario de modelos ALS entrenados inicialmente, segmentados por claves.
    df_val : pandas.DataFrame
        DataFrame de pandas que contiene los datos de validación de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros modeling.

    Returns
    -------
    Dict[str, AlternatingLeastSquares]: Diccionario de modelos ALS con los mejores hiperparámetros por segmento.
    """
    logger.info("Iniciando la búsqueda de hiperparámetros para cada segmento...")

    # Parámetros
    rating_cols = params['train_als']['rating_cols']  # ['recencia', 'frecuencia', 'monto']
    user_col = params['train_als']['user_col']
    item_col = params['train_als']['item_col']
    segment_col = params['train_als']['segmento_fc']  # Columna para segmentar
    param_grid = params['param_grid']

    # Diccionario para guardar los mejores modelos
    best_models = {}

    # Búsqueda de hiperparámetros para cada segmento
    for segment, model in models.items():
        logger.info(f"Buscando mejores hiperparámetros para el segmento: {segment}")
        df_segment = df_val[df_val[segment_col] == segment]

        user_item_matrix = csr_matrix((df_segment[rating_cols].mean(axis=1),
                                       (df_segment[user_col].astype('category').cat.codes,
                                        df_segment[item_col].astype('category').cat.codes)))

        best_score = float('inf')
        best_params = {}

        # Búsqueda de hiperparámetros
        for factors in param_grid['factors']:
            for regularization in param_grid['regularization']:
                for iterations in param_grid['iterations']:
                    temp_model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
                    temp_model.fit(user_item_matrix)

                    score = evaluate_model(temp_model, user_item_matrix)  # Usar conjunto de validación para evaluar
                    if score < best_score:
                        best_score = score
                        best_params = {
                            'factors': factors,
                            'regularization': regularization,
                            'iterations': iterations
                        }

        logger.info(f"Mejores hiperparámetros para el segmento {segment}: {best_params} con una puntuación de {best_score}")

        # Entrenar el modelo final con los mejores hiperparámetros
        best_model = AlternatingLeastSquares(factors=best_params['factors'],
                                             regularization=best_params['regularization'],
                                             iterations=best_params['iterations'])
        best_model.fit(user_item_matrix)
        
        best_models[segment] = best_model
        logger.info(f"Modelo ALS optimizado para el segmento {segment} entrenado con éxito.")

    logger.info("Finalizado la búsqueda de hiperparámetros para todos los segmentos!")
    
    return best_models


# Paso 3: evaluar rendimiento del algoritmo
def evaluate_model(
    model: AlternatingLeastSquares, 
    val_data: csr_matrix
) -> float:
    """
    Evalúa el modelo ALS en el conjunto de validación.

    Parameters
    ----------
    model : Alternating Least Squares
        Modelo ALS entrenado.
    val_data : csr_matrix
        Matriz CSR de validación.

    Returns
    -------
    float: Puntuación de evaluación del modelo.
    """
    logger.info("Iniciando la evaluación del modelo...")

    # Calcular la predicción y el valor real
    user_factors = model.user_factors

    # Transponer los factores de los productos
    item_factors = model.item_factors

    # Calcular la predicción
    prediction = user_factors.dot(item_factors.T)

    # Obtener el valor real
    actual = val_data.toarray()

    # Calcular el error cuadrático medio
    mse = np.mean((prediction - actual) ** 2)

    logger.info(f"Puntuación de evaluación del modelo: {mse}")

    return mse


