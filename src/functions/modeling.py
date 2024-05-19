from typing import Dict, Any

import pandas as pd
import numpy as np
import logging

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import RepeatedKFold

from threadpoolctl import threadpool_limits

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#1. Filtrado Colaborativo

#Paso 1: entrenar el modelo con crossvalidation y el conjunto train
def train_als(
    df_train: pd.DataFrame, 
    params: Dict[str, Any]
) -> AlternatingLeastSquares:
    """
    Entrena un modelo ALS (Alternating Least Squares) con cross-validation.

    Parameters
    ----------
    df_train : pandas.DataFrame
        DataFrame de pandas que contiene los datos de entrenamiento de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros modeling.

    Returns
    -------
    AlternatingLeastSquares: Modelo ALS entrenado.
    """
    logger.info("Iniciando el entrenamiento del modelo ALS...")

    rating_cols = params['train_als']['rating_cols']  # ['recencia', 'frecuencia', 'monto']
    user_col = params['train_als']['user_col']
    item_col = params['train_als']['item_col']

    df_train[user_col] = df_train[user_col].astype('category')
    df_train[item_col] = df_train[item_col].astype('category')

    user_item_matrix = csr_matrix((df_train[rating_cols].mean(axis=1).values,
                                   (df_train[user_col].cat.codes.values,
                                    df_train[item_col].cat.codes.values)),
                                  shape=(df_train[user_col].cat.categories.size, 
                                         df_train[item_col].cat.categories.size))

    # Limitar los hilos de OpenBLAS
    with threadpool_limits(limits=1, user_api='blas'):
        model = AlternatingLeastSquares(factors=params['train_als']['factors'],
                                        regularization=params['train_als']['regularization'],
                                        iterations=params['train_als']['iterations'])
        model.fit(user_item_matrix)

    logger.info("Finalizado el entrenamiento del modelo ALS!")
    
    return model


# Paso 2: búsqueda de hiperparametros óptimos con gridsearch
def grid_search_als(
    model: AlternatingLeastSquares, 
    df_val: pd.DataFrame, 
    params: Dict[str, Any]
) -> AlternatingLeastSquares:
    """
    Realiza una búsqueda de hiperparámetros para el modelo ALS.

    Parameters
    ----------
    model : AlternatingLeastSquares
        Modelo ALS entrenado inicialmente.
    df_val : pandas.DataFrame
        DataFrame de pandas que contiene los datos de validación de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros modeling.

    Returns
    -------
    AlternatingLeastSquares: Modelo ALS con los mejores hiperparámetros.
    """
    logger.info("Iniciando la búsqueda de hiperparámetros...")

    rating_cols = params['train_als']['rating_cols']  # ['recencia', 'frecuencia', 'monto']
    user_col = params['train_als']['user_col']
    item_col = params['train_als']['item_col']
    param_grid = params['param_grid']

    # Crear matriz CSR para ALS
    user_item_matrix = csr_matrix((df_val[rating_cols].mean(axis=1),
                                   (df_val[user_col].astype('category').cat.codes,
                                    df_val[item_col].astype('category').cat.codes)))

    best_score = float('inf')
    best_params = {}

    for factors in param_grid['factors']:
        for regularization in param_grid['regularization']:
            for iterations in param_grid['iterations']:
                model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
                model.fit(user_item_matrix)

                score = evaluate_model(model, user_item_matrix)  # Usar conjunto de validación para evaluar
                if score < best_score:
                    best_score = score
                    best_params = {
                        'factors': factors,
                        'regularization': regularization,
                        'iterations': iterations
                    }

    logger.info(f"Mejores hiperparámetros encontrados: {best_params} con una puntuación de {best_score}")

    # Entrenar el modelo final con los mejores hiperparámetros
    model = AlternatingLeastSquares(factors=best_params['factors'],
                                    regularization=best_params['regularization'],
                                    iterations=best_params['iterations'])
    model.fit(user_item_matrix)
    
    logger.info("Finalizado la búsqueda de hiperparámetros!")
    
    return model


# Paso 3: Evaluar el modelo
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



