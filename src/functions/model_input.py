from typing import Dict, Any, Tuple

import pandas as pd
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1. Transformación de métricas RFM
def transform_rfm(
        df: pd.DataFrame, #data_featured
        params: Dict[str, Any] #featuring
) -> pd.DataFrame:
    """
    Transforma las métricas RFM en el dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene las métricas RFM de los clientes.
    params: Dict[str, Any]
        Diccionario de parámetros model_input.

    Returns
    -------
    pd.DataFrame: DataFrame con las nuevas columnas RFM transformadas.
    """
    logger.info("Iniciando la transformación de métricas RFM...")

    # Parámetros
    recencia_col = params['recencia'][0]
    frecuencia_col = params['frecuencia'][0]
    monto_col = params['monto'][0]

    # Copia del dataframe original para no modificarlo
    df_copy = df.copy()

    # Transformación de las métricas RFM
    df_copy['R_calificacion'] = df_copy[recencia_col].max() - df_copy[recencia_col]
    df_copy['F_calificacion'] = df_copy[frecuencia_col]
    df_copy['M_calificacion'] = df_copy[monto_col]

    logger.info("Finalizada la transformación de métricas RFM!")

    return df_copy


# 2. Escalado de los valores de calificación
def min_max_scaler(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Estandariza las columnas numéricas (excluyendo binarias) de un DataFrame utilizando el método Min-Max Scaler.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas que se estandarizará.
    params: Dict[str, Any]
        Diccionario de parámetros model_input.

    Returns
    -------
    pd.DataFrame
        DataFrame estandarizado.
    """
    logger.info("Iniciando la estandarización con Min-Max Scaler...")

    # Identificar las columnas numéricas
    numeric_cols = params['numeric_cols']

    # Aplicar Min-Max Scaler solo a las columnas numéricas especificadas
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        if range_val != 0:  # Evita la división por cero en caso de que todas las entradas en una columna sean iguales
            df[col] = (df[col] - min_val) / range_val

    logger.info("Estandarización con Min-Max Scaler completada!")

    return df


# 3. Conjutno entrenamiento, validación y prueba
def split_data(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en tres subconjuntos: entrenamiento, validación y prueba.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas que se dividirá.
    params: Dict[str, Any]
        Diccionario de parámetros model_input.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tupla con los DataFrames de entrenamiento, validación y prueba.
    """
    logger.info("Iniciando la división de datos en entrenamiento, validación y prueba...")

    test_size = params['train_test_split']['test_size']
    validation_size = params['train_test_split']['validation_size']
    random_state = params['train_test_split']['random_state']
    shuffle = params['train_test_split']['shuffle']

    # Primero dividimos en entrenamiento+validación y prueba
    df_train_val, df_test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=shuffle)

    # Ahora dividimos el entrenamiento+validación en entrenamiento y validación
    val_size = validation_size / (1 - test_size)
    df_train, df_val = train_test_split(df_train_val, test_size=val_size, random_state=random_state, shuffle=shuffle)

    logger.info("División de datos en entrenamiento, validación y prueba completada!")

    return df_train, df_val, df_test

