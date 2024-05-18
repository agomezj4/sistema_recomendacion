from typing import Dict, Any

import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#1. Realizar join entre las 2 fuentes de datos
def left_join_dfs(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Realiza un left join entre dos dataframes de pandas basándose en una columna especificada en los parámetros.

    Parameters
    ----------
    df1 : pd.DataFrame
         Dataframe transaccional (izquierdo en el join).
    df2 : pd.DataFrame
        Dataframe demográfico (derecho en el join).
    params : Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pd.DataFrame
        Dataframe resultante del left join.
    """

    # Registra un mensaje de información indicando el inicio del proceso de left join
    logger.info("Iniciando el proceso de left join entre los dataframes...")

    # Parámetros
    join_column = params['join_column'][0]

    # Validación de que la columna existe en ambos dataframes
    if join_column not in df1.columns or join_column not in df2.columns:
        raise ValueError(f"La columna '{join_column}' no se encuentra en ambos dataframes.")

    # Eliminar duplicados en el dataframe demográfico
    df2_unique = df2.drop_duplicates(subset=[join_column])

    # Realización del left join
    result_df = df1.merge(df2_unique, how='left', on=join_column)

    # Registra un mensaje de información indicando que el join se ha completado
    logger.info("Left join completado con éxito!")

    return result_df


#2. cambiar tipado
def change_dtypes(
        df: pd.DataFrame,
        params: Dict[str, str]
) -> pd.DataFrame:
    """
    Cambia el tipo de datos de las columnas especificadas en un DataFrame de pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cuyos tipos de columnas serán cambiados.
    params : Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pd.DataFrame
        DataFrame con los tipos de datos de las columnas modificados.
    """

    # Registra un mensaje de información indicando el inicio del proceso de cambio de tipos de datos
    logger.info("Iniciando el proceso de cambio de tipos de datos...")

    # Parámetros
    cols = params.get('cols', {})

    # Itera sobre los parámetros para cambiar los tipos de datos de las columnas
    for col, dtype in cols.items():
        if col in df.columns:
            try:
                if dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"Columna '{col}' convertida a datetime.")
                elif dtype == 'object':
                    df[col] = df[col].astype('object')
                    logger.info(f"Columna '{col}' convertida a object.")
                else:
                    logger.warning(f"Tipo de dato '{dtype}' no reconocido para la columna '{col}'.")
            except Exception as e:
                logger.error(f"Error al convertir la columna '{col}' a {dtype}: {e}")
        else:
            logger.warning(f"La columna '{col}' no existe en el DataFrame.")

    # Registra un mensaje de información indicando que el proceso se ha completado
    logger.info("Proceso de cambio de tipos de datos completado con éxito!")

    return df


#3. Imputar categorías nulas
def impute_missing_categories(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Imputa las categorías nulas en un DataFrame basándose en las categorías conocidas para cada producto.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con una columna 'CATEGORIA' que puede contener valores nulos y una columna 'COD_PRODUCTO'.
    params : Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pd.DataFrame
        DataFrame con las categorías nulas imputadas.
    """
    logger.info("Iniciando la imputación de categorías nulas...")

    # Parámetros
    columns = params['columns']

    # Paso 1: Identificar los productos con categorías nulas
    categorias_nulas = df[df[columns[0]].isnull()]
    logger.info(f"Número de categorías nulas: {len(categorias_nulas)}")

    # Paso 2: Buscar categorías conocidas para esos productos
    categorias_conocidas = df[~df[columns[0]].isnull()]
    categoria_por_producto = categorias_conocidas.groupby(columns[1])[columns[0]].first().reset_index()

    # Paso 3: Imputar las categorías nulas
    df_imputado = pd.merge(df, categoria_por_producto, on=columns[1], how='left', suffixes=('', '_imputada'))
    df_imputado[columns[0]] = df_imputado[columns[0]].fillna(df_imputado['CATEGORIA_imputada'])
    df_imputado = df_imputado.drop(columns=['CATEGORIA_imputada'])

    logger.info(f"Número de categorías nulas después de imputar: {len(df_imputado[df_imputado[columns[0]].isnull()])}\n")

    logger.info("Imputación de categorías nulas completada con éxito!")

    return df_imputado


def get_mode(x: pd.Series) -> Any:
    """
    Obtiene la moda de una serie de pandas.

    Parameters
    ----------
    x : pd.Series
        Serie de pandas de la cual se obtendrá la moda.

    Returns
    -------
    Any
        La moda de la serie. Si la serie está vacía, retorna None.
    """
    moda = x.mode()
    return moda[0] if not moda.empty else None


def impute_by_mode(
        df: pd.DataFrame,
        strat_columns: list,
        target_columns: list
) -> pd.DataFrame:
    """
    Imputa valores faltantes en un DataFrame usando la moda estratificada por columnas específicas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con posibles valores ausentes.
    strat_columns : list
        Lista de columnas para estratificar la imputación.
    target_columns : list
        Lista de columnas a imputar.

    Returns
    -------
    pd.DataFrame
        DataFrame con los valores imputados.
    """
    for column in target_columns:
        # Calcular completitud antes de imputar
        completitud = df[column].notnull().mean()
        if completitud < 0.75:
            raise ValueError(f"La completitud de la columna '{column}' es menor a 0.75 y no se puede imputar.")

        # Obtener la moda para cada grupo
        modas = df.groupby(strat_columns)[column].apply(get_mode).reset_index(name='moda')

        # Merge con el DataFrame original para imputar los valores faltantes
        df = df.merge(modas, on=strat_columns, how='left')

        # Usar la columna de moda para imputar los valores faltantes
        df[column] = df[column].fillna(df['moda'])

        # Eliminar la columna auxiliar de moda
        df.drop(columns=['moda'], inplace=True)

        # Asegurar la tipificación correcta de los datos
        df[column] = df[column].infer_objects(copy=False)

    return df


def impute_missing_values(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Imputa los valores ausentes en un DataFrame siguiendo dos procesos:
    1. Imputar la columna 'CATEGORIA'.
    2. Imputar de forma estratificada por la moda.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con posibles valores ausentes.
    params : Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pd.DataFrame
        DataFrame con los valores ausentes imputados.
    """
    logger.info("Iniciando el proceso de imputación de valores ausentes...")

    # Parámetros
    strat_columns = params['columns']
    target_columns = params['target_columns']

    # Imputar de forma estratificada por la moda
    df = impute_by_mode(df, strat_columns, target_columns)

    logger.info("Proceso de imputación completado con éxito!")

    return df

