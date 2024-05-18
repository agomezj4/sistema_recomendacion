from typing import Dict, Any

import pandas as pd
import logging
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#1. Ingeniería de características
def features_new(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula nuevas caracteristicas para cada cliente en el dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas que contiene las caracteristicas de los clientes.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pd.DataFrame: DataFrame con las nuevas columnas agregadas.
    """
    logger.info("Iniciando el cálculo de nuevas características...")

    # Parámetros
    edad = params['edad']
    grupo_etario = params['grupo_etario']
    bins_grupo_etario = params['bins_grupo_etario']
    labels_grupo_etario = params['labels_grupo_etario']
    generacion = params['generacion']
    bins_generacion = params['bins_generacion']
    labels_generacion = params['labels_generacion']
    regiones = params['regiones']
    categorias_regiones = params['categorias_regiones']
    fechas = params['fechas']

    # Fecha actual
    tiempo_actual = datetime.now()

    # Copia del dataframe original para no modificarlo
    df_copy = df.copy()

    # Calcula la edad y la antigüedad del cliente en años
    df_copy[edad[0]] = (tiempo_actual - df_copy[edad[1]]).dt.days // 365

    # Crea columnas 'grupo_etario' y 'generacion'
    df_copy[grupo_etario[0]] = pd.cut(df_copy[edad[0]], bins=bins_grupo_etario,
                                      labels=labels_grupo_etario)

    df_copy[generacion[0]] = pd.cut(df_copy[edad[0]], bins=bins_generacion,
                                    labels=labels_generacion, right=False)

    # Crea la nueva columna 'regiones'
    df_copy[regiones[0]] = df_copy[regiones[1]].map(categorias_regiones)

    # Crea columna 'periodo_compra'
    df_copy[fechas[0]] = df_copy[fechas[1]].dt.strftime('%Y%m')

    logger.info("Finalizado el cálculo de nuevas características!")

    return df_copy


#2. Calculo del RFM
def calculate_recency(
        df: pd.DataFrame,
        latest_date: pd.Timestamp,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula la recencia de las compras de los clientes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las transacciones de los clientes.
    latest_date : pd.Timestamp
        La fecha más reciente en el dataset.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pd.DataFrame
        DataFrame con la recencia calculada.
    """
    logger.info("Calculando la recencia...")

    # Parámetros
    recencia = params['recencia'][0]
    fecha = params['fechas']

    # Calcula la recencia en días
    df[recencia] = (latest_date - df[fecha[1]]).dt.days

    logger.info("Recencia calculada con éxito!\n")

    return df


def calculate_frequency(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula la frecuencia de compras de los clientes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las transacciones de los clientes.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pd.DataFrame
        DataFrame con la frecuencia calculada.
    """
    logger.info("Calculando la frecuencia...")

    # Parámetros
    id_cliente = params['id_cliente'][0]
    fecha = params['fechas'][0]
    pedido = params['pedido'][0]
    frecuencia = params['frecuencia'][0]

    # Calcula la frecuencia de compras de los clientes
    frequency = df.groupby([id_cliente, fecha])[pedido].nunique().reset_index()

    # Renombra las columnas
    frequency.columns = [id_cliente, fecha, frecuencia]

    # Realiza un merge con el dataframe original
    df = pd.merge(df, frequency, on=[id_cliente, fecha], how='left')

    logger.info("Frecuencia calculada con éxito!\n")

    return df


def calculate_monetary(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula el monto total gastado por los clientes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las transacciones de los clientes.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pd.DataFrame
        DataFrame con el monto total calculado.
    """
    logger.info("Calculando el monto total...")

    # Parámetros
    id_cliente = params['id_cliente'][0]
    fecha = params['fechas'][0]
    monto = params['monto']

    # Calcula el monto total gastado por los clientes
    monetary = df.groupby([id_cliente, fecha])[monto[1]].sum().reset_index()

    # Renombra las columnas
    monetary.columns = [id_cliente, fecha, monto[0]]

    # Realiza un merge con el dataframe original
    df = pd.merge(df, monetary, on=[id_cliente, fecha], how='left')

    logger.info("Monto total calculado con éxito!\n")

    return df


def assign_rfm_scores(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Asigna los puntajes RFM a los clientes en base a quintiles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las métricas RFM calculadas.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pd.DataFrame
        DataFrame con los puntajes RFM asignados.
    """
    logger.info("Asignando puntajes RFM...")

    # Parámetros
    recencia = params['recencia'][0]
    frecuencia = params['frecuencia'][0]
    monto = params['monto'][0]

    # Asigna los quintiles a las métricas RFM
    df['R_quintil'] = pd.qcut(df[recencia], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_quintil'] = pd.qcut(df[frecuencia].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_quintil'] = pd.qcut(df[monto], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Crea el código RFM y el factor RFM
    df['RFM_Code'] = df['R_quintil'].astype(str) + df['F_quintil'].astype(str) + df['M_quintil'].astype(str)
    df['Factor_RFM'] = df[['R_quintil', 'F_quintil', 'M_quintil']].prod(axis=1)

    logger.info("Puntajes RFM asignados con éxito!\n")

    return df


def segment_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta a los clientes en base a su factor RFM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los puntajes RFM asignados.

    Returns
    -------
    pd.DataFrame
        DataFrame con la segmentación RFM aplicada.
    """
    logger.info("Segmentando clientes por su factor RFM...")

    # Segmenta a los clientes en base a su factor RFM
    def segment_customer(factor_rfm):
        if factor_rfm <= 6:
            return 'Lost Customers'
        elif factor_rfm <= 15:
            return 'At Risk'
        elif factor_rfm <= 24:
            return 'Frequent Customer'
        elif factor_rfm <= 36:
            return 'Promising'
        elif factor_rfm <= 48:
            return 'New Customers'
        elif factor_rfm <= 64:
            return 'Potential Loyalist'
        elif factor_rfm <= 99:
            return 'Loyal Customers'
        elif factor_rfm <= 125:
            return 'Champions'
        else:
            return 'Revisar'

    # Aplica la función de segmentación
    df['Segmento_RFM'] = df['Factor_RFM'].apply(segment_customer)

    logger.info("Segmentación de clientes por su factor RFM completada con éxito!\n")

    return df


def segment_fc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta a los clientes en base a su segmentación RFM en categorías de alto valor, estratégicos y potenciales.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con la segmentación RFM aplicada.

    Returns
    -------
    pd.DataFrame
        DataFrame con la segmentación FC aplicada.
    """
    logger.info("Segmentando clientes para Filtrado Colaborativo...")

    # Segmenta a los clientes en base a su segmentación RFM
    def segment_fc(segment_rfm):
        if segment_rfm in ['Champions', 'Loyal Customers']:
            return 'Alto Valor'
        elif segment_rfm in ['Potential Loyalist', 'New Customers', 'Promising']:
            return 'Estratégicos'
        else:
            return 'Potenciales'

    # Aplica la función de segmentación
    df['Segmento_FC'] = df['Segmento_RFM'].apply(segment_fc)

    logger.info("Segmentación FC aplicada con éxito!\n")

    return df


def rfm(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula la recencia, frecuencia y monto para cada cliente, asigna un código RFM, factor RFM, y segmenta a los clientes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las transacciones de los clientes.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pd.DataFrame
        DataFrame con las métricas RFM, código RFM, factor RFM y segmentaciones aplicadas.
    """
    logger.info("Iniciando el cálculo de RFM...\n")

    # Parámetros
    fecha = params['fechas']
    id_cliente = params['id_cliente'][0]
    pedido = params['pedido'][0]
    producto = params['producto']
    monto = params['monto']

    # Calcular la fecha más reciente en el dataset
    latest_date = df[fecha[1]].max()

    # Calcular Recencia, Frecuencia y Monto
    df = calculate_recency(df, latest_date, params)
    df = calculate_frequency(df, params)
    df = calculate_monetary(df, params)

    # Remover duplicados para asegurar unicidad
    rfm_table = df.drop_duplicates(subset=[id_cliente]).copy()

    # Asignar puntajes RFM
    rfm_table = assign_rfm_scores(rfm_table, params)

    # Segmentar a los clientes
    rfm_table = segment_rfm(rfm_table)

    # Segmentación FC
    rfm_table = segment_fc(rfm_table)

    # Filtrar columnas
    rfm_table.drop(columns=[pedido, fecha[1], producto[0], producto[1], producto[2], monto[1], fecha[2]], inplace=True)

    logger.info("Cálculo de RFM completado con éxito!\n")

    return rfm_table


#3. Selección de características