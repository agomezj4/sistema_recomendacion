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

    # Copia del dataframe original para no modificarlo
    df_copy = df.copy()

    # Calcula la recencia en días
    df_copy[recencia] = (latest_date - df[fecha[1]]).dt.days

    # Validación
    initial_count = df.shape[0]
    final_count = df_copy.shape[0]
    if initial_count != final_count:
        logger.info(f"Recencia: Se perdieron {initial_count - final_count} registros")

    logger.info("Recencia calculada con éxito!\n")

    return df_copy


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
    initial_count = df.shape[0]
    df = pd.merge(df, frequency, on=[id_cliente, fecha], how='left')
    final_count = df.shape[0]
    if initial_count != final_count:
        logger.info(f"Frecuencia: Se perdieron {initial_count - final_count} registros")

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
    initial_count = df.shape[0]
    df = pd.merge(df, monetary, on=[id_cliente, fecha], how='left')
    final_count = df.shape[0]
    if initial_count != final_count:
        logger.info(f"Monto total: Se perdieron {initial_count - final_count} registros")

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
    initial_count = df.shape[0]
    df['R_quintil'] = pd.qcut(df[recencia], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_quintil'] = pd.qcut(df[frecuencia].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_quintil'] = pd.qcut(df[monto], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    final_count = df.shape[0]
    if initial_count != final_count:
        logger.info(f"Asignación de puntajes RFM: Se perdieron {initial_count - final_count} registros")

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
    initial_count = df.shape[0]
    df['Segmento_RFM'] = df['Factor_RFM'].apply(segment_customer)
    final_count = df.shape[0]
    if initial_count != final_count:
        logger.info(f"Segmentación RFM: Se perdieron {initial_count - final_count} registros")

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
    initial_count = df.shape[0]
    df['Segmento_FC'] = df['Segmento_RFM'].apply(segment_fc)
    final_count = df.shape[0]
    if initial_count != final_count:
        logger.info(f"Segmentación FC: Se perdieron {initial_count - final_count} registros")

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

    # Configuración de visualización
    pd.options.display.max_rows = None

    # Verificar cantidad de clientes únicos por período en el DataFrame de entrada
    clientes_por_periodo_inicial = df.groupby(fecha[0])[id_cliente].nunique().reset_index()
    logger.info(f"Clientes únicos por período en el DataFrame de entrada:\n{clientes_por_periodo_inicial}")

    # Calcular la fecha más reciente en el dataset
    latest_date = df[fecha[1]].max()

    # Calcular Recencia, Frecuencia y Monto
    df = calculate_recency(df, latest_date, params)
    logger.info(f"Número de clientes únicos después de calcular recencia: {df[id_cliente].nunique()}")

    df = calculate_frequency(df, params)
    logger.info(f"Número de clientes únicos después de calcular frecuencia: {df[id_cliente].nunique()}")

    df = calculate_monetary(df, params)
    logger.info(f"Número de clientes únicos después de calcular monto: {df[id_cliente].nunique()}")

    # Asignar puntajes RFM
    rfm_table = assign_rfm_scores(df, params)
    logger.info(f"Número de clientes únicos después de asignar puntajes RFM: {rfm_table[id_cliente].nunique()}")

    # Segmentar a los clientes
    rfm_table = segment_rfm(rfm_table)
    logger.info(f"Número de clientes únicos después de segmentar RFM: {rfm_table[id_cliente].nunique()}")

    # Segmentación FC
    rfm_table = segment_fc(rfm_table)
    logger.info(f"Número de clientes únicos después de segmentar FC: {rfm_table[id_cliente].nunique()}")

    # Verificar cantidad de clientes únicos por período en el DataFrame resultante
    clientes_por_periodo_final = rfm_table.groupby(fecha[0])[id_cliente].nunique().reset_index()
    logger.info(f"Clientes únicos por período en el DataFrame resultante:\n{clientes_por_periodo_final}")

    # Comparar las cantidades de clientes únicos por período
    if not clientes_por_periodo_inicial.equals(clientes_por_periodo_final):
        logger.warning("La cantidad de clientes únicos por período en el DataFrame resultante no coincide con la del DataFrame de entrada.")

    # Filtrar columnas
    rfm_table.drop(columns=[pedido, fecha[1], producto[0], producto[1], producto[2], monto[1], fecha[2]], inplace=True)

    logger.info("Cálculo de RFM completado con éxito!\n")

    return rfm_table


#3. Añadir segmento filtros colaborativos a conjunto de datos
def add_segment_fl(
        df1: pd.DataFrame, #conjunto de datos
        df2: pd.DataFrame, #tabla rfm
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Agrega un campo específico de df2 a df1 mediante un left join basado en las columnas 'id' y 'periodo'.

    Parameters
    ----------
    df1 : pd.DataFrame
         Dataframe principal al que se agregará el campo.
    df2 : pd.DataFrame
        Dataframe secundario del cual se obtendrá el campo.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye el campo específico a agregar.

    Returns
    -------
    pd.DataFrame
        Dataframe df1 con el campo específico adicional de df2.
    """

    # Registra un mensaje de información indicando el inicio del proceso
    logger.info("Iniciando el proceso de agregar segmento filtros colaborativos al conjunto de datos...")

    # Parámetros
    segmento_fl = params['segmento_fl'][0]
    id_cliente = params['id_cliente'][0]
    fecha = params['fechas'][0]
    recencia = params['recencia'][0]
    frecuencia = params['frecuencia'][0]
    monto = params['monto'][0]

    # Validación de que las columnas 'id' y 'periodo' existen en ambos dataframes
    if id_cliente not in df1.columns or id_cliente not in df2.columns:
        raise ValueError(f"La columna {id_cliente} no se encuentra en ambos dataframes.")
    if fecha not in df1.columns or fecha not in df2.columns:
        raise ValueError(f"La columna {fecha} no se encuentra en ambos dataframes.")

    # Validación de que el campo específico existe en df2
    if segmento_fl not in df2.columns:
        raise ValueError(f"La columna '{segmento_fl}' no se encuentra en df2.")

    # Selección de las columnas necesarias en df2
    df2_reduced = df2[[id_cliente, fecha, recencia,
                       frecuencia, monto, segmento_fl]].drop_duplicates(subset=[id_cliente, fecha])

    # Realización del left join
    result_df = df1.merge(df2_reduced, how='left', on=[id_cliente, fecha])

    # Registra un mensaje de información indicando que el join se ha completado
    logger.info("Campo específico agregado con éxito!")

    return result_df


