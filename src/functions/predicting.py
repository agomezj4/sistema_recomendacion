from typing import Dict, Any

import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Salida de modelos
def predicting_als(
    best_models: Dict[str, AlternatingLeastSquares], 
    df_input: pd.DataFrame, 
    params: Dict[str, Any], 
    num_recommendations: int = 1
) -> pd.DataFrame:
    """
    Genera un DataFrame con las recomendaciones para cada usuario.

    Parameters
    ----------
    best_models : Dict[str, AlternatingLeastSquares]
        Diccionario de modelos ALS optimizados, segmentados por claves.
    df_input : pandas.DataFrame
        DataFrame de pandas que contiene los datos de entrada de los clientes y productos.
    params: Dict[str, Any]
        Diccionario de parámetros modeling.
    num_recommendations : int, optional
        Número de recomendaciones a generar para cada usuario, por defecto 1.

    Returns
    -------
    pandas.DataFrame: DataFrame con las recomendaciones para cada usuario.
    """
    logger.info("Iniciando la generación de recomendaciones...")

    # Parámetros
    user_col = params['train_als']['user_col']
    item_col = params['train_als']['item_col']
    segment_col = params['train_als']['segmento_fc']  # Columna para segmentar

    all_recommendations = []
    user_out_of_bounds_count = {}
    item_out_of_bounds_count = {}

    # Iterar sobre los segmentos y modelos
    for segment, model in best_models.items():
        logger.info(f"Generando recomendaciones para el segmento: {segment}")
        df_segment = df_input[df_input[segment_col] == segment]

        # Inicializar contadores para el segmento actual
        user_out_of_bounds_count[segment] = 0
        item_out_of_bounds_count[segment] = 0

        # Mapear los códigos de usuario y producto para que coincidan con los del conjunto de entrenamiento
        user_mapper = {cat: idx for idx, cat in enumerate(df_segment[user_col].astype('category').cat.categories)}
        item_mapper = {idx: cat for idx, cat in enumerate(df_segment[item_col].astype('category').cat.categories)}

        # Iterar sobre los usuarios del segmento
        for user_id in df_segment[user_col].unique():
            if user_id not in user_mapper:
                continue
            user_code = user_mapper[user_id]

            # Verificar que el código de usuario esté dentro de los límites
            if user_code >= model.user_factors.shape[0]:
                user_out_of_bounds_count[segment] += 1
                continue

            # Crear una matriz vacía para las interacciones del usuario con los ítems
            user_interactions = csr_matrix((1, model.item_factors.shape[0]))

            # Generar recomendaciones
            recommendations = model.recommend(user_code, user_interactions, N=num_recommendations)
            recommended_items = []
            for item in recommendations:
                item_id = item[0]
                if item_id in item_mapper:
                    recommended_items.append(item_mapper[item_id])
                else:
                    item_out_of_bounds_count[segment] += 1
                    recommended_items.append(None)  # Añadir None si el ítem está fuera de los límites

            # Asegurarse de que haya suficientes ítems recomendados
            while len(recommended_items) < num_recommendations:
                recommended_items.append(None)

            # Encontrar el ítem más frecuente
            most_frequent_item = df_segment[df_segment[user_col] == user_id][item_col].mode().values[0]

            # Agregar las recomendaciones a la lista
            all_recommendations.append({
                'customer_id': user_id,
                'fc_segment': segment,
                'recommendation_1': recommended_items[0],
                'most_frequent_item': most_frequent_item
            })

    # Convertir la lista de recomendaciones en un DataFrame
    recommendations_df = pd.DataFrame(all_recommendations)

    # Loguear los contadores de usuarios y ítems fuera de los límites por segmento
    for segment in best_models.keys():
        logger.info(f"Segmento {segment}: {user_out_of_bounds_count[segment]} usuarios fuera de los límites.")
        logger.info(f"Segmento {segment}: {item_out_of_bounds_count[segment]} ítems recomendados fuera de los límites.")

    logger.info("Finalizada la generación de recomendaciones!")

    return recommendations_df










