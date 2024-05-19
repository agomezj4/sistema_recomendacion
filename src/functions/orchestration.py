import pandas as pd
import os
import sys
import yaml
import pickle

# Ajustar el path del sistema para incluir el directorio 'src' al nivel superior
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Importar funciones de procesamiento
from functions.processing import (left_join_dfs,
                                  change_dtypes,
                                  impute_missing_categories,
                                  impute_missing_values)

# Importar funciones de ingeniería de características
from functions.featuring import (features_new,
                                 rfm,
                                 add_segment_fl)

# Importar funciones de model input
from functions.model_input import (transform_rfm, min_max_scaler, split_data)

# Importar funciones de modeling
from functions.modeling import (train_als, grid_search_als)

# Directorios para los archivos de parámetros y los datos
parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')
data_processed_directory = os.path.join(project_root, 'data', '02_processed')
data_featured_directory = os.path.join(project_root, 'data', '03_featured')
data_train_directory = os.path.join(project_root, 'data', '04_model_input', 'train')
data_test_directory = os.path.join(project_root, 'data', '04_model_input', 'test')
data_validation_directory = os.path.join(project_root, 'data', '04_model_input', 'validation')
modeling_directory = os.path.join(project_root, 'data', '05_modeling')

# Lista todos los archivos YAML en el directorio especificado
yaml_files = [f for f in os.listdir(parameters_directory) if f.endswith('.yml')]

# Diccionario para guardar los parámetros cargados
parameters = {}

# Carga cada archivo YAML
for yaml_file in yaml_files:
    with open(os.path.join(parameters_directory, yaml_file), 'r') as file:
        data = yaml.safe_load(file)
        key_name = f'parameters_{yaml_file.replace(".yml", "")}'
        parameters[key_name] = data


# Pipeline de procesamiento
def run_processing():

    # Cargar datos
    raw_data_set1_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_set1_path'])
    raw_data_set2_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_set2_path'])

    data_set1_raw = pd.read_csv(raw_data_set1_path)  # transaccional
    data_set2_raw = pd.read_csv(raw_data_set2_path)  # demográfico


    # Realizar join entre las 2 fuentes de datos
    data_raw = left_join_dfs(data_set1_raw, data_set2_raw, parameters['parameters_processing'])

    # Cambiar tipos de datos
    data_change_dtypes = change_dtypes(data_raw, parameters['parameters_processing'])

    # Imputar valores faltantes en categorías
    data_impute_missing_categories = impute_missing_categories(data_change_dtypes, parameters['parameters_processing'])

    # Imputar valores faltantes
    data_processing = impute_missing_values(data_impute_missing_categories, parameters['parameters_processing'])

    # Guardar datos procesados
    processed_data_path = os.path.join(data_processed_directory,
                                       parameters['parameters_catalog']['processed_data_path'])
    data_processing.to_parquet(processed_data_path, index=False)


# Pipeline de ingeniería de características
def run_featuring():

    # Cargar datos
    processed_data_path = os.path.join(data_processed_directory, parameters['parameters_catalog']['processed_data_path'])
    data_processed = pd.read_parquet(processed_data_path)

    # Crear nuevas características
    data_features_new = features_new(data_processed, parameters['parameters_featuring'])

    # Crear RFM
    rfm_table = rfm(data_features_new, parameters['parameters_featuring'])

    # Agregar segmento filtros colaborativos al conjunto de datos
    data_featured = add_segment_fl(data_features_new, rfm_table, parameters['parameters_featuring'])

    # Guardar datos con características
    featured_data_path = os.path.join(data_featured_directory, parameters['parameters_catalog']['featured_data_path'])
    rfm_table_path = os.path.join(data_featured_directory, parameters['parameters_catalog']['rfm_table_path'])

    data_featured.to_parquet(featured_data_path, index=False)
    rfm_table.to_parquet(rfm_table_path, index=False)

# Pipeline de model input
def run_model_input():

    # Cargar datos
    featured_data_path = os.path.join(data_featured_directory, parameters['parameters_catalog']['featured_data_path'])

    data_featured = pd.read_parquet(featured_data_path)

    # Transformar RFM
    data_transform_rfm = transform_rfm(data_featured, parameters['parameters_featuring'])

    # Escalar los datos
    data_min_max_scaler = min_max_scaler(data_transform_rfm, parameters['parameters_model_input'])

    # Dividir los datos en entrenamiento, validación y prueba
    train_data, val_data, test_data = split_data(data_min_max_scaler, parameters['parameters_model_input'])

    # Guardar los datos de entrenamiento, validación y prueba
    train_data_path = os.path.join(data_train_directory, parameters['parameters_catalog']['train_data_path'])
    val_data_path = os.path.join(data_validation_directory, parameters['parameters_catalog']['validation_data_path'])
    test_data_path = os.path.join(data_test_directory, parameters['parameters_catalog']['test_data_path'])

    train_data.to_parquet(train_data_path, index=False)
    val_data.to_parquet(val_data_path, index=False)
    test_data.to_parquet(test_data_path, index=False)

# Pipeline de modelado
def run_modeling():

    # Cargar datos
    train_data_path = os.path.join(data_train_directory, parameters['parameters_catalog']['train_data_path'])
    train_data = pd.read_parquet(train_data_path)

    validation_data_path = os.path.join(data_validation_directory, parameters['parameters_catalog']['validation_data_path'])
    validation_data = pd.read_parquet(validation_data_path)

    # Entrenar modelo ALS
    models_als = train_als(train_data, parameters['parameters_modeling'])

    # Realizar grid search
    bests_models_als = grid_search_als(models_als, validation_data, parameters['parameters_modeling'])

    # Guardar modelo
    bests_models_path = os.path.join(modeling_directory, parameters['parameters_catalog']['bests_models_path'])
    with open(bests_models_path, 'wb') as f:
        pickle.dump(bests_models_als, f)


