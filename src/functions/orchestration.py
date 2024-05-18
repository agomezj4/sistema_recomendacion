import pandas as pd
import os
import sys
import yaml

# Ajustar el path del sistema para incluir el directorio 'src' al nivel superior
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Importar funciones de procesamiento
from functions.processing import (left_join_dfs,
                                  change_dtypes,
                                  impute_missing_categories,
                                  impute_missing_values)

# Directorios para los archivos de parámetros y los datos
parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')
data_processed_directory = os.path.join(project_root, 'data', '02_processed')

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
    data_processing.to_csv(processed_data_path, index=False)
