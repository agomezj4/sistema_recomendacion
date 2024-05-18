import sys
import os

# Añadir la carpeta 'src' al path para poder importar los módulos
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'src'))

# Importar funciones de orquestación
from functions.orchestration import (run_processing)

def main():
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        if stage == 'all pipelines':
            run_processing()

        elif stage == 'preparation_pipeline':  # Corregido aquí
            run_processing()

        else:
            print(f"Etapa '{stage}' no reconocida. Las etapas válidas son: "
                  f"'all pipelines', "
                  f"'preparation_pipeline'")
    else:
        print("No se especificó una etapa. Uso: python __main__.py [etapa]")

if __name__ == "__main__":
    main()
