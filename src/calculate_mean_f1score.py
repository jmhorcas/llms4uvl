import pandas as pd

def calcular_media_f1(file_path):
    try:
        # 1. Leer el archivo CSV
        df = pd.read_csv(file_path)
        
        # Limpiar posibles espacios en blanco en los nombres de las columnas
        df.columns = df.columns.str.strip()
        
        # 2. Asegurar que f1_score sea numérico
        # errors='coerce' convertirá celdas vacías o texto inválido en NaN
        df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')
        
        # 3. Agrupar por LLM y calcular la media
        # dropna() en el cálculo asegura que los modelos sin datos no afecten la media
        medias = df.groupby('llm')['f1_score'].mean().reset_index()
        
        # Renombrar columna para claridad
        medias.rename(columns={'f1_score': 'mean_f1_score'}, inplace=True)
        
        print(f"--- Media de f1_score por LLM ({file_path}) ---")
        print(medias.to_string(index=False))
        
        return medias

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{file_path}'.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Ejecutar el script
if __name__ == "__main__":
    # Cambia 'tus_datos.csv' por el nombre real de tu archivo
    calcular_media_f1('comparison_model_results_kb.csv')