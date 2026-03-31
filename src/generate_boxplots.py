import pandas as pd
import numpy as np

# --- CONFIGURACIÓN ---
FILE_NAME = 'comparison_model_results_no_context.csv'  # Asegúrate de que el nombre coincida
METRICS = ['levenshtein_similarity_ratio', 'f1_score', 'global_similarity']
# ---------------------

def calculate_boxplot_stats(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        results = []

        for metric in METRICS:
            if metric not in df.columns:
                continue
            
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
            for llm, group in df.groupby('llm'):
                # Extraemos los valores y quitamos nulos
                raw_data = group[metric].dropna().values
                if len(raw_data) == 0:
                    continue
                
                # CAMBIO CLAVE: np.sort() devuelve una COPIA ordenada, evitando el error de "read-only"
                data = np.sort(raw_data)
                
                # Cálculo de cuartiles
                q1 = np.percentile(data, 25)
                median = np.median(data)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                
                # Límites teóricos
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Bigotes (Whiskers): valores reales más extremos dentro de los límites
                lower_whisker = data[data >= lower_bound].min()
                upper_whisker = data[data <= upper_bound].max()
                
                # Valores atípicos (Outliers)
                outliers = data[(data < lower_whisker) | (data > upper_whisker)]
                
                results.append({
                    'LLM': llm,
                    'Metric': metric,
                    'Lower Whisker': round(lower_whisker, 4),
                    'Q1': round(q1, 4),
                    'Median': round(median, 4),
                    'Q3': round(q3, 4),
                    'Upper Whisker': round(upper_whisker, 4),
                    'Outliers': [round(x, 4) for x in outliers.tolist()]
                })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            print(f"\n📊 Estadísticos para {FILE_NAME}:")
            print("-" * 110)
            print(results_df.to_string(index=False))
        else:
            print("No se encontraron datos válidos para procesar.")

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{file_path}'.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    calculate_boxplot_stats(FILE_NAME)