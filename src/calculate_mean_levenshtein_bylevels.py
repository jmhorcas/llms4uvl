import pandas as pd

def calcular_medias_por_nivel(file_path):
    try:
        # 1. Cargar el CSV
        df = pd.read_csv(file_path)
        
        # Limpiar espacios en los nombres de las columnas
        df.columns = df.columns.str.strip()

        # 2. Función para simplificar el 'language_level'
        def categorizar_nivel(nivel):
            if pd.isna(nivel):
                return None
            nivel = str(nivel).strip().replace('"', '')  # Eliminar comillas si existen
            if "Boolean" in nivel:
                return "Boolean"
            elif "Arithmetic" in nivel or 'Attributes' in nivel:
                return "Arithmetic"
            elif "Type" in nivel:
                return "Type"
            return None

        # Aplicar la categorización
        df['nivel_simplificado'] = df['model'].apply(categorizar_nivel)
        # 3. Asegurar que f1_score sea numérico (convierte vacíos a NaN)
        df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')

        # 4. Agrupar por LLM y por el nuevo nivel simplificado
        # Usamos dropna=True para ignorar filas que no entren en las 3 categorías
        medias = df.groupby(['llm', 'nivel_simplificado'])['f1_score'].mean().unstack()
        # 5. Formatear y mostrar resultados
        print("\n📊 Media de F1-Score por LLM y Nivel de Lenguaje:")
        print("-" * 60)
        # Reordenar columnas para que aparezcan en el orden lógico si existen
        column_order = [c for c in ["Boolean", "Arithmetic", "Type"] if c in medias.columns]
        print(medias[column_order].round(4).fillna("Sin datos"))
        
        return medias

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{file_path}'.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    # Sustituye 'tus_datos.csv' por el nombre de tu archivo
    calcular_medias_por_nivel('comparison_model_results_all.csv')