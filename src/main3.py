import requests
import json

def generar_uvl_caja_negra(dominio, ruta_gbnf="../resources/uvl.gbnf"):
    # 1. Leer la gramática que convertimos de tus .g4
    try:
        with open(ruta_gbnf, "r") as f:
            grammar_content = f.read()
    except FileNotFoundError:
        return "Error: No se encuentra el archivo .gbnf"

    # 2. Configurar la llamada directa a la API
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "llama3",  # O el modelo 1B que estés usando
        "prompt": f"Generate a valid UVL file for the domain: {dominio}. Use features and constraints.",
        "grammar": grammar_content,  # El servidor de Ollama acepta esto perfectamente
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024
        }
    }

    # 3. Realizar la petición
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"Error en la conexión con Ollama: {e}"

# --- PRUEBA ---
dominio_test = "Pizza Delivery System"
resultado = generar_uvl_caja_negra(dominio_test)

print(f"Resultado para {dominio_test}:")
print("-" * 30)
print(resultado)