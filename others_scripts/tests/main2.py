from llama_cpp import Llama, LlamaGrammar

# Configuración
MODEL_PATH = "tu_modelo_1B_o_3B.gguf" # Ejemplo: phi-3-mini-4k-instruct.Q4_K_M.gguf
GRAMMAR_FILE = "uvl.gbnf"

def generate_uvl(domain):
    # Inicializar el modelo
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048)
    
    # Cargar la gramática GBNF
    with open(GRAMMAR_FILE, "r") as f:
        grammar_text = f.read()
    uvl_grammar = LlamaGrammar.from_string(grammar_text)

    # Prompt minimalista
    prompt = f"Task: Generate a UVL file for {domain}.\nOutput:\n"

    # Generación restringida
    response = llm(
        prompt=prompt,
        grammar=uvl_grammar,
        max_tokens=1000,
        temperature=0.1 # Temperatura baja para mayor precisión
    )

    return response['choices'][0]['text']

# Ejemplo de uso
print(generate_uvl("Smart Home System"))