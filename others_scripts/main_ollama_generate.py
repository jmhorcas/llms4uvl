import ollama


PROMPT_TEMPLATE_FILEPATH = '../resources/prompts/prompt02_GenerateModel.txt'

def get_prompt() -> str:
    with open(PROMPT_TEMPLATE_FILEPATH, 'r') as f:
        template = f.read()
    return template


def extract_triplets_local(model="gemma3:1b"):
    prompt = get_prompt()
    # full_prompt = f"""
    # {prompt}
    
    # TASK:
    # Generate a UVL model for the Pizza domain. Be technical and precise.
    # """
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.2, # Baja temperatura para evitar alucinaciones
                "top_p": 0.9
            }
        )
        return response['response']
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    raw_output = extract_triplets_local()
    print(raw_output)


if __name__ == "__main__":
    main() 