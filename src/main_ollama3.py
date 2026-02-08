import ollama
import logging

PROMPT_TEMPLATE_FILEPATH = '../resources/prompts/prompt01_KnowledgeElicitation.txt'

def get_prompt() -> str:
    return f"""CONTEXT: Software product line.
    TASK: Generate a UVL model.
    DOMAIN: pizzas.
    SIZE: 12 features.
    OUTPUT FORMAT: only UVL model text, no explanations.
    """

def extract_triplets_local(model="gemma3:1b"):
    prompt = get_prompt()
    
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']


def main() -> None:
    raw_output = extract_triplets_local()
    print(raw_output)


if __name__ == "__main__":
    main() 