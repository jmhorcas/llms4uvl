import ollama


PROMPT_TEMPLATE_FILEPATH = '../resources/prompts/prompt01_KnowledgeElicitation.txt'

def get_prompt(seed: str) -> str:
    with open(PROMPT_TEMPLATE_FILEPATH, 'r') as f:
        template = f.read()
    return template.replace("[INSERT_CONCEPT_HERE]", seed)

def extract_triplets_local(seed, model="gemma3:1b"):
    prompt = get_prompt(seed)
    
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']

# # Ahora puedes meter esto en un bucle:
# seeds = ["UVL Language", "Feature Cardinality", "Constraints"]
# for s in seeds:
#     for run in range(1, 6):
#         raw_output = extract_triplets_local(s)
#         # Aquí llamarías a tu clase UVLKnowledgeBase para guardar

def main() -> None:
    raw_output = extract_triplets_local("UVL Language")
    print(raw_output)

if __name__ == "__main__":
    main() 