import json
import os

import numpy
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity



SPACY_LANGUAGE_MODEL = "en_core_web_sm"
TRANSFORMER_LANGUAGE_MODEL = 'all-MiniLM-L6-v2'  # Light and free model (approx 80MB)


def setup_custom_tokenizer(nlp):
    """Set up a custom tokenizer for the spaCy language model to ensure that certain special cases (like "==", "=>", "<=", "!=", etc.) are treated as single tokens rather than being split into multiple tokens."""
    special_cases = ["==", ">=", "<=", "!=", "=>", "<=>"]
    for case in special_cases:
        nlp.tokenizer.add_special_case(case, [{spacy.symbols.ORTH: case}])
    return nlp


def initialize_language_models() -> tuple[spacy.Language, SentenceTransformer]:
    """Initialize the language models and resources needed for text processing and similarity calculations."""
    nltk.download('wordnet', quiet=True)
    nlp = spacy.load(SPACY_LANGUAGE_MODEL)
    nlp = setup_custom_tokenizer(nlp)
    language_model = SentenceTransformer(TRANSFORMER_LANGUAGE_MODEL)
    return nlp, language_model

    
def case_folding(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower().strip()


def lemmatization(text: str, nlp: spacy.Language) -> str:
    """Lemmatize the text (convert words to their base form)."""
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)


def remove_stopwords(text: str, nlp: spacy.Language) -> str:
    """Remove stop words (articles, prepositions, etc.) from the text."""
    doc = nlp(text)
    keep_tokens = [t for t in doc if not t.is_stop and not t.is_punct]
    if not keep_tokens:
        return text  # Return original text if all tokens are removed
    result = "".join([t.text_with_ws for t in keep_tokens]).strip()  # Preserve original spacing
    return result if result else text  # Return original text if result is empty after removing stop words


def get_synonyms(word: str, nlp: spacy.Language) -> list:
    """Get synonyms for a given word using the specified language model."""
    doc = nlp(word)
    synonyms = set()
    for token in doc:
        for syn in nltk.corpus.wordnet.synsets(token.lemma_):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return list(synonyms)


def get_established_concept(text: str, concept_mapping_file: str) -> str:
    """Map a given text to an established concept based on a provided mapping.
    
    If the text is not found in the mapping, it returns the original text.
    """
    with open(concept_mapping_file, 'r', encoding='utf-8') as f:
        concept_mapping = json.load(f)
    return concept_mapping.get(text, text)


def normalize_text(text: str, concept_mapping_file: str, nlp: spacy.Language) -> str:
    """Normalize the text by applying case folding, concept mapping, stop word removal, and lemmatization."""
    original_text = text
    text = case_folding(text)
    mapped_concept = get_established_concept(text, concept_mapping_file)
    if mapped_concept != text:
        return mapped_concept  # Return the mapped concept if found
    
    text = remove_stopwords(text, nlp)
    text = lemmatization(text, nlp)
    mapped_concept = get_established_concept(text, concept_mapping_file)
    if text == '':
        return original_text  # Return the original text if no mapping found for it
    if mapped_concept != text:
        return mapped_concept  # Return the mapped concept if found after normalization
    return text  # Return the normalized text if no mapping found for it


# def flatten_predicate(pred: str) -> str:
#     """Normalize the predicate by converting it to lowercase and mapping it to a standard form if it matches certain patterns."""
#     pred = pred.lower().strip()
#     if pred in ['isa', 'typeof', 'is']: return 'is_a'
#     if pred in ['partof', 'includedin', 'memberof']: return 'part_of'
#     if pred in ['defines', 'supports', 'allows', 'has']: return 'defines'
#     return pred


def get_similarity(text1: str, 
                   text2: str,
                   language_model: SentenceTransformer, 
                   weight_semantic: float = 0.7, 
                   weight_lexical: float = 0.3) -> float:
    """Compute the similarity between two texts by combining Semantic Similarity (using embeddings) and Lexical Similarity (using Fuzzy Matching).
    
    This method is designed to be robust for comparing triplets as sentences, where the order of words may vary but the meaning is similar (e.g., "UVL supports Boolean" vs "Boolean is part of UVL").

    Args:
        text1, text2: Texts to compare (e.g., concatenated triplets).
        weight_semantic: Weight given to semantic similarity (0.0 to 1.0).
        weight_lexical: Weight given to lexical similarity (0.0 to 1.0).
    """
    if not text1 or not text2:
        return 0.0

    # 1. Semantic Similarity (based on the language model)
    # Convert the texts into vectors and compute cosine similarity
    embeddings = language_model.encode([text1, text2])
    sem_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
 
    # 2. Lexical Similarity (based on characters/Fuzzy Matching)
    # token_set_ratio is excellent because it ignores word order
    # Example: "UVL Boolean" vs "Boolean UVL" would return 100%
    lex_score = fuzz.token_set_ratio(text1, text2) / 100.0
 
    # 3. Combined Score
    final_score = (sem_score * weight_semantic) + (lex_score * weight_lexical)
    
    return float(final_score)


def get_atomic_similarity(triplet1: tuple[str, str, str], 
                          triplet2: tuple[str, str, str],
                          language_model: SentenceTransformer,
                          weight_subject: float = 0.4, 
                          weight_predicate: float = 0.2, 
                          weight_object: float = 0.4) -> float:
    """Compute the similarity between two triplets by comparing their subject, predicate, and object separately and then combining the scores.
    
    This method is designed to be more fine-grained, as it evaluates the similarity of each component of the triplet independently, which can be useful when the order of words in the predicate varies but the meaning is preserved.
    
    Args:
        triple1, triple2: Triplets to compare (e.g., tuples of strings with subject, predicate, object).
        weight_subject: Weight given to the similarity of the subjects (0.0 to 1.0).
        weight_predicate: Weight given to the similarity of the predicates (0.0 to 1.0).
        weight_object: Weight given to the similarity of the objects (0.0 to 1.0).
    """
    # Split the triplets into their components
    s1, p1, o1 = triplet1
    s2, p2, o2 = triplet2

    # Generate embeddings for each component separately
    embs = language_model.encode([s1, s2, p1, p2, o1, o2])
    
    # Compute similarity for each component
    sim_s = cosine_similarity([embs[0]], [embs[1]])[0][0]
    sim_p = cosine_similarity([embs[2]], [embs[3]])[0][0]
    sim_o = cosine_similarity([embs[4]], [embs[5]])[0][0]
    
    # Ponderation: Subject and object are the factual knowledge (0.4 each)
    # Predicate is the "conector" (0.2)
    final_score = (sim_s * weight_subject) + (sim_o * weight_object) + (sim_p * weight_predicate)
    
    return final_score




def get_hybrid_similarity(triple1: tuple[str, str, str], 
                          triple2: tuple[str, str, str], 
                          language_model: SentenceTransformer) -> float:
    """Compute the similarity between two triplets using a hybrid strategy that combines both global and atomic similarities.

    Use the global similarity to capture the overall meaning of the triplet as a sentence, and the atomic similarity to ensure that the individual components (subject, predicate, object) are also similar.
    Take the maximum of both scores to get a robust similarity measure that can handle variations in wording and structure while still capturing the underlying meaning. 
    This is particularly useful for comparing triplets where the same knowledge can be expressed in different ways (e.g., "UVL supports Boolean" vs "Boolean is part of UVL").
    """
    # 1. Preparar las frases globales
    phrase1 = f"{triple1[0]} {triple1[1]} {triple1[2]}"
    phrase2 = f"{triple2[0]} {triple2[1]} {triple2[2]}"
    
    # --- SIMILITUD GLOBAL (Semántica + Léxica) ---
    # Obtenemos embeddings de las frases completas
    embs_global = language_model.encode([phrase1, phrase2])
    sem_global = cosine_similarity([embs_global[0]], [embs_global[1]])[0][0]
    
    # Similitud léxica (Fuzzy) de la frase completa
    lex_global = fuzz.token_set_ratio(phrase1, phrase2) / 100.0
    
    # Score Global: Combinación ponderada (70% semántico, 30% léxico)
    score_global = (sem_global * 0.7) + (lex_global * 0.3)
    
    # --- SIMILITUD ATÓMICA (S, P, O por separado) ---
    # Creamos embeddings para cada componente
    # s1, p1, o1 vs s2, p2, o2
    embs_atomic = language_model.encode([
        triple1[0], triple2[0], # Sujetos
        triple1[1], triple2[1], # Predicados
        triple1[2], triple2[2]  # Objetos
    ])
    
    sim_s = cosine_similarity([embs_atomic[0]], [embs_atomic[1]])[0][0]
    sim_p = cosine_similarity([embs_atomic[2]], [embs_atomic[3]])[0][0]
    sim_o = cosine_similarity([embs_atomic[4]], [embs_atomic[5]])[0][0]
    
    # Score Atómico: Ponderamos Sujeto y Objeto más que el Predicado
    score_atomic = (sim_s * 0.4) + (sim_o * 0.4) + (sim_p * 0.2)
    
    # --- RESULTADO FINAL: ESTRATEGIA DE MÁXIMO ---
    # Nos quedamos con la mejor evidencia de similitud encontrada
    final_score = max(score_global, score_atomic)
    
    return float(final_score)


def fast_semantic_deduplication(language_model: SentenceTransformer,
                                triples: list[tuple[str, str, str]],  
                                threshold: float = 0.92) -> list[tuple[str, str, str]]:
    if not triples:
        return []

    # 1. Convertir tripletas a frases y vectorizar EN BLOQUE (Muy rápido)
    phrases = [f"{s} {p} {o}" for s, p, o in triples]
    embeddings = language_model.encode(phrases) # Una sola llamada al modelo

    # 2. Calcular matriz de similitud (N x N)
    # Esto compara todas contra todas en milisegundos usando álgebra lineal
    sim_matrix = cosine_similarity(embeddings)

    # 3. Identificar duplicados
    indices_to_remove = set()
    n = len(triples)

    for i in range(n):
        if i in indices_to_remove:
            continue
        
        # Buscamos índices j que sean muy similares a i
        # Solo miramos la mitad superior de la matriz (j > i) para no compararnos con nosotros mismos
        duplicate_indices = numpy.where(sim_matrix[i, i+1:] >= threshold)[0] + (i + 1)
        
        for idx in duplicate_indices:
            indices_to_remove.add(idx)

    # 4. Construir la lista final filtrada
    unique_triples = [triples[i] for i in range(n) if i not in indices_to_remove]

    #print(f"Original: {n} | Únicos: {len(unique_triples)} | Eliminados: {len(indices_to_remove)}")
    return unique_triples