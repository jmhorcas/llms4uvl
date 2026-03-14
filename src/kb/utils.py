import json
import os

import numpy
from collections import Counter, defaultdict
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity












# def flatten_predicate(pred: str) -> str:
#     """Normalize the predicate by converting it to lowercase and mapping it to a standard form if it matches certain patterns."""
#     pred = pred.lower().strip()
#     if pred in ['isa', 'typeof', 'is']: return 'is_a'
#     if pred in ['partof', 'includedin', 'memberof']: return 'part_of'
#     if pred in ['defines', 'supports', 'allows', 'has']: return 'defines'
#     return pred




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




