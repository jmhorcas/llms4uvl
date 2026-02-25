import json
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import nltk
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity



SPACY_LANGUAGE_MODEL = "en_core_web_sm"
TRANSFORMER_LANGUAGE_MODEL = 'all-MiniLM-L6-v2'  # Modelo ligero y gratuito (aprox 80MB)


nltk.download('wordnet')
nlp = spacy.load(SPACY_LANGUAGE_MODEL)
language_model = SentenceTransformer(TRANSFORMER_LANGUAGE_MODEL)

    
def case_folding(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower().strip()


def lemmatization(text: str) -> str:
    """Lemmatize the text (convert words to their base form)."""
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)


def remove_stopwords(text: str) -> str:
    """Remove stop words (articles, prepositions, etc.) from the text."""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    result = " ".join(filtered_tokens)
    if result.strip() == "":
        result = text  # Return original text if all tokens are removed (e.g., for syntax terms like "==", "=>", etc.)
    return result


def get_synonyms(word: str) -> list:
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


def normalize_text(text: str, concept_mapping_file: str) -> str:
    """Normalize the text by applying:
     (1) case folding, 
     (2) lemmatization, 
     (3) removing stopwords,
     (4) search for synonyms, and
     (5) concept mapping for each synonym (including the original word).
    
    The function returns the normalized text.
    """
    text = case_folding(text)
    text = lemmatization(text)
    text = remove_stopwords(text)
    normalize_text = get_established_concept(text, concept_mapping_file)
    if normalize_text != text:
        return normalize_text  # Return the mapped concept if found
    synonyms = get_synonyms(text)
    for synonym in synonyms:
        synonym_concept = get_established_concept(synonym, concept_mapping_file)
        if synonym_concept != synonym:
            return synonym_concept  # Return the first found concept mapping for a synonym
    return text  # Return the original text if no mapping found for it or its synonyms


def flatten_predicate(pred: str) -> str:
    """Normalize the predicate by converting it to lowercase and mapping it to a standard form if it matches certain patterns."""
    pred = pred.lower().strip()
    if pred in ['isa', 'typeof', 'is']: return 'is_a'
    if pred in ['partof', 'includedin', 'memberof']: return 'part_of'
    if pred in ['defines', 'supports', 'allows', 'has']: return 'defines'
    return pred


def get_similarity(text1: str, 
                   text2: str, 
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


def get_atomic_similarity(triple1: tuple[str, str, str], 
                          triple2: tuple[str, str, str],
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
    s1, p1, o1 = triple1
    s2, p2, o2 = triple2

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
