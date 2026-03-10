import numpy
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from kb import KnowledgeBase


class TaxonomyConstructor:
    """Class responsible for constructing a taxonomy (hierarchy) from a knowledge base using the algorithm described in the article [Hu2025_ExtensiveMaterialization](https://doi.org/10.18653/v1/2025.acl-long.789)."""

    def __init__(self, language_model: SentenceTransformer, beta: float = 0.5) -> None:
        self.language_model = language_model
        self.beta = beta
        self.taxonomy: dict[str, list[str]] = {}       # H (The final hierarchy)
        self.V: set[str] = set()           # Set of classes already visited/processed
        self.class_embeddings: dict[str, numpy.ndarray] = {}
        self.logical_operators = {'==', '<=', '>=', '!=', '>', '<', '=>', '<=>' '&', '|', '!', '+', '-', '/', '*'}

    def _get_custom_similarity(self, c: str, p: str) -> float:
        """Calculate similarity: Strict identity for logical operators, Cosine for the rest."""
        # If either is a logical operator, similarity is 1 only if they are exactly equal
        if c in self.logical_operators or p in self.logical_operators:
            return 1.0 if c == p else 0.0
        
        # For normal text, we use pre-calculated embeddings
        c_emb = self.class_embeddings[c].reshape(1, -1)
        p_emb = self.class_embeddings[p].reshape(1, -1)
        return float(cosine_similarity(c_emb, p_emb)[0][0])
    
    def construct_taxonomy(self, kb: KnowledgeBase) -> dict[str, list[str]]:
        """Taxonomy construction."""
        # Extract all unique classes from KB (objects and subjects)
        all_classes = list(set([t.object for t in kb.triplets] + [t.subject for t in kb.triplets]))
        
        # Pre-calculate embeddings for the Score calculation
        self.class_embeddings = {c: self.language_model.encode(c) for c in all_classes}

        for c in all_classes:
            if c not in self.V:
                self.insert_class_recursive(c, kb)
        return self.taxonomy

    def insert_class_recursive(self, c: str, kb: KnowledgeBase) -> None:
        """Insert_class_recursive(c, kb)."""
        # 1. V <- V U {c}
        self.V.add(c)

        # 2. p* <- find p in C that maximizes Score(c, p)
        # Search for candidates to be parent: triplets where 'c' is the subject
        candidates = [t.object for t in kb.triplets if t.subject == c and t.object != c]
        
        if not candidates:
            return # No parent found, it's a root

        # Calculate Score according to the article's formula:
        # Score(c, p) = frequency(c, p) + beta * similarity(c, p)
        counts = Counter(candidates)
        best_p = None
        max_score = -1

        c_emb = self.class_embeddings[c].reshape(1, -1)

        for p, freq in counts.items():
            p_emb = self.class_embeddings[p].reshape(1, -1)
            # Cosine similarity (semantic)
            sim = self._get_custom_similarity(c, p)
            score = freq + (self.beta * sim)
            if score > max_score:
                max_score = score
                best_p = p

        # 3. if p* not in V then
        if best_p and best_p not in self.V:
            # 4. Insert_class_recursive(p*, kb)
            self.insert_class_recursive(best_p, kb)

        # 5. H <- H U { (c, subclass_of, p*) }
        if best_p:
            if best_p not in self.taxonomy:
                self.taxonomy[best_p] = []
            self.taxonomy[best_p].append(c)