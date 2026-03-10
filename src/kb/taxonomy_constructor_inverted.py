import numpy
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from kb import KnowledgeBase


class TaxonomyConstructorInverted:
    def __init__(self, language_model: SentenceTransformer, beta: float = 0.5) -> None:
        self.language_model = language_model
        self.beta = beta
        self.taxonomy: dict[str, list[str]] = {} 
        self.V: set[str] = set() 
        self.class_embeddings: dict[str, numpy.ndarray] = {}
        self.logical_operators = {'==', '<=', '>=', '!=', '>', '<', '=>', '<=>', '&', '|', '!', '+', '-', '/', '*'}

    def _get_custom_similarity(self, c: str, p: str) -> float:
        if c in self.logical_operators or p in self.logical_operators:
            return 1.0 if c == p else 0.0
        c_emb = self.class_embeddings[c].reshape(1, -1)
        p_emb = self.class_embeddings[p].reshape(1, -1)
        return float(cosine_similarity(c_emb, p_emb)[0][0])
    
    def construct_taxonomy(self, kb: KnowledgeBase) -> dict[str, list[str]]:
        all_classes = list(set([t.object for t in kb.triplets] + [t.subject for t in kb.triplets]))
        self.class_embeddings = {c: self.language_model.encode(c) for c in all_classes}

        # CAMBIO 1: Identificar raíces por frecuencia de SUJETO
        # En esta jerarquía invertida, el Padre es el Sujeto.
        sub_counts = Counter([t.subject for t in kb.triplets])
        
        # Procesamos primero los que más aparecen como Sujeto (más "Padres")
        sorted_classes = sorted(all_classes, key=lambda x: sub_counts[x], reverse=True)

        self.V = set()
        self.taxonomy = {}

        for c in sorted_classes:
            if c not in self.V:
                self.insert_class_recursive(c, kb)
        
        return self.taxonomy

    def insert_class_recursive(self, p: str, kb: KnowledgeBase) -> None:
        """Versión Invertida: p es Padre, buscamos sus hijos y el predicado que los une."""
        if p in self.V: return
        self.V.add(p)

        # Buscamos tripletas donde p es el sujeto (Padre -> Predicado -> Hijo)
        candidates = [t for t in kb.triplets if t.subject == p and t.object != p]
        
        if not candidates:
            return 

        for t in candidates:
            child = t.object
            predicate = t.predicate
            
            # Calculamos score para validar si la relación es fuerte
            sim = self._get_custom_similarity(p, child)
            score = 1 + (self.beta * sim) # Usamos frecuencia 1 por tripleta individual
            
            if score > 0.5:
                if p not in self.taxonomy:
                    self.taxonomy[p] = []
                
                # Guardamos (Hijo, Predicado) para la impresión
                relacion = (child, predicate)
                if relacion not in self.taxonomy[p]:
                    self.taxonomy[p].append(relacion)
                
                if child not in self.V:
                    self.insert_class_recursive(child, kb)

    def get_clean_tree(self):
        """Convierte el grafo de la taxonomía en un árbol estricto (1 padre por hijo)"""
        clean_tree = {}
        seen_children = set()
        
        # Ordenar por importancia (opcional)
        for parent, children in self.taxonomy.items():
            if parent not in clean_tree:
                clean_tree[parent] = []
            for child in children:
                if child not in seen_children and child != parent:
                    clean_tree[parent].append(child)
                    seen_children.add(child)
        return clean_tree