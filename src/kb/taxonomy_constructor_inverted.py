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
        """Aquí 'p' es el Padre y buscamos a sus hijos."""
        if p in self.V: return
        self.V.add(p)

        # CAMBIO 2: Buscar candidatos a HIJOS (donde p es el sujeto)
        # Buscamos los objetos que están bajo este sujeto
        candidates = [t.object for t in kb.triplets if t.subject == p and t.object != p]
        
        if not candidates:
            return 

        # Seleccionar los mejores hijos (pueden ser varios en un árbol)
        # Nota: A diferencia del artículo original que busca UN padre, 
        # un padre puede tener MÚLTIPLES hijos legítimos.
        counts = Counter(candidates)
        
        for child, freq in counts.items():
            # Calculamos si el hijo es lo suficientemente similar/frecuente
            sim = self._get_custom_similarity(p, child)
            score = freq + (self.beta * sim)
            
            # Si el score es alto, lo aceptamos como hijo legítimo
            if score > 0.5: # Umbral de aceptación
                if p not in self.taxonomy:
                    self.taxonomy[p] = []
                if child not in self.taxonomy[p]:
                    self.taxonomy[p].append(child)
                
                # Recursión hacia abajo: procesar al hijo
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