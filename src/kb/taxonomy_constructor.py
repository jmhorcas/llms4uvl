import torch
from collections import Counter
from sentence_transformers import util

from kb import KnowledgeBase, NaturalLanguageProcessor


class TaxonomyConstructor:
    """Build a taxonomy from a knowledge base.

    inverted=False: Search upwards (from Object to Subject).
    inverted=True:  Search downwards (from Subject to Object).
    """

    def __init__(self, nlp: NaturalLanguageProcessor, beta: float = 0.5, inverted: bool = False) -> None:
        self.nlp = nlp
        self.beta = beta
        self.inverted = inverted
        self.taxonomy: dict[str, list] = {}
        self.V: set[str] = set()
        self.class_embeddings: dict[str, torch.Tensor] = {}

    def _get_custom_similarity(self, c: str, p: str) -> float:
        """Custom similarity function based on util.cos_sim with operator protection."""
        if c in self.nlp.special_cases or p in self.nlp.special_cases:
            return 1.0 if c == p else 0.0
        
        c_emb = self.class_embeddings[c]
        p_emb = self.class_embeddings[p]
        return float(util.cos_sim(c_emb, p_emb).item())
    
    def construct_taxonomy(self, kb: KnowledgeBase) -> dict[str, list]:
        """Construct the taxonomy graph from the knowledge base."""
        # Extraer todas las clases
        all_classes = list(set([t.object for t in kb.triplets] + [t.subject for t in kb.triplets]))
        
        # Pre-calcular embeddings
        self.class_embeddings = {
            c: self.nlp.language_model.encode(c, convert_to_tensor=True) 
            for c in all_classes
        }

        self.V = set()
        self.taxonomy = {}

        # Determinar orden de procesamiento
        if self.inverted:
            # Invertida: Procesar primero los que más actúan como Sujetos (Potenciales Raíces)
            counts = Counter([t.subject for t in kb.triplets])
            sorted_classes = sorted(all_classes, key=lambda x: counts[x], reverse=True)
        else:
            # Normal: Orden estándar
            sorted_classes = all_classes

        for c in sorted_classes:
            if c not in self.V:
                self.insert_class_recursive(c, kb)
                
        return self.taxonomy

    def insert_class_recursive(self, current: str, kb: KnowledgeBase) -> None:
        if current in self.V: return
        self.V.add(current)

        if self.inverted:
            # --- LÓGICA INVERTIDA (Padre -> Predicado -> Hijo) ---
            candidates = [t for t in kb.triplets if t.subject == current and t.object != current]
            for t in candidates:
                child = t.object
                sim = self._get_custom_similarity(current, child)
                score = 1 + (self.beta * sim)
                
                if score > 0.5:
                    if current not in self.taxonomy: self.taxonomy[current] = []
                    relation = (child, t.predicate) # Predicado original
                    if relation not in self.taxonomy[current]:
                        self.taxonomy[current].append(relation)
                    self.insert_class_recursive(child, kb)
        else:
            # --- LÓGICA NORMAL (Objeto -> Sujeto) ---
            # Buscamos tripletas donde 'current' es el OBJETO para hallar su PADRE (Sujeto)
            relevant_triplets = [t for t in kb.triplets if t.object == current and t.subject != current]
            if not relevant_triplets: return
            
            # Agrupamos sujetos para encontrar el mejor padre por frecuencia/similitud
            parents_list = [t.subject for t in relevant_triplets]
            counts = Counter(parents_list)
            best_p, max_score = None, -1

            for p, freq in counts.items():
                sim = self._get_custom_similarity(current, p)
                score = freq + (self.beta * sim)
                if score > max_score:
                    max_score = score
                    best_p = p

            if best_p:
                # 1. Recuperar el predicado original que une a este hijo con el mejor padre
                # Buscamos en las tripletas relevantes la que conecta current con best_p
                orig_t = next((t for t in relevant_triplets if t.subject == best_p), None)
                predicate = orig_t.predicate if orig_t else "related_to"

                # 2. Recursión hacia el padre
                if best_p not in self.V:
                    self.insert_class_recursive(best_p, kb)
                
                # 3. Guardar en la taxonomía (Padre -> [Hijo, Predicado])
                if best_p not in self.taxonomy:
                    self.taxonomy[best_p] = []
                
                relation = (current, predicate) # <--- AQUÍ mantenemos el original
                if relation not in self.taxonomy[best_p]:
                    self.taxonomy[best_p].append(relation)

    def get_clean_tree(self):
        """Maintain the tree cleaning logic."""
        clean_tree = {}
        seen_children = set()
        for parent, children in self.taxonomy.items():
            if parent not in clean_tree:
                clean_tree[parent] = []
            for item in children:
                # If inverted, item is (child, predicate), otherwise it's a string child
                child = item[0] if isinstance(item, tuple) else item
                if child not in seen_children and child != parent:
                    clean_tree[parent].append(item)
                    seen_children.add(child)
        return clean_tree
    
    def print_taxonomy_iterative_with_predicates(self):
        """Print the taxonomy in a readable format, showing predicates when available."""
        if not self.taxonomy:
            print("Empty taxonomy.")
            return

        # 1. Identificar raíces
        all_children = set(rel[0] for children in self.taxonomy.values() for rel in children)
        roots = [n for n in self.taxonomy.keys() if n not in all_children]
        
        if not roots:
            roots = list(self.taxonomy.keys())[:1]
            print("(!) Cycle detected. Starting from an arbitrary node.")

        stack = []
        for i, root in enumerate(reversed(roots)):
            stack.append((root, "", i == 0, None))

        visited = set()

        while stack:
            node, indent, is_last, pred = stack.pop()
            
            # El identificador de visita incluye el predicado para permitir 
            # que un nodo aparezca bajo distintos predicados si fuera necesario
            node_id = f"{node}_{pred}" 

            if node in visited:
                marker = "└── " if is_last else "├── "
                label = f" [{pred}]──>" if pred else ""
                print(f"{indent}{marker}{label}{node} (RECURSION)")
                continue
            
            visited.add(node)

            marker = "└── " if is_last else "├── "
            if pred:
                print(f"{indent}{marker}[{pred}]──> {node}")
            else:
                print(f"{indent}{marker}{node}")

            children_rels = self.taxonomy.get(node, [])
            new_indent = indent + ("    " if is_last else "│   ")
            
            for i, (child, p_name) in enumerate(reversed(children_rels)):
                stack.append((child, new_indent, i == 0, p_name))