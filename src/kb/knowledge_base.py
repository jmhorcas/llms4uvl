import itertools
import csv
import numpy
import spacy
from sentence_transformers import SentenceTransformer

from kb import Triplet
from kb import utils


class KnowledgeBase:
    """Class representing a knowledge base, which is a collection of triplets (subject, predicate, object)."""

    CONCEPT_MAPPING_FILE = "../resources/concept_mapping.json"

    def __init__(self, nlp: spacy.Language, language_model: SentenceTransformer) -> None:
        """Initialize an empty knowledge base."""
        self.triplets: list[Triplet] =[]
        self.nlp = nlp
        self.language_model = language_model

    def add_triplet(self, triplet: Triplet) -> None:
        """Add a triplet to the knowledge base."""
        self.triplets.append(triplet)
    
    def join_kb(self, other_kb: 'KnowledgeBase') -> None:
        """Join another knowledge base with the current one by adding all its triplets to the current knowledge base."""
        self.triplets.extend(other_kb.triplets)
        
    def __len__(self):
        """Return the number of triplets in the knowledge base."""
        return len(self.triplets)

    def load_from_csv(self, file_path: str) -> None:
        """Load triplets from a CSV file that contains the columns named Subject, Predicate, and Object."""
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)    
            for row in reader:
                self.add_triplet(
                    triplet=Triplet(
                        subject=row['Subject'],
                        predicate=row['Predicate'],
                        object=row['Object']
                    )
                )
    
    def normalize(self) -> 'KnowledgeBase':
        """Normalize all triplets in the knowledge base and return a new KnowledgeBase instance."""
        normalized_kb = KnowledgeBase(self.nlp, self.language_model)
        for triplet in self.triplets:
            normalized_triplet = Triplet(
                subject=utils.normalize_text(triplet.subject, self.CONCEPT_MAPPING_FILE, self.nlp),
                predicate=triplet.predicate,  # Assuming predicates do not need normalization
                object=utils.normalize_text(triplet.object, self.CONCEPT_MAPPING_FILE, self.nlp)
            )
            normalized_kb.add_triplet(normalized_triplet)
        return normalized_kb
    
    def save_to_csv(self, file_path: str) -> None:
        """Save the triplets in the knowledge base to a CSV file."""
        with open(file_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Subject', 'Predicate', 'Object'])
            for triplet in self.triplets:
                writer.writerow([triplet.subject, triplet.predicate, triplet.object])

    # def deduplicate(self, threshold: float = 0.85) -> 'KnowledgeBase':
    #     """Remove those triplets that are equals or similar to other triplets."""
    #     kb = KnowledgeBase(self.nlp, self.language_model)
    #     for triplet1 in self.triplets:
    #         for triplet2 in self.triplets:
    #             pass
    #     pass

    def remove_exact_duplicates(self) -> 'KnowledgeBase':
        """Remove those triplets that are exactly the same as other triplets."""
        kb = KnowledgeBase(self.nlp, self.language_model)
        seen = set()
        for triplet in self.triplets:
            if triplet not in seen:
                kb.add_triplet(triplet)
                seen.add(triplet)
        return kb
    
    # def remove_semantic_duplicates(self, threshold=0.92) -> 'KnowledgeBase':
    #     """Remove those triplets that are semantically similar to other triplets based on a similarity threshold."""
    #     triples = self.triplets
    #     if not triples: 
    #         return KnowledgeBase()
        
    #     unique = {triples[0]}
        
    #     for i in range(1, len(triples)):
    #         is_duplicate = False
    #         for u in unique:
    #             # Usamos tu función de similitud (la del máximo entre global y atómica)
    #             score = utils.get_hybrid_similarity(triples[i].to_tuple(), u.to_tuple()) 
                
    #             if score >= threshold:
    #                 is_duplicate = True
    #                 break
            
    #         if not is_duplicate:
    #             unique.add(triples[i])
                
    #     kb = KnowledgeBase()
    #     for t in unique:
    #         kb.add_triplet(t)
    #     return kb
    
    def remove_semantic_duplicates(self, threshold: float = 0.92) -> 'KnowledgeBase':
        """Remove those triplets that are semantically similar to other triplets based on a similarity threshold."""
        unique_triples = utils.fast_semantic_deduplication(
            self.language_model,
            triples=[t.to_tuple() for t in self.triplets],
            threshold=threshold
        )
        kb = KnowledgeBase(self.nlp, self.language_model)
        for t in unique_triples:
            kb.add_triplet(Triplet(*t))
        return kb

    def get_leafs(self) -> list[Triplet]:
        """Return those triplets whose Object do not appear in any Subject."""
        pass

    def get_possible_seeds(self, n: int = 2) -> list[Triplet]:
        """Return those Subject that appear in more at least n triplets."""
        pass

    # def combine(self, other_kb: 'KnowledgeBase') -> None:
    #     self.triplets.extend(other_kb.triplets)

    def knowledge_consolidation(self, threshold: float = 0.92) -> 'KnowledgeBase':
        """Consolidate the knowledge base by normalizing and removing duplicates."""
        kb = self.normalize()
        kb = kb.consolidate_relations()
        kb = kb.consolidate_objects()
        kb = kb.consolidate_subjects()
        kb = kb.remove_exact_duplicates()
        kb = kb.remove_semantic_duplicates(threshold=threshold)
        return kb
    
    def consolidate_relations(self, alpha: float = 1.4, threshold_max: float = 0.95, threshold_min: float = 0.75) -> 'KnowledgeBase':
        """Apply Relation Clustering to unify semantically similar predicates."""
        # Extract all predicates
        all_predicates = [t.predicate for t in self.triplets]
        # Obtain the unification map
        mapping = utils.relation_clustering(self.language_model, all_predicates, alpha=alpha, H=threshold_max, L=threshold_min)
        # Create a new KB with the updated predicates
        new_kb = KnowledgeBase(self.nlp, self.language_model)
        for t in self.triplets:
            unified_predicate = mapping.get(t.predicate, t.predicate)
            new_kb.add_triplet(Triplet(t.subject, unified_predicate, t.object))
        return new_kb
    
    def consolidate_objects(self, alpha: float = 1.4, threshold_max: float = 0.95, threshold_min: float = 0.75) -> 'KnowledgeBase':
        """Apply Relation Clustering to unify semantically similar predicates."""
        # Extract all objects
        all_objects = [t.object for t in self.triplets]
        # Obtain the unification map
        mapping = utils.relation_clustering(self.language_model, all_objects, alpha=alpha, H=threshold_max, L=threshold_min)
        # Create a new KB with the updated objects
        new_kb = KnowledgeBase(self.nlp, self.language_model)
        for t in self.triplets:
            unified_object = mapping.get(t.object, t.object)
            new_kb.add_triplet(Triplet(t.subject, t.predicate, unified_object))
        return new_kb
    
    def consolidate_subjects(self, alpha: float = 1.4, threshold_max: float = 0.95, threshold_min: float = 0.75) -> 'KnowledgeBase':
        """Apply Relation Clustering to unify semantically similar predicates."""
        # Extract all subjects
        all_subjects = [t.subject for t in self.triplets]
        # Obtain the unification map
        mapping = utils.relation_clustering(self.language_model, all_subjects, alpha=alpha, H=threshold_max, L=threshold_min)
        # Create a new KB with the updated subjects
        new_kb = KnowledgeBase(self.nlp, self.language_model)
        for t in self.triplets:
            unified_subject = mapping.get(t.subject, t.subject)
            new_kb.add_triplet(Triplet(unified_subject, t.predicate, t.object))
        return new_kb
    
    def consistency(self, csv_file: str) -> dict:
        """Calculate the consistency of the knowledge base by comparing it with other runs of the same seed."""
        results = load_and_evaluate_consistency(csv_file, self.nlp, self.language_model)
        calculate_global_consistency(results, {})  # Aquí deberías pasar también los Kbs por semilla para el cálculo global

def calculate_global_consistency(results_by_seed: dict, kbs_by_seed: dict) -> dict:
    """
    Calcula las consistencias globales Macro y Micro.
    
    Args:
        results_by_seed: Diccionario {seed: score_consistencia}
        kbs_by_seed: Diccionario {seed: {run: KnowledgeBase}}
    """
    
    # 1. MACRO CONSISTENCY (Promedio simple de los scores)
    macro_consistency = sum(results_by_seed.values()) / len(results_by_seed)
    
    # 2. MICRO CONSISTENCY (Ponderada por el número de tripletas)
    # La idea es: (Score_semilla1 * Total_tripletas_semilla1 + ...) / Total_tripletas_global
    total_triplets_all_seeds = 0
    weighted_sum = 0
    
    for seed, runs_dict in kbs_by_seed.items():
        # Calculamos el promedio de tripletas que genera esta semilla en sus 5 runs
        avg_triplets_in_seed = sum(len(kb) for kb in runs_dict.values()) / len(runs_dict)
        
        weighted_sum += results_by_seed[seed] * avg_triplets_in_seed
        total_triplets_all_seeds += avg_triplets_in_seed
        
    micro_consistency = weighted_sum / total_triplets_all_seeds if total_triplets_all_seeds > 0 else 0
    
    return {
        "Macro Consistency": macro_consistency,
        "Micro Consistency": micro_consistency,
        "Total Avg Triplets": total_triplets_all_seeds
    }
    
def load_and_evaluate_consistency(csv_file: str, nlp: spacy.Language, model: SentenceTransformer) -> dict[str, float]:
    # Diccionario para agrupar: seeds -> runs -> KnowledgeBase
    data_map = {}
    
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = row['Seed']
            run = row['Run']
            if seed not in data_map: data_map[seed] = {}
            if run not in data_map[seed]: data_map[seed][run] = KnowledgeBase(nlp, model)
            
            data_map[seed][run].add_triplet(
                Triplet(row['Subject'], row['Predicate'], row['Object'])
            )

    # Calcular consistencia por semilla
    results = {}
    for seed, runs_dict in data_map.items():
        # Pasamos a lista las KBs de las 5 ejecuciones
        list_of_kbs = list(runs_dict.values())
        # Opcional: Consolidar (normalizar + deduplicar exactos) antes de comparar
        consolidated_kbs = [kb.remove_exact_duplicates() for kb in list_of_kbs]
        
        consistency_score = calculate_consistency(consolidated_kbs)
        results[seed] = consistency_score
        print(f"Semilla: {seed} | Consistencia Factual: {consistency_score:.4f}")

    global_results = calculate_global_consistency(results, data_map)  # Aquí pasamos también los Kbs por semilla para el cálculo global
    print(f"Consistencia Global Macro: {global_results['Macro Consistency']:.4f}")
    print(f"Consistencia Global Micro: {global_results['Micro Consistency']:.4f}")
    print(f"Total Promedio de Tripletas por Semilla: {global_results['Total Avg Triplets']:.2f}")
    
    return results


def calculate_consistency(kbs: list['KnowledgeBase'], threshold: float = 0.85) -> float:
    """
    Calcula la consistencia promedio entre un conjunto de KnowledgeBases (runs).
    Utiliza el comparador semántico para encontrar coincidencias.
    
    Args:
        kbs: Lista de objetos KnowledgeBase (uno por cada run).
        threshold: Umbral de similitud de coseno para considerar que dos tripletas son iguales.
    """
    if len(kbs) < 2:
        return 1.0

    similarities = []

    # Comparar todos los pares posibles de ejecuciones (10 combinaciones para 5 runs)
    for kb_i, kb_j in itertools.combinations(kbs, 2):
        # Convertir tripletas a tuplas para el comparador
        triples_i = [t.to_tuple() for t in kb_i.triplets]
        triples_j = [t.to_tuple() for t in kb_j.triplets]

        if not triples_i or not triples_j:
            similarities.append(0.0)
            continue

        # Encontrar la intersección semántica usando tu utilidad de comparación
        # Nota: Usamos la matriz de similitud de sentence_transformers para eficiencia
        # Si 'utils.fast_semantic_intersection' no existe, aquí simulamos la lógica:
        
        matches = 0
        # Calculamos similitudes entre todos los elementos de i y j
        # Para artículos de 14 páginas, se recomienda vectorizar primero:
        embeddings_i = kb_i.language_model.encode([" ".join(t) for t in triples_i])
        embeddings_j = kb_j.language_model.encode([" ".join(t) for t in triples_j])
        
        # Matriz de similitud de coseno
        cosine_matrix = numpy.dot(embeddings_i, embeddings_j.T) / (
            numpy.outer(numpy.linalg.norm(embeddings_i, axis=1), numpy.linalg.norm(embeddings_j, axis=1))
        )

        # Contamos cuántas tripletas de i tienen al menos una pareja similar en j
        matches = numpy.sum(numpy.max(cosine_matrix, axis=1) >= threshold)
        
        # Jaccard Semántico: Intersección / Unión
        # Intersección aprox = matches
        # Unión aprox = len(i) + len(j) - matches
        union = len(triples_i) + len(triples_j) - matches
        jaccard_semantic = matches / union if union > 0 else 0
        similarities.append(jaccard_semantic)

    # Devolvemos la media de todas las comparaciones
    return sum(similarities) / len(similarities)