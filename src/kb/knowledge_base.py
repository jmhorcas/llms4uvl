import itertools
import csv
from typing import Any

from sentence_transformers import util

from kb import Triplet
from kb import NaturalLanguageProcessor


class KnowledgeBase:
    """Class representing a knowledge base, which is a collection of triplets (subject, predicate, object)."""

    CONCEPT_MAPPING_FILE = "../resources/concept_mapping.json"

    def __init__(self, nlp: NaturalLanguageProcessor) -> None:
        """Initialize an empty knowledge base."""
        self.triplets: list[Triplet] = []
        self.nlp = nlp
        self.iterations_seeds: dict[str, dict[int, 'KnowledgeBase']] = {}
        
    def load_from_csv(self, file_path: str) -> None:
        """Load triplets from a CSV file that contains the columns named Subject, Predicate, and Object.
        
        Optionally, the CSV can incorporate a header with the number of Iteration, the Run, and the Seed.
        """
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'Seed' in reader.fieldnames and 'Run' in reader.fieldnames:
                for row in reader:
                    triplet = process_triplet(row)
                    seed = row['Seed']
                    run = int(row['Run'])
                    self.add_triplet(triplet)
                    if seed not in self.iterations_seeds:
                        self.iterations_seeds[seed] = {}
                    if run not in self.iterations_seeds[seed]:
                        self.iterations_seeds[seed][run] = KnowledgeBase(self.nlp)
                    
                    self.iterations_seeds[seed][run].add_triplet(triplet)
            else:
                for row in reader:
                    triplet = process_triplet(row)
                    self.add_triplet(triplet)            

    def add_triplet(self, triplet: Triplet) -> None:
        """Add a triplet to the knowledge base."""
        self.triplets.append(triplet)
    
    def join_kb(self, other_kb: 'KnowledgeBase') -> None:
        """Join another knowledge base with the current one by adding all its triplets to the current knowledge base."""
        self.triplets.extend(other_kb.triplets)
        
    def __len__(self):
        """Return the number of triplets in the knowledge base."""
        return len(self.triplets)
    
    def normalize(self) -> 'KnowledgeBase':
        """Normalize all triplets in the knowledge base and return a new KnowledgeBase instance."""
        normalized_kb = KnowledgeBase(self.nlp)
        for triplet in self.triplets:
            normalized_triplet = Triplet(
                                    subject=self.nlp.normalize_text(triplet.subject),
                                    #predicate=self.nlp.remove_stopwords(self.nlp.case_folding(triplet.predicate)),
                                    predicate=self.nlp.normalize_text(triplet.predicate),
                                    object=self.nlp.normalize_text(triplet.object)
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

    def remove_exact_duplicates(self) -> 'KnowledgeBase':
        """Remove those triplets that are exactly the same as other triplets."""
        deduplicated_kb = KnowledgeBase(self.nlp)
        seen = set()
        for triplet in self.triplets:
            if triplet not in seen:
                deduplicated_kb.add_triplet(triplet)
                seen.add(triplet)
        return deduplicated_kb

    def deduplicate(self, threshold: float = 0.92) -> 'KnowledgeBase':
        """Remove those triplets that are semantically similar to other triplets based on a similarity threshold."""
        deduplicated_kb = KnowledgeBase(self.nlp)
        deduplicated_triplets = self.nlp.deduplicate_triplets(self.triplets, threshold=threshold)
        deduplicated_kb.triplets = list(deduplicated_triplets)
        return deduplicated_kb

    def clustering(self) -> 'KnowledgeBase':
        """Merges each component of the triplet into a more frequen component."""
        unique_subs = list({t.subject for t in self.triplets})
        unique_preds = list({t.predicate for t in self.triplets})
        unique_objs = list({t.object for t in self.triplets})

        s_map = self.nlp.relation_clustering(unique_subs)
        p_map = self.nlp.relation_clustering(unique_preds)
        o_map = self.nlp.relation_clustering(unique_objs)

        new_triplets_set = set()
        for t in self.triplets:
            new_t = Triplet(
                subject=s_map.get(t.subject, t.subject),
                predicate=p_map.get(t.predicate, t.predicate),
                object=o_map.get(t.object, t.object)
            )
            new_triplets_set.add(new_t)

        new_kb = KnowledgeBase(self.nlp)
        new_kb.triplets = list(new_triplets_set)
        return new_kb
    
    def consolidate(self, threshold: float = 0.92) -> 'KnowledgeBase':
        kb = self.normalize()
        kb = kb.deduplicate(threshold=threshold)
        kb = kb.clustering()
        return kb

    def calculate_consistency(self) -> dict[str, Any]:
        """Calcula la consistencia factual (Macro y Micro) de la KB."""
        if not self.iterations_seeds:
            print("No hay datos de Seed/Run para calcular consistencia.")
            return {}

        seed_scores = {}
        for seed, runs_dict in self.iterations_seeds.items():
            # Pasamos la lista de KBs de esa semilla (Run 1, Run 2...)
            # IMPORTANTE: Aquí se pasan SIN normalizar
            kbs_of_seed = list(runs_dict.values())
            
            # Calculamos consistencia para esta semilla
            score = self._calculate_pair_consistency(kbs_of_seed)
            seed_scores[seed] = score

        # Cálculo Global
        global_metrics = self._calculate_global_metrics(seed_scores)
        global_metrics['Seed_Scores'] = seed_scores  # Agregamos los scores por semilla para transparencia
        return global_metrics

    def _calculate_pair_consistency(self, kbs: list['KnowledgeBase'], threshold: float = 0.90) -> float:
        """Calcula el Jaccard Semántico promedio entre pares de runs."""
        if len(kbs) < 2: return 1.0
        
        scores = []
        for kb_i, kb_j in itertools.combinations(kbs, 2):
            # Usamos el método de deduplicación batch que optimizamos antes 
            # para comparar i contra j de forma eficiente
            texts_i = [t.to_sentence() for t in kb_i.triplets]
            texts_j = [t.to_sentence() for t in kb_j.triplets]
            
            if not texts_i or not texts_j:
                scores.append(0.0); continue

            # Vectorización batch
            embs_i = self.nlp.language_model.encode(texts_i, convert_to_tensor=True)
            embs_j = self.nlp.language_model.encode(texts_j, convert_to_tensor=True)
            
            # Matriz de similitud
            sim_matrix = util.cos_sim(embs_i, embs_j)
            
            # Intersección Semántica: tripletas de i que tienen match en j
            matches_i = (sim_matrix.max(dim=1).values >= threshold).sum().item()
            # Para Jaccard simétrico, calculamos también matches de j en i
            matches_j = (sim_matrix.max(dim=0).values >= threshold).sum().item()
            
            avg_matches = (matches_i + matches_j) / 2
            union = len(texts_i) + len(texts_j) - avg_matches
            scores.append(avg_matches / union if union > 0 else 0)
            
        return sum(scores) / len(scores)

    def _calculate_global_metrics(self, seed_scores: dict[str, float]) -> dict[str, float]:
        """Calcula Macro (media simple) y Micro (ponderada por volumen)."""
        macro = sum(seed_scores.values()) / len(seed_scores)
        
        total_weight = 0
        weighted_sum = 0
        
        for seed, score in seed_scores.items():
            # Peso = promedio de tripletas en los runs de esta semilla
            runs = self.iterations_seeds[seed].values()
            avg_size = sum(len(kb) for kb in runs) / len(runs)
            
            weighted_sum += score * avg_size
            total_weight += avg_size
            
        return {
            "Macro": macro,
            "Micro": weighted_sum / total_weight if total_weight > 0 else 0,
            "Avg_Triplets": total_weight / len(seed_scores)
        }

    def create_global_union_by_seed(self, source_dicts_list: list['KnowledgeBase']) -> dict[str, 'KnowledgeBase']:
        """
        source_dicts_list: [dict_paper, dict_web, dict_all]
        Cada dict tiene la estructura: { 'Seed Name': { 1: KB, 2: KB, ... } }
        Recibe los KB en raw.
        """
        global_union = {}

        # 1. Identificamos todas las semillas que aparecen en CUALQUIERA de los dicts
        # Esto evita que si una semilla solo está en 'Web', la ignoremos.
        all_seeds = set()
        for d in source_dicts_list:
            all_seeds.update(d.iterations_seeds.keys())

        for seed in all_seeds:
            # Creamos una KB vacía que servirá de "bolsa" para esta semilla
            seed_accumulator = KnowledgeBase(self.nlp)
            
            # 2. Recorremos cada fuente (Paper, Web, All...)
            for source_dict in source_dicts_list:
                if seed in source_dict.iterations_seeds:
                    # 3. Recorremos cada una de las 5 ejecuciones (Runs)
                    for run_id, kb in source_dict.iterations_seeds[seed].items():
                        # IMPORTANTE: join_kb añade las tripletas, NO reemplaza la KB
                        seed_accumulator.join_kb(kb)
            # 4. Consolidación: Aquí es donde la magia ocurre
            # Pasamos de 500 tripletas repetidas a las 50-100 "verdades" únicas
            union_kb = seed_accumulator.consolidate()
            
            # Guardamos el resultado final consolidado para esta semilla
            global_union[seed] = union_kb

        return global_union

def process_triplet(row: dict[str, Any]) -> Triplet:
    return Triplet(subject=row['Subject'], predicate=row['Predicate'], object=row['Object'])