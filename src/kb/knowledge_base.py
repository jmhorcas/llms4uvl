import csv
from collections import Counter
from dataclasses import dataclass

from thefuzz import process, fuzz
import spacy
nlp = spacy.load("en_core_web_sm")  # O "es_core_news_sm" si es en español


# 'Noise' words that we do not want as technical seeds
UVL_NON_TECHNICAL_WORDS = {'software product line', 
                           'domain specific language', 
                           'variability modeling language', 
                           'automate analysis',
                           'tool interoperability',
                           'research community',
                           'variability analysis tool',
                           'product configuration',
                           'variation point',
                           'universal variability language'}


@dataclass(frozen=True)  # frozen=True permite que sea "hashable" para deduplicar fácilmente
class Triplet:
    subject: str
    predicate: str
    obj: str

    def __str__(self):
        return f"{self.subject} | {self.predicate} | {self.obj}"


@dataclass
class ExtractionEntry:
    seed: str
    iteration: int
    run: int
    triplet: Triplet


class KnowledgeBase:

    def __init__(self) -> None:
        self.entries: list[ExtractionEntry] = []

    def add_entry(self, seed: str, iteration: int, run: int, s: str, p: str, o: str) -> None:
        s = _normalize(s)
        s = _lemmatize(s)
        p = _normalize(p)
        p = _lemmatize(p)
        o = _normalize(o)
        o = _lemmatize(o)
        triplet = Triplet(s, p, o)
        self.entries.append(ExtractionEntry(seed, iteration, run, triplet))
    
    def load_from_csv(self, file_path: str) -> None:
        """Load triplets from a CSV file with columns: Iteration, Run, Seed, Subject, Predicate, Object."""
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)    
            for row in reader:
                self.add_entry(
                    iteration=int(row['Iteration']),
                    run=int(row['Run']),
                    seed=row['Seed'],
                    s=row['Subject'],
                    p=row['Predicate'],
                    o=row['Object']
                )

    def get_all_triplets_for_seed(self, seed: str) -> list[Triplet]:
        """Return a list of all triplets extracted for the given seed (including duplicates across runs)."""
        return [e.triplet for e in self.entries if e.seed == seed]
   
    def get_deduplicated_triplets(self, seed: str = None) -> set[Triplet]:
        """Returns a set of unique Triplet objects. If a seed is provided, filters only by that seed."""
        if seed:
            subset = {e.triplet for e in self.entries if e.seed == seed}
        else:
            subset = {e.triplet for e in self.entries}
        return subset

    def calculate_strict_consistency(self, seed: str) -> tuple[set[Triplet], set[Triplet], float]:
        """Returns (consistent_triplets, inconsistent_triplets, consistency_index) for the given seed."""
        # Filter entries for this seed
        seed_entries = [e for e in self.entries if e.seed == seed]
        total_runs = max([e.run for e in seed_entries]) if seed_entries else 0
        
        if total_runs == 0: 
            return set(), set(), 0
        
        # Count in how many runs each triplet appears
        triplet_run_map = {}
        for e in seed_entries:
            if e.triplet not in triplet_run_map:
                triplet_run_map[e.triplet] = set()
            triplet_run_map[e.triplet].add(e.run)

        consistent_triplets = {t for t, runs in triplet_run_map.items() if len(runs) == total_runs}
        inconsistent_triplets = {t for t, runs in triplet_run_map.items() if len(runs) == 1}
        consistent_count = len(consistent_triplets)
        total_unique = len(triplet_run_map)
        
        return (consistent_triplets, inconsistent_triplets, (consistent_count / total_unique) if total_unique > 0 else 0)

    def calculate_semantic_consistency(self, seed: str) -> tuple[set[Triplet], set[Triplet], float]:
        """Returns (consistent_triplets, inconsistent_triplets, consistency_index) for the given seed.
        This method considers two triplets consistent if they have the same subject and object, regardless of the predicate.
        """
        # Filter entries for this seed
        seed_entries = [e for e in self.entries if e.seed == seed]
        total_runs = max([e.run for e in seed_entries]) if seed_entries else 0

        if total_runs == 0: 
            return set(), set(), 0
        
        # Count in how many runs each triplet appears
        triplet_run_map = {}
        for e in seed_entries:
            fact_id = (e.triplet.subject, e.triplet.obj)  # Ignore the predicate for consistency
            if fact_id not in triplet_run_map:
                triplet_run_map[fact_id] = set()
            triplet_run_map[fact_id].add(e.run)

        consistent_triplets = {t for t, runs in triplet_run_map.items() if len(runs) == total_runs}
        inconsistent_triplets = {t for t, runs in triplet_run_map.items() if len(runs) == 1}
        consistent_count = len(consistent_triplets)
        total_unique = len(triplet_run_map)
        
        return (consistent_triplets, inconsistent_triplets, (consistent_count / total_unique) if total_unique > 0 else 0)

    def get_candidates(self, top_n: int = 5, min_frequency: int = 2, seed: str = None) -> list[tuple]:
        """Analyzes all 'Objects' from the current triplets and suggests which should be the next search seeds."""
        all_unique_triplets = self.get_deduplicated_triplets(seed)
        existing_seeds = set()
        if seed is None:
            existing_seeds = {e.seed.lower() for e in self.entries}
        
        objects_as_candidates = [
            t.obj for t in all_unique_triplets 
            if t.obj.lower() not in existing_seeds 
            and t.obj.lower() not in UVL_NON_TECHNICAL_WORDS
            # and len(t.obj) > 3  # Avoid terms that are too short
        ]
        
        # Returns the top N most common objects that exceed a minimum frequency
        counter = Counter(objects_as_candidates)
        return [(concept, count) for concept, count in counter.most_common(top_n) if count >= min_frequency]
    
    def get_fuzzy_candidates(self, 
                             top_n: int = 5,
                             seed: str = None,
                             similarity_threshold: int = 85) -> list[tuple]:
        """Groups similar objects and suggests seeds based on unified frequency.
        Threshold: 0-100 (85-90 is ideal for plural/hyphen variations).
        """
        all_unique_triplets = self.get_deduplicated_triplets()
        raw_objects = [t.obj for t in all_unique_triplets]
        
        # Filter existing seeds and stop_words
        existing_seeds = set()
        if seed is None:
            existing_seeds = {e.seed.lower() for e in self.entries}
        
        filtered_objects = [
            obj for obj in raw_objects 
            if obj.lower() not in existing_seeds and obj.lower() not in UVL_NON_TECHNICAL_WORDS
        ]

        if not filtered_objects:
            return []

        # Fuzzy Grouping Logic
        counts = Counter(filtered_objects)
        unique_concepts = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        
        fuzzy_counts = {}
        processed = set()

        for concept in unique_concepts:
            if concept in processed:
                continue
            
            # Search for very similar terms in the list
            # limit=10 to avoid processing the entire list each time
            matches = process.extract(concept, unique_concepts, scorer=fuzz.token_set_ratio, limit=10)
            
            # The current concept is the "representative" of the group
            fuzzy_counts[concept] = 0
            for match, score in matches:
                if score >= similarity_threshold and match not in processed:
                    fuzzy_counts[concept] += counts[match]
                    processed.add(match)

        # Return the top N unified concepts
        sorted_candidates = sorted(fuzzy_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_n]
    

def _normalize(text: str) -> str:
    """Normalize text by stripping, converting to lowercase, and replacing hyphens with spaces."""
    return text.strip().lower().replace("-", " ")


def _lemmatize(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])