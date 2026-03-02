
from dataclasses import dataclass
import csv
from kb import utils


@dataclass(frozen=True)
class Triplet:
    subject: str
    predicate: str
    object: str

    def __str__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"
    
    def to_tuple(self) -> tuple[str, str, str]:
        """Convert the Triplet dataclass instance into a tuple of strings (subject, predicate, object)."""
        return (self.subject, self.predicate, self.object)
    
    def to_sentence(self) -> str:
        """Convert a triplet (subject, predicate, object) into a single sentence for similarity comparison.
        For example, given the triplet (UVL, partOf, Language Levels), it will return "uvl partof language level" after normalization.
        """
        return f"{self.subject} {self.predicate} {self.object}"


class KnowledgeBase:

    CONCEPT_MAPPING_FILE = "../resources/concept_mapping.json"

    def __init__(self) -> None:
        self.triplets: list[Triplet] =[]

    def add_triplet(self, triplet: Triplet) -> None:
        self.triplets.append(triplet)
    
    def __len__(self):
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
        normalized_kb = KnowledgeBase()
        for triplet in self.triplets:
            normalized_triplet = Triplet(
                subject=utils.normalize_text(triplet.subject, self.CONCEPT_MAPPING_FILE),
                predicate=utils.normalize_text(triplet.predicate, self.CONCEPT_MAPPING_FILE),
                object=utils.normalize_text(triplet.object, self.CONCEPT_MAPPING_FILE)
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

    def deduplicate(self, threshold: float = 0.85) -> 'KnowledgeBase':
        """Remove those triplets that are equals or similar to other triplets."""
        kb = KnowledgeBase()
        for triplet1 in self.triplets:
            for triplet2 in self.triplets:
                pass
        pass

    def remove_exact_duplicates(self) -> 'KnowledgeBase':
        """Remove those triplets that are exactly the same as other triplets."""
        kb = KnowledgeBase()
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
    
    def remove_semantic_duplicates(self, threshold=0.92) -> 'KnowledgeBase':
        """Remove those triplets that are semantically similar to other triplets based on a similarity threshold."""
        unique_triples = utils.fast_semantic_deduplication(
            triples=[t.to_tuple() for t in self.triplets],
            threshold=threshold
        )
        kb = KnowledgeBase()
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