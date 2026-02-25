
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

