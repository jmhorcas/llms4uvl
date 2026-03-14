
from dataclasses import dataclass


@dataclass(frozen=True)
class Triplet:
    """A dataclass representing a triplet in the form of (subject, predicate, object)."""
    subject: str
    predicate: str
    object: str

    def __str__(self):
        """Return a string representation of the Triplet instance in the format (subject, predicate, object)."""
        return f"({self.subject}, {self.predicate}, {self.object})"
    
    def to_tuple(self) -> tuple[str, str, str]:
        """Convert the Triplet dataclass instance into a tuple of strings (subject, predicate, object)."""
        return (self.subject, self.predicate, self.object)
    
    def to_sentence(self) -> str:
        """Convert a triplet (subject, predicate, object) into a single sentence for similarity comparison."""
        return f"{self.subject} {self.predicate} {self.object}"
