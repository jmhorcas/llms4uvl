import json
from collections import Counter

import numpy
import nltk
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

from kb import Triplet


class NaturalLanguageProcessor:
    """Class representing a natural language processor for handling text processing tasks."""

    SPACY_LANGUAGE_MODEL = "en_core_web_sm"
    TRANSFORMER_LANGUAGE_MODEL = 'all-MiniLM-L6-v2'  # Light and free model (approx 80MB)
    CONCEPT_MAPPING_FILE = "../resources/concept_mapping.json"
    SPECIAL_CASES = ['==', '<=', '>=', '!=', '>', '<', '=>', '<=>', '&', '|', '!', '+', '-', '/', '*']

    def __init__(self, 
                 spacy_language_model: str = SPACY_LANGUAGE_MODEL, 
                 transformer_language_model: str = TRANSFORMER_LANGUAGE_MODEL,
                 concept_mapping_file: str = CONCEPT_MAPPING_FILE,
                 special_cases: list[str] = SPECIAL_CASES) -> None:
        """Initialize the natural language processor with the specified language models and concept mapping."""
        nltk.download('wordnet', quiet=True)
        self.nlp: spacy.Language = spacy.load(spacy_language_model)
        for case in special_cases:
            self.nlp.tokenizer.add_special_case(case, [{spacy.symbols.ORTH: case}])
        self.language_model: SentenceTransformer = SentenceTransformer(transformer_language_model)
        self.concept_mapping: dict[str, str] = load_concept_mapping(concept_mapping_file)
        self.special_cases = special_cases

    def case_folding(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower().strip()
    
    def lemmatization(self, text: str) -> str:
        """Lemmatize the text (convert words to their base form)."""
        doc = self.nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc]
        return " ".join(lemmatized_tokens)

    def remove_stopwords(self, text: str) -> str:
        """Remove stop words (articles, prepositions, etc.) from the text."""
        doc = self.nlp(text)
        keep_tokens = [t for t in doc if not t.is_stop and not t.is_punct]
        if not keep_tokens:
            return text  # Return original text if all tokens are removed
        result = "".join([t.text_with_ws for t in keep_tokens]).strip()  # Preserve original spacing
        return result if result else text  # Return original text if result is empty after removing stop words

    def get_synonyms(self, word: str) -> list:
        """Get synonyms for a given word using the specified language model."""
        doc = self.nlp(word)
        synonyms = set()
        for token in doc:
            for syn in nltk.corpus.wordnet.synsets(token.lemma_):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
        return list(synonyms)

    def get_concept_mapping(self, text: str) -> str:
        """Get an established concept defined in the concept mapping or the original text if not found."""
        return self.concept_mapping.get(text, text)

    def normalize_text(self, text: str) -> str:
        """Normalize the text by applying case folding, concept mapping, stop word removal, and lemmatization."""
        original_text = text
        text = self.case_folding(text)
        mapped_concept = self.get_concept_mapping(text)
        if mapped_concept != text:
            return mapped_concept  # Return the mapped concept if found        
        text = self.remove_stopwords(text)
        text = self.lemmatization(text)
        mapped_concept = self.get_concept_mapping(text)
        if text == '':
            return original_text  # Return the original text if no mapping found for it
        if mapped_concept != text:
            return mapped_concept  # Return the mapped concept if found after normalization
        return text  # Return the normalized text if no mapping found for it

    def get_similarity(self, text1: str, text2: str, weight_semantic: float = 0.7, weight_lexical: float = 0.3) -> float:
        """Compute the similarity between two texts by combining Semantic Similarity (using embeddings) and Lexical Similarity (using Fuzzy Matching).
        
        This method is designed to be robust for comparing triplets as sentences, where the order of words may vary but the meaning is similar (e.g., "UVL supports Boolean" vs "Boolean is part of UVL").

        Args:
            text1, text2: Texts to compare (e.g., concatenated triplets).
            weight_semantic: Weight given to semantic similarity (0.0 to 1.0).
            weight_lexical: Weight given to lexical similarity (0.0 to 1.0).
        """
        if not text1 or not text2:
            return 0.0

        # 1. Semantic Similarity (based on the language model)
        # Convert the texts into vectors and compute cosine similarity
        embeddings = self.language_model.encode([text1, text2])
        sem_score = util.cos_sim(embeddings[0], embeddings[1]).item()

        # 2. Lexical Similarity (based on characters/Fuzzy Matching)
        # token_set_ratio is excellent because it ignores word order
        # Example: "UVL Boolean" vs "Boolean UVL" would return 100%
        lex_score = fuzz.token_set_ratio(text1, text2) / 100.0
    
        # 3. Combined Score
        final_score = (sem_score * weight_semantic) + (lex_score * weight_lexical)
        
        return float(final_score)

    def compute_triplet_similarity(self, 
                                   triplet1: tuple[str, str, str], 
                                   triplet2: tuple[str, str, str], 
                                   w_semantic: float = 0.7, 
                                   w_lexical: float = 0.3) -> float:
        """Compute semantic and lexical similarity of triplets."""
        # 1. Prepare texts:
        # Full sentences
        t1_full = f"{triplet1[0]} {triplet1[1]} {triplet1[2]}"
        t2_full = f"{triplet2[0]} {triplet2[1]} {triplet2[2]}"
        
        # Individual components
        texts_to_encode = [
            t1_full, t2_full,         # Global (0, 1)
            triplet1[0], triplet2[0], # Sujetos (2, 3)
            triplet1[1], triplet2[1], # Predicados (4, 5)
            triplet1[2], triplet2[2]  # Objetos (6, 7)
        ]

        # 2. Encode (Batching) - for efficiency
        embs = self.language_model.encode(texts_to_encode, convert_to_tensor=True)

        # 3. Semantic similarity (Cosine)
        sim_global = util.cos_sim(embs[0], embs[1]).item()
        sim_s = util.cos_sim(embs[2], embs[3]).item()
        sim_p = util.cos_sim(embs[4], embs[5]).item()
        sim_o = util.cos_sim(embs[6], embs[7]).item()

        # 4. Lexic similarity (Fuzzy)
        # We use token_set_ratio because it ignore the order of the words
        lex_score = fuzz.token_set_ratio(t1_full, t2_full) / 100.0

        # 5. Decision logic (Hybrid)
        # Atomic score: Weighted (S:0.4, O:0.4, P:0.2)
        score_atomic = (sim_s * 0.4) + (sim_o * 0.4) + (sim_p * 0.2)
        
        # Semantic Score combined: Best of global and detailed (for component)
        # This solve "A supports B" vs "B is supported by A"
        best_semantic = max(sim_global, score_atomic)

        # 6. Final result
        final_score = (best_semantic * w_semantic) + (lex_score * w_lexical)
        
        return float(final_score)

    def deduplicate_triplets(self, triples: list[Triplet], threshold: float = 0.90) -> list[Triplet]:
        """Deduplicate a list of triplets by comparing their semantic and lexical similarity, and removing those that are too similar based on a specified threshold."""
        if not triples:
            return []

        n = len(triples)
        # Prepare texts for batch encoding
        sentences = [t.to_sentence() for t in triples]
        subjects = [t.subject for t in triples]
        predicates = [t.predicate for t in triples]
        objects = [t.object for t in triples]
        
        all_texts = sentences + subjects + predicates + objects
        all_embs = self.language_model.encode(all_texts, convert_to_tensor=True)

        # Split results from batch
        s_embs = all_embs[0:n]          # Globals
        sub_embs = all_embs[n:2*n]      # Subjects
        pre_embs = all_embs[2*n:3*n]    # Predicates
        obj_embs = all_embs[3*n:4*n]    # Objects

        # Pre-calculate the global matrix
        sim_matrix_global = util.cos_sim(s_embs, s_embs).cpu().numpy()

        indices_to_remove = set()

        # Smart comparison
        for i in range(n):
            if i in indices_to_remove:
                continue
            
            for j in range(i + 1, n):
                if j in indices_to_remove:
                    continue

                # Global semantic
                sim_global = sim_matrix_global[i][j]

                # Fast filtering: if global is low, omit syntax comparison
                if sim_global < 0.6: 
                    continue

                # Atomic score (component by component)
                sim_s = util.cos_sim(sub_embs[i], sub_embs[j]).item()
                sim_p = util.cos_sim(pre_embs[i], pre_embs[j]).item()
                sim_o = util.cos_sim(obj_embs[i], obj_embs[j]).item()
                
                score_atomic = (sim_s * 0.4) + (sim_o * 0.4) + (sim_p * 0.2)
                
                # Best of both semantics scores
                best_semantic = max(sim_global, score_atomic)

                # Lexical similarity
                lex_score = fuzz.token_set_ratio(sentences[i], sentences[j]) / 100.0

                # Final hybrid score
                final_score = (best_semantic * 0.7) + (lex_score * 0.3)

                if final_score >= threshold:
                    indices_to_remove.add(j)

        return [triples[i] for i in range(n) if i not in indices_to_remove]

    def relation_clustering(self,
                            relations: set[str], 
                            alpha: float = 1.4, 
                            H: float = 0.95, 
                            L: float = 0.75) -> dict[str, str]:
        """
        A greedy clustering algorithm that merges relation r into a more frequen relation s,
        selected as the one with the highest textual embedding similarity to r among all relations more frequent than r,
        if the similarity is greaer than an adaptive threshold.
        This threshold varies with the frequency of the relation, leading to a more aggressive removal of relations with low frequency.
        Algorithm from [Hu2025_ExtensiveMaterialization](https://doi.org/10.18653/v1/2025.acl-long.789).
        Args:
            relations: Set of relations.
            alpha: Sensitivity threshold (default 1.4).
            H: Maximum threshold for very frequent relations (0.95).
            L: Minimum threshold for rare relations (0.75).
        """
        if not relations:
            return {}

        # Count frequencies and sort: R_sort <- sort R by frequency descending
        counts = Counter(relations)
        sorted_relations = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        r_names = [r for r, f in sorted_relations]
        r_freqs = [f for r, f in sorted_relations]
        max_f = r_freqs[0]

        # Pre-calculate embeddings for better performance
        embeddings = self.language_model.encode(r_names, convert_to_tensor=True)
        
        # C: Map original relation -> Representative relation (the most frequent in the group)
        clusters = {} 
        cluster_heads_names = []
        cluster_heads_embeddings = []

        # For r in R_sort
        for i, r_name in enumerate(r_names):
            f = r_freqs[i]
            r_emb = embeddings[i]
            
            best_match = None
            max_sim = -1

            # If we already have cluster heads, compare r_emb against all of them at once
            if cluster_heads_embeddings:
                heads_tensor = torch.stack(cluster_heads_embeddings)
                # Calculate cosine similarities
                sim_matrix = util.cos_sim(r_emb.reshape(1, -1), heads_tensor)
                similarities = sim_matrix[0]
                
                # Search for the highest value and its index
                max_sim_tensor = similarities.max()
                max_sim = float(max_sim_tensor.item())
                best_idx = similarities.argmax().item()
                best_match = cluster_heads_names[best_idx]

            # Calculate the adaptive threshold T
            log_f = numpy.log(f) if f > 1 else 0
            log_max_f = numpy.log(max_f) if max_f > 1 else 1
            
            if log_max_f > 0:
                threshold = L + (H - L) * (pow(log_f / log_max_f, alpha))
            else:
                threshold = H

            # Decide: Does r join an existing cluster or create a new one?
            if best_match and max_sim > threshold:
                clusters[r_name] = best_match
            else:
                clusters[r_name] = r_name
                cluster_heads_names.append(r_name)
                cluster_heads_embeddings.append(r_emb)

        return clusters


def load_concept_mapping(concept_mapping_file: str) -> dict[str, str]:
    """Load the concept mapping from a JSON file."""
    with open(concept_mapping_file, 'r', encoding='utf-8') as file:
        return json.load(file)
