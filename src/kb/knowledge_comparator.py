import torch
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

from kb import KnowledgeBase, Triplet, NaturalLanguageProcessor


class KnowledgeComparator:
    
    def __init__(self, nlp: NaturalLanguageProcessor, ground_truth_kb: KnowledgeBase, llm_kb: KnowledgeBase) -> None:
        self.nlp = nlp
        self.ground_truth_kb = ground_truth_kb
        self.llm_kb = llm_kb

    def compare(self, threshold: float = 0.85) -> list[tuple[Triplet, Triplet, float]]:
        """Compare the triplets in this knowledge base with those in another knowledge base and return a list of tuples containing the compared triplets and their similarity score.
        
        This method uses a Semantic Similarity Matrix approach, where each triplet is converted into a sentence and then into an embedding vector. 
        The similarity between the triplets is computed using cosine similarity of their embeddings. 
        This allows for a more robust comparison that can capture semantic similarities even when the wording is different.
        """
        if not self.ground_truth_kb.triplets or not self.llm_kb.triplets:
            return [{"real": t, "match": False} for t in self.ground_truth_kb.triplets]

        # 1. Preparar textos para codificación por componentes
        gt_triples = self.ground_truth_kb.triplets
        llm_triples = self.llm_kb.triplets

        # Codificamos oraciones completas para la similitud global
        gt_sentences = [t.to_sentence() for t in gt_triples]
        llm_sentences = [t.to_sentence() for t in llm_triples]

        # 2. Generar Embeddings usando util.cos_sim requiere tensores
        # Codificamos todo por separado para máxima precisión atómica
        gt_vecs = self.nlp.language_model.encode(gt_sentences, convert_to_tensor=True)
        llm_vecs = self.nlp.language_model.encode(llm_sentences, convert_to_tensor=True)
        
        # Componentes para el score atómico
        gt_sub_vecs = self.nlp.language_model.encode([t.subject for t in gt_triples], convert_to_tensor=True)
        llm_sub_vecs = self.nlp.language_model.encode([t.subject for t in llm_triples], convert_to_tensor=True)
        
        gt_pre_vecs = self.nlp.language_model.encode([t.predicate for t in gt_triples], convert_to_tensor=True)
        llm_pre_vecs = self.nlp.language_model.encode([t.predicate for t in llm_triples], convert_to_tensor=True)
        
        gt_obj_vecs = self.nlp.language_model.encode([t.object for t in gt_triples], convert_to_tensor=True)
        llm_obj_vecs = self.nlp.language_model.encode([t.object for t in llm_triples], convert_to_tensor=True)

        # 3. Calcular matrices de similitud
        sim_matrix_global = util.cos_sim(gt_vecs, llm_vecs)
        sim_matrix_sub = util.cos_sim(gt_sub_vecs, llm_sub_vecs)
        sim_matrix_pre = util.cos_sim(gt_pre_vecs, llm_pre_vecs)
        sim_matrix_obj = util.cos_sim(gt_obj_vecs, llm_obj_vecs)

        results = []
        used_llm_indices = set()

        for i in range(len(gt_triples)):
            # Buscamos el mejor match en la fila i
            # Obtenemos los candidatos ordenados por similitud global
            row_scores = sim_matrix_global[i]
            potential_indices = torch.argsort(row_scores, descending=True)

            best_match_idx = -1
            best_score = 0.0

            for j_idx in potential_indices:
                j = j_idx.item()
                global_score = row_scores[j].item()

                if global_score < 0.6: # Filtro rápido igual que en deduplicate
                    break
                
                # --- PROTECCIÓN DE OPERADORES ---
                ops_gt = {op for op in self.nlp.special_cases if op in gt_triples[i].object}
                ops_llm = {op for op in self.nlp.special_cases if op in llm_triples[j].object}

                if ops_gt or ops_llm:
                    if ops_gt != ops_llm:
                        continue 

                # --- SCORE HÍBRIDO (Semántica Atómica + Léxica) ---
                score_s = sim_matrix_sub[i][j].item()
                score_p = sim_matrix_pre[i][j].item()
                score_o = sim_matrix_obj[i][j].item()
                
                score_atomic = (score_s * 0.4) + (score_o * 0.4) + (score_p * 0.2)
                best_semantic = max(global_score, score_atomic)
                
                # Similitud léxica (Fuzzy)
                lex_score = fuzz.token_set_ratio(gt_sentences[i], llm_sentences[j]) / 100.0
                
                final_score = (best_semantic * 0.7) + (lex_score * 0.3)

                if final_score >= threshold:
                    best_match_idx = j
                    best_score = final_score
                    break

            if best_match_idx != -1:
                is_first_time_used = best_match_idx not in used_llm_indices
                used_llm_indices.add(best_match_idx)
                
                results.append({
                    "real": gt_triples[i],
                    "llm": llm_triples[best_match_idx],
                    "score": best_score,
                    "match": True,
                    "llm_index": best_match_idx,
                    "is_unique_hit": is_first_time_used
                })
            else:
                results.append({
                    "real": gt_triples[i],
                    "match": False
                })
        
        return results
    

    def calculate_precision(self, batch_results: list[tuple[Triplet, Triplet, float]]) -> float:
        """Precision: From all the triplets generated by the LLM, what percentage is correct (true)?
        It measures the fidelity and lack of hallucinations.
        
        Precision = Correctly Matched Triplets / Total LLM-Generated Triplets
        """
        total_llm_triples = len(self.llm_kb.triplets)
        if total_llm_triples == 0:
            return 0.0
        
        # Count how many of the LLM-generated triplets had a match
        unique_matches = sum(1 for res in batch_results if res.get("match") and res.get("is_unique_hit"))        
        precision = unique_matches / total_llm_triples
        return precision

    def calculate_recall(self, batch_results: list[tuple[Triplet, Triplet, float]]) -> float:
        """Recall: Of all the knowledge that exists in the ground truth, how much did the LLM retrieve?
        It measures the coverage or "% of knowledge retrieved".

        Recall = Correctly Matched Triplets / Total Triplets in the Ground Truth
        """
        total_triples_in_ground_truth = len(self.ground_truth_kb.triplets)
        if total_triples_in_ground_truth == 0:
            return 0.0
        
        # Count how many of the ground truth triplets were found by the LLM
        matches = sum(1 for res in batch_results if res.get("match") is True)
        
        recall = matches / total_triples_in_ground_truth
        return float(recall)

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """F1 Score: The harmonic mean of precision and recall, providing a single metric that balances both aspects of performance.

        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        """
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def calculate_hallucination_rate(self, batch_results: list[dict]) -> float:
        """
        Devuelve el % de tripletas que el LLM generó pero que no 
        tienen sustento en el Ground Truth.
        """
        return 1.0 - self.calculate_precision(batch_results)

    def get_hallucinations(self, batch_results: list[dict]) -> list[Triplet]:
        """
        Retorna la lista de tripletas que el LLM 'alucinó' 
        (las que no hicieron match con nada del GT).
        """
        # 1. Obtenemos los índices de las tripletas del LLM que sí fueron correctas
        matched_indices = {res["llm_index"] for res in batch_results if res.get("match")}
        
        # 2. Las alucinaciones son todas las que NO están en ese conjunto de índices
        hallucinations = [
            triplet for i, triplet in enumerate(self.llm_kb.triplets)
            if i not in matched_indices
        ]
        
        return hallucinations