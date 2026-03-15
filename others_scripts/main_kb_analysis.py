from kb import KnowledgeBase, KnowledgeComparator


GROUND_TRUTH_CSV = '../resources/raw_data/UVL_KB_GroundTruth-NotebookLLM-All.csv'
LLM_CSV = '../resources/kb_uvl/20260305_214918_gemini-3.1-flash-lite-preview.csv'


def main() -> None:
    kb_groundtruth = KnowledgeBase()
    kb_groundtruth.load_from_csv(GROUND_TRUTH_CSV)

    print(f'Total triplets in ground truth: {len(kb_groundtruth.triplets)}')

    kb_groundtruth_normalized = kb_groundtruth.normalize()
    print(f'Total normalized triplets: {len(kb_groundtruth_normalized.triplets)}')

    kb_groundtruth_cleaned = kb_groundtruth_normalized.remove_exact_duplicates()
    print(f'Total normalized triplets after removing duplicates: {len(kb_groundtruth_cleaned.triplets)}')
    kb_groundtruth_cleaned.save_to_csv('../UVL_KB_GroundTruth-NotebookLLM-PaperUVL_cleaned.csv')
    #kb_cleaned = kb_cleaned.remove_semantic_duplicates(threshold=0.95)
    #print(f'Total normalized triplets after deduplication: {len(kb_cleaned2.triplets)}')
    #kb_cleaned2.save_to_csv('../UVL_KB_GroundTruth-NotebookLLM-PaperUVL_deduplicated.csv')
    #raise Exception("Stop execution after processing ground truth. Comment this line to continue with LLM comparison.")

    llm_kb = KnowledgeBase()
    llm_kb.load_from_csv(LLM_CSV)
    llm_kb_normalized = llm_kb.normalize()
    print(f'Total normalized triplets from LLM: {len(llm_kb_normalized.triplets)}')
    llm_kb_cleaned = llm_kb_normalized.remove_exact_duplicates()
    print(f'Total normalized triplets after removing duplicates: {len(llm_kb_cleaned.triplets)}')
    llm_kb_cleaned.save_to_csv('../UVL_KB_LLM_cleaned.csv')

    
    # Hay que eliminar tripletas repetidas antes de comparar.
    #kb_comparator = KnowledgeComparator(kb_groundtruth_normalized, llm_kb_normalized)
    kb_comparator = KnowledgeComparator(kb_groundtruth_cleaned, llm_kb_cleaned)
    results = kb_comparator.compare(threshold=0.75)
    precision = kb_comparator.calculate_precision(results)
    recall = kb_comparator.calculate_recall(results)
    f1_score = kb_comparator.calculate_f1_score(precision, recall)
    hallucination_rate = kb_comparator.calculate_hallucination_rate(results)
    hallucinations = kb_comparator.get_hallucinations(results)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print(f'Hallucination Rate: {hallucination_rate:.4f}')
    print(f'Number of hallucinations: {len(hallucinations)}')


if __name__ == "__main__":
    main()