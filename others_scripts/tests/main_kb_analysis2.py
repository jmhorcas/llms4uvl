from kb import KnowledgeBase, KnowledgeComparator


CSV_FILEPATH = '../resources/raw_data/UVL_KB_GroundTruth-NotebookLLM-PaperUVL.csv'
CSV_FILEPATH2 = '../resources/raw_data/UVL_KB_GroundTruth-NotebookLLM-All.csv'
#CSV_FILEPATH = '../resources/raw_data/uvl_manualKB.csv'


def main() -> None:
    kb_groundtruth = KnowledgeBase()
    kb_groundtruth.load_from_csv(CSV_FILEPATH)

    print(f'Total triplets in ground truth: {len(kb_groundtruth.triplets)}')

    kb_groundtruth_normalized = kb_groundtruth.normalize()
    print(f'Total normalized triplets: {len(kb_groundtruth_normalized.triplets)}')

    kb_cleaned = kb_groundtruth_normalized.remove_exact_duplicates()
    print(f'Total normalized triplets after removing duplicates: {len(kb_cleaned.triplets)}')
    kb_cleaned.save_to_csv('../UVL_KB_GroundTruth-NotebookLLM-PaperUVL_cleaned.csv')
    #kb_cleaned = kb_cleaned.remove_semantic_duplicates(threshold=0.95)
    #print(f'Total normalized triplets after deduplication: {len(kb_cleaned2.triplets)}')
    #kb_cleaned2.save_to_csv('../UVL_KB_GroundTruth-NotebookLLM-PaperUVL_deduplicated.csv')
    #raise Exception("Stop execution after processing ground truth. Comment this line to continue with LLM comparison.")

    





    kb_groundtruth2 = KnowledgeBase()
    kb_groundtruth2.load_from_csv(CSV_FILEPATH2)

    print(f'Total triplets in ground truth: {len(kb_groundtruth2.triplets)}')

    kb_groundtruth_normalized2 = kb_groundtruth2.normalize()
    print(f'Total normalized triplets: {len(kb_groundtruth_normalized2.triplets)}')

    kb_cleaned2 = kb_groundtruth_normalized2.remove_exact_duplicates()
    print(f'Total normalized triplets after removing duplicates: {len(kb_cleaned2.triplets)}')
    kb_cleaned2.save_to_csv('../UVL_KB_GroundTruth-NotebookLLM-All_cleaned.csv')
    #kb_cleaned2 = kb_cleaned2.remove_semantic_duplicates(threshold=0.95)
    #print(f'Total normalized triplets after deduplication: {len(kb_cleaned2.triplets)}')
    #kb_cleaned2.save_to_csv('../UVL_KB_GroundTruth-NotebookLLM-All_deduplicated.csv')
    #raise Exception("Stop execution after processing ground truth. Comment this line to continue with LLM comparison.")

    union_kb = KnowledgeBase()
    union_kb.join_kb(kb_cleaned)
    union_kb.join_kb(kb_cleaned2)
    union_kb = union_kb.remove_exact_duplicates()
    print(f'Total triplets in the union of both ground truths: {len(union_kb.triplets)}')
    

    # llm_kb = KnowledgeBase()
    # llm_kb.load_from_csv('../resources/kb_uvl/OpenAI_GPT-5.2.csv')
    # llm_kb_normalized = llm_kb.normalize()
    # print(f'Total normalized triplets from LLM: {len(llm_kb_normalized.triplets)}')
    # #llm_kb_normalized.save_to_csv('../uvl_ll

    # Hay que eliminar tripletas repetidas antes de comparar.
    #kb_comparator = KnowledgeComparator(kb_groundtruth_normalized, llm_kb_normalized)
    kb_comparator = KnowledgeComparator(union_kb, kb_cleaned2)
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
    for h in hallucinations:
        print(f'- {h}')
        
if __name__ == "__main__":
    main()