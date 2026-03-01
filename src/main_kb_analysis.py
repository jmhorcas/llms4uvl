from kb import KnowledgeBase, KnowledgeComparator


#CSV_FILEPATH = '../resources/raw_data/uvl_groundtruth.csv'
CSV_FILEPATH = '../resources/raw_data/uvl_manualKB.csv'


def main() -> None:
    kb_groundtruth = KnowledgeBase()
    kb_groundtruth.load_from_csv(CSV_FILEPATH)

    print(f'Total triplets in ground truth: {len(kb_groundtruth.triplets)}')

    kb_groundtruth_normalized = kb_groundtruth.normalize()
    print(f'Total normalized triplets: {len(kb_groundtruth_normalized.triplets)}')
    #kb_groundtruth_normalized.save_to_csv('../uvl_groundtruth_normalized.csv')

    llm_kb = KnowledgeBase()
    llm_kb.load_from_csv('../resources/kb_uvl/OpenAI_GPT-5.2.csv')
    llm_kb_normalized = llm_kb.normalize()
    print(f'Total normalized triplets from LLM: {len(llm_kb_normalized.triplets)}')
    #llm_kb_normalized.save_to_csv('../uvl_ll

    # Hay que eliminar tripletas repetidas antes de comparar.
    #kb_comparator = KnowledgeComparator(kb_groundtruth_normalized, llm_kb_normalized)
    kb_comparator = KnowledgeComparator(kb_groundtruth, llm_kb)
    results = kb_comparator.compare(threshold=0.75)
    precision = kb_comparator.calculate_precision(results)
    recall = kb_comparator.calculate_recall(results)
    f1_score = kb_comparator.calculate_f1_score(precision, recall)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')


if __name__ == "__main__":
    main()