import argparse
import pathlib

from kb import KnowledgeBase, KnowledgeComparator, NaturalLanguageProcessor


def _csv_file_path(value: str) -> str:
    if not value.lower().endswith('.csv'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .csv extension.')
    return value


def main(gt_kb_filepath: str, llm_kb_filepath: str, threshold: float) -> None:
    gt_path = pathlib.Path(gt_kb_filepath)
    llm_path = pathlib.Path(llm_kb_filepath)

    # Initialize the language model and the natural language processor.
    print('📥 Loading language model and natural language processor...')
    nlp = NaturalLanguageProcessor()

    # Load the ground truth knowledge base from the provided CSV file.
    print('📥 Loading ground truth KB from CSV...')
    gt_kb = KnowledgeBase(nlp)
    gt_kb.load_from_csv(gt_kb_filepath)

    # Load the LLM-generated knowledge base from the provided CSV file.
    print('📥 Loading LLM-generated KB from CSV...')
    llm_kb = KnowledgeBase(nlp)
    llm_kb.load_from_csv(llm_kb_filepath)
    
    # Compare the two knowledge bases and calculate evaluation metrics.
    kb_comparator = KnowledgeComparator(nlp, gt_kb, llm_kb)
    print('⚖️  Comparing KBs and calculating evaluation metrics...')
    results = kb_comparator.compare(threshold=0.85)
    precision = kb_comparator.calculate_precision(results)
    recall = kb_comparator.calculate_recall(results)
    f1_score = kb_comparator.calculate_f1_score(precision, recall)
    hallucination_rate = kb_comparator.calculate_hallucination_rate(results)
    hallucinations = kb_comparator.get_hallucinations(results)

    print('📊 Report:')
    print(f'    - Triplets in Ground Truth KB: {len(gt_kb.triplets)}')
    print(f'    - Triplets in LLM KB: {len(llm_kb.triplets)}')
    print(f'    - Precision: {precision:.4f}')
    print(f'    - Recall: {recall:.4f}')
    print(f'    - F1 Score: {f1_score:.4f}')
    print(f'    - Hallucination Rate: {hallucination_rate:.4f}')
    print(f'    - Number of hallucinations: {len(hallucinations)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Comparator: Compare two knowledge bases.")
    parser.add_argument('gt_kb', type=_csv_file_path, help='Ground truth knowledge base CSV file path.')
    parser.add_argument('llm_kb', type=_csv_file_path, help='LLM-generated knowledge base CSV file path.')
    parser.add_argument('--threshold', type=float, default=0.92, help='Similarity threshold for deduplication (default: 0.92).')
    args = parser.parse_args()

    main(args.gt_kb, args.llm_kb, threshold=args.threshold)