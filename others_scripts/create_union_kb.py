import argparse
from importlib.resources import path
import pathlib
import os
import csv

from kb import KnowledgeBase, TaxonomyConstructor, NaturalLanguageProcessor, KnowledgeComparator


KBS_FOLDER = '../resources/kb_uvl/ground_truth/raw/'
CSV_HEADER = [
    'resource', 
    'seed',
    'triplets', 
    'precision', 
    'recall', 
    'f1_score', 
    'hallucination_rate',
    'hallucinations'
]

def get_csv_files(path):
    """Returns a list of .csv files given a file or directory."""
    if os.path.isfile(path):
        return [path]
    # It's a directory: search recursively
    csv_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)



def write_csv_row(csv_filepath: str, row: dict) -> None:
    """Escribe una fila en el CSV, creando el fichero con cabecera si no existe."""
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



def main() -> None:
    threshold = 0.92
    nlp = NaturalLanguageProcessor()
    kbs_filepaths = get_csv_files(KBS_FOLDER)
    kbs_list = []
    for kb_filepath in kbs_filepaths:
        kb = KnowledgeBase(nlp)
        kb.load_from_csv(kb_filepath)
        kbs_list.append(kb)

    union_kb = KnowledgeBase(nlp)
    dict_union_kbs_by_seed = union_kb.create_global_union_by_seed(kbs_list)  # this also consolidate

    for kb_filepath in kbs_filepaths:
        print(f'📥 Comparing KB from {kb_filepath}...')
        kb = KnowledgeBase(nlp)
        kb.load_from_csv(kb_filepath)
        for seed, seed_kb in kb.iterations_seeds.items():
            kb_seed = KnowledgeBase(nlp)
            for run_id, run_kb in seed_kb.items():
                kb_seed.join_kb(run_kb)
            kb_seed = kb_seed.consolidate(threshold)

            kb_comparator = KnowledgeComparator(nlp, dict_union_kbs_by_seed[seed], kb_seed)
            print('⚖️  Comparing KBs and calculating evaluation metrics...')
            results = kb_comparator.compare(threshold=0.85)
            precision = kb_comparator.calculate_precision(results)
            recall = kb_comparator.calculate_recall(results)
            f1_score = kb_comparator.calculate_f1_score(precision, recall)
            hallucination_rate = kb_comparator.calculate_hallucination_rate(results)
            hallucinations = kb_comparator.get_hallucinations(results)

            print(f'📊 Report for seed {seed}:')
            print(f'    - Triplets in Ground Truth KB: {len(dict_union_kbs_by_seed[seed].triplets)}')
            print(f'    - Triplets in LLM KB: {len(kb_seed.triplets)}')
            print(f'    - Precision: {precision:.4f}')
            print(f'    - Recall: {recall:.4f}')
            print(f'    - F1 Score: {f1_score:.4f}')
            print(f'    - Hallucination Rate: {hallucination_rate:.4f}')
            print(f'    - Number of hallucinations: {len(hallucinations)}')

            csv_row = {
                'resource': pathlib.Path(kb_filepath).stem,
                'seed': seed,
                'triplets': len(kb_seed.triplets),
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'hallucination_rate': hallucination_rate,
                'hallucinations': len(hallucinations)
            }
            write_csv_row('union_kb_results_by_seed.csv', csv_row)

if __name__ == "__main__":
    main()