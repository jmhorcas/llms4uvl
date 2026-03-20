import argparse
from importlib.resources import path
import pathlib
import os

from kb import KnowledgeBase, TaxonomyConstructor, NaturalLanguageProcessor


KBS_FOLDER = '../resources/kb_uvl/ground_truth/consolidated/'


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


def main() -> None:
    threshold = 0.92
    nlp = NaturalLanguageProcessor()
    union_kb = KnowledgeBase(nlp)
    kbs_filepaths = get_csv_files(KBS_FOLDER)
    for kb_filepath in kbs_filepaths:
        kb = KnowledgeBase(nlp)
        kb.load_from_csv(kb_filepath)
        union_kb.join_kb(kb)
    
    kb = union_kb
    raw_triplets = len(kb.triplets)

    print('⚖️  Calculating consistency...')
    consistency = kb.calculate_consistency()

    # Consolidate knowledge base by normalizing and removing duplicates.
    print('💎 Consolidating KB...')
    print('    ✂️  Normalizing KB...')
    kb = kb.normalize()
    triplets_after_normalization = len(kb.triplets)
    print(f'    🔍 Deduplicating KB with threshold {threshold}...')
    kb = kb.remove_exact_duplicates()
    kb = kb.deduplicate(threshold)
    triplets_after_deduplication = len(kb.triplets)
    print('    🧬 Clustering KB...')
    kb = kb.clustering()
    triplets_after_clustering = len(kb.triplets)
    
    # Save the consolidated knowledge base to a new CSV file.
    new_filepath = 'union_kb_consolidated.csv'
    print(f'💾 Saving KB to {new_filepath}...')
    kb.save_to_csv(new_filepath)

    print('📊 Report:')
    print(f'    - Raw triplets: {raw_triplets}')
    print(f'    - Triplets after normalization: {triplets_after_normalization}')
    print(f'    - Triplets after deduplication: {triplets_after_deduplication}')
    print(f'    - Triplets after clustering: {triplets_after_clustering}')
    print(f'    - 📈 Consistency: ', end='')
    if not consistency:
        print('Not available')
    else:
        print()
        print(f'        - Macro: {consistency['Macro']}')
        print(f'        - Micro: {consistency['Micro']}')
        print(f'        - Avg_Triplets: {consistency['Avg_Triplets']}')
        print('        - Seed_Scores:')
        for seed, score in consistency['Seed_Scores'].items():
            print(f'            - {seed}: {score}')


if __name__ == "__main__":
    main()