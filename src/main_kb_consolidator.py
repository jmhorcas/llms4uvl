import argparse
import pathlib

from kb import KnowledgeBase, TaxonomyConstructor, NaturalLanguageProcessor


def _csv_file_path(value: str) -> str:
    if not value.lower().endswith('.csv'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .csv extension.')
    return value


def main(kb_filepath: str, threshold: float) -> None:
    path = pathlib.Path(kb_filepath)
    
    # Initialize the language model and the natural language processor.
    print('📥 Loading language model and natural language processor...')
    nlp = NaturalLanguageProcessor()
   
    # Load the knowledge base from the provided CSV file.
    print('📥 Loading KB from CSV...')
    kb = KnowledgeBase(nlp)
    kb.load_from_csv(kb_filepath)
    
    raw_triplets = len(kb.triplets)

    print('⚖️  Calculating consistency...')
    consistency = kb.calculate_consistency()

    # Consolidate knowledge base by normalizing and removing duplicates.
    print('💎 Consolidating KB...')
    print('    ✂️  Normalizing KB...')
    kb = kb.normalize()
    triplets_after_normalization = len(kb.triplets)
    print(f'    🔍 Deduplicating KB with threshold {threshold}...')
    kb = kb.deduplicate(threshold)
    triplets_after_deduplication = len(kb.triplets)
    print('    🧬 Clustering KB...')
    kb = kb.clustering()
    triplets_after_clustering = len(kb.triplets)
    
    # Save the consolidated knowledge base to a new CSV file.
    new_filepath = path.with_stem(f'{path.stem}_consolidated')
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

    # Construct taxonomy
    # builder = TaxonomyConstructor(language_model, beta=0.5)
    # taxonomy_graph = builder.construct_taxonomy(kb)

    # for parent, children in taxonomy_graph.items():
    #     print(f"Class [{parent}] children: {children}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Consolidator: Consolidate a knowledge base data.")
    parser.add_argument('csv_file', type=_csv_file_path, help='Knowledge base CSV file path.')
    parser.add_argument('--threshold', type=float, default=0.92, help='Similarity threshold for deduplication (default: 0.92).')
    args = parser.parse_args()

    main(args.csv_file, threshold=args.threshold)