import argparse
import pathlib

from kb import KnowledgeBase, utils, TaxonomyConstructor


def _csv_file_path(value: str) -> str:
    if not value.lower().endswith('.csv'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .csv extension.')
    return value


def main(kb_filepath: str) -> None:
    path = pathlib.Path(kb_filepath)
    
    nlp, language_model = utils.initialize_language_models()

    # Load the knowledge base from the provided CSV file.
    kb = KnowledgeBase(nlp, language_model)
    kb.load_from_csv(kb_filepath)
    
    total_triplets = len(kb.triplets)

    kb.consistency(kb_filepath)

    # Consolidate knowledge base by normalizing and removing duplicates.
    kb = kb.knowledge_consolidation()

    triplets_after_processing = len(kb.triplets)

    # Save the consolidated knowledge base to a new CSV file.
    new_filepath = path.with_stem(f'{path.stem}_consolidated')
    kb.save_to_csv(new_filepath)

    print(f'Knowledge base consolidated and saved to: {new_filepath}')
    print(f'Total triplets in original knowledge base: {total_triplets}')
    print(f'Total triplets after consolidation: {triplets_after_processing}')

    # Construct taxonomy
    # builder = TaxonomyConstructor(language_model, beta=0.5)
    # taxonomy_graph = builder.construct_taxonomy(kb)

    # for parent, children in taxonomy_graph.items():
    #     print(f"Class [{parent}] children: {children}")

    kb.consistency(kb_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Consolidator: Normalize a knowledge base data.")
    parser.add_argument('csv_file', type=_csv_file_path, help='Knowledge base CSV file path.')
    args = parser.parse_args()

    main(args.csv_file)