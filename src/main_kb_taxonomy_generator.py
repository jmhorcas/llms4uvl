import argparse
import pathlib

from kb import KnowledgeBase, TaxonomyConstructor, NaturalLanguageProcessor


def _csv_file_path(value: str) -> str:
    if not value.lower().endswith('.csv'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .csv extension.')
    return value


def main(kb_filepath: str, beta: float, inverted: bool) -> None:
    path = pathlib.Path(kb_filepath)
    
    # Initialize the language model and the natural language processor.
    print('📥 Loading language model and natural language processor...')
    nlp = NaturalLanguageProcessor()
   
    # Load the knowledge base from the provided CSV file.
    print('📥 Loading KB from CSV...')
    kb = KnowledgeBase(nlp)
    kb.load_from_csv(kb_filepath)
    
    # Construct taxonomy
    print('🔨 Constructing taxonomy...')
    builder = TaxonomyConstructor(nlp, beta=beta, inverted=inverted)
    taxonomy_graph = builder.construct_taxonomy(kb)
    print('📉 KB taxonomy:')
    builder.print_taxonomy_iterative_with_predicates()
    # for parent, children in taxonomy_graph.items():
    #     print(f"Class [{parent}] children: {children}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Taxonomy Generator: Generate a taxonomy from a knowledge base.")
    parser.add_argument('csv_file', type=_csv_file_path, help='Knowledge base CSV file path.')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter for similarity calculation (default: 0.5).')
    parser.add_argument('--inverted', action='store_true', help='Invert the taxonomy construction (default: False).')

    args = parser.parse_args()

    main(args.csv_file, args.beta, args.inverted)