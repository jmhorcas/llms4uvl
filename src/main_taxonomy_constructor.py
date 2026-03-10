import argparse
import pathlib
from treelib import Tree

from kb import KnowledgeBase, utils, TaxonomyConstructor


def _csv_file_path(value: str) -> str:
    if not value.lower().endswith('.csv'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .csv extension.')
    return value


def print_taxonomy(taxonomy: dict[str, list[str]], node: str, indent: str = "", is_last: bool = True):
    # Select the visual prefix
    marker = "└── " if is_last else "├── "
    print(indent + marker + str(node))
    # Prepare the indentation for the children
    indent += "    " if is_last else "│   "
    # Get children if they exist
    children = taxonomy.get(node, [])
    for i, child in enumerate(children):
        last_child = (i == len(children) - 1)
        print_taxonomy(taxonomy, child, indent, last_child)


def display_pretty_taxonomy(taxonomy: dict[str, list[str]]) -> None:
    tree = Tree()
    
    # 1. Identificar raíces
    all_children = set(item for sublist in taxonomy.values() for item in sublist)
    roots = [n for n in taxonomy.keys() if n not in all_children]
    
    # 2. Añadir raíces al árbol de treelib
    for root in roots:
        if not tree.contains(root):
            tree.create_node(root, root) # name, identifier
            
        # 3. Añadir hijos recursivamente
        stack = [root]
        while stack:
            current = stack.pop()
            for child in taxonomy.get(current, []):
                if not tree.contains(child):
                    tree.create_node(child, child, parent=current)
                    stack.append(child)
    
    tree.show()


def main(kb_filepath: str) -> None:
    path = pathlib.Path(kb_filepath)
    
    nlp, language_model = utils.initialize_language_models()

    # Load the knowledge base from the provided CSV file.
    kb = KnowledgeBase(nlp, language_model)
    kb.load_from_csv(kb_filepath)
    
    # Construct taxonomy
    builder = TaxonomyConstructor(language_model, beta=0.5)
    taxonomy_graph = builder.construct_taxonomy(kb)
    for parent, children in taxonomy_graph.items():
        print(f"Class [{parent}] children: {children}")
    
    # Identify root nodes (nodes that are not children of any other node)
    all_children = set(item for sublist in taxonomy_graph.values() for item in sublist)
    roots = [n for n in taxonomy_graph.keys() if n not in all_children]
    for root in roots:
        print_taxonomy(taxonomy_graph, root)

    display_pretty_taxonomy(taxonomy_graph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy Constructor: Construct a taxonomy from a knowledge base.")
    parser.add_argument('csv_file', type=_csv_file_path, help='Knowledge base CSV file path.')
    args = parser.parse_args()

    main(args.csv_file)