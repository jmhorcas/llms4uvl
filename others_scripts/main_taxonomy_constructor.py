import argparse
import pathlib

from kb import KnowledgeBase, utils, TaxonomyConstructor, TaxonomyConstructorInverted


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


def print_pretty_inverted_taxonomy(taxonomy, node=None, indent="", is_last=True, visited=None):
    if visited is None:
        visited = set()
    
    # Identificar raíces: Nodos que son llaves pero NO son hijos de nadie
    if node is None:
        all_children = set(h for hijos in taxonomy.values() for h in hijos)
        roots = [n for n in taxonomy.keys() if n not in all_children]
        
        if not roots:
            # Si no hay raíces claras (todo es un ciclo), tomamos las llaves principales
            roots = list(taxonomy.keys())[:5] # Solo las primeras para no saturar
            print("(!) Aviso: No se detectaron raíces claras, imprimiendo nodos principales:")

        for i, root in enumerate(roots):
            print_pretty_inverted_taxonomy(taxonomy, root, "", i == len(roots) - 1, visited)
        return

    # CONTROL DE CICLOS: Si ya vimos este nodo en esta rama, paramos
    if node in visited:
        print(indent + ("└── " if is_last else "├── ") + f"{node} (RECURSION)")
        return
    
    visited.add(node)

    # Imprimir el nodo actual
    marker = "└── " if is_last else "├── "
    print(indent + marker + str(node))
    
    # Preparar indentación para hijos
    new_indent = indent + ("    " if is_last else "│   ")
    children = taxonomy.get(node, [])
    
    for i, child in enumerate(children):
        is_child_last = (i == len(children) - 1)
        print_pretty_inverted_taxonomy(taxonomy, child, new_indent, is_child_last, visited.copy())

def print_iterative(taxonomy):
    all_children = set(h for hijos in taxonomy.values() for h in hijos)
    roots = [n for n in taxonomy.keys() if n not in all_children]
    
    stack = [(r, "", True) for r in reversed(roots)]
    visited = set()

    while stack:
        node, indent, is_last = stack.pop()
        if node in visited: continue
        visited.add(node)
        
        marker = "└── " if is_last else "├── "
        print(indent + marker + node)
        
        children = taxonomy.get(node, [])
        new_indent = indent + ("    " if is_last else "│   ")
        for i, child in enumerate(reversed(children)):
            stack.append((child, new_indent, i == 0))

def print_taxonomy_iterative_with_predicates(taxonomy):
    """
    Versión iterativa de la impresión de taxonomía para evitar RecursionError.
    Maneja ciclos y muestra predicados en las conexiones.
    """
    # 1. Identificar raíces (nodos que no son hijos de nadie)
    all_children = set(rel[0] for children in taxonomy.values() for rel in children)
    roots = [n for n in taxonomy.keys() if n not in all_children]
    
    if not roots and taxonomy:
        # Si todo es un ciclo, tomamos las llaves para empezar
        roots = list(taxonomy.keys())[:3]
        print("(!) Ciclo detectado en la raíz. Empezando por nodos arbitrarios:")

    # La pila guardará: (nodo_actual, indentación, es_el_último_hijo, predicado_padre)
    # Metemos las raíces en orden inverso para que se impriman en el orden correcto
    stack = []
    for i, root in enumerate(reversed(roots)):
        stack.append((root, "", i == 0, None))

    visited = set()

    while stack:
        node, indent, is_last, pred = stack.pop()

        # Si ya visitamos este nodo en este "camino" (o globalmente para árboles estrictos)
        # Aquí usamos un set global para evitar ciclos infinitos en el grafo
        if node in visited:
            marker = "└── " if is_last else "├── "
            label = f" [{pred}]──>" if pred else ""
            print(f"{indent}{marker}{label} {node} (RECURSIÓN/CICLO)")
            continue
        
        visited.add(node)

        # Imprimir el nodo actual
        marker = "└── " if is_last else "├── "
        if pred:
            # Si viene de un padre, mostramos el predicado
            print(f"{indent}{marker}[{pred}]──> {node}")
        else:
            # Si es raíz
            print(f"{indent}{marker}{node}")

        # Preparar hijos
        children_rels = taxonomy.get(node, [])
        new_indent = indent + ("    " if is_last else "│   ")
        
        # Añadir hijos a la pila en orden inverso
        for i, (child, p_name) in enumerate(reversed(children_rels)):
            stack.append((child, new_indent, i == 0, p_name))



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

    #display_pretty_taxonomy(taxonomy_graph)

    inverted_builder = TaxonomyConstructorInverted(language_model, beta=0.5)
    inverted_taxonomy = inverted_builder.construct_taxonomy(kb)
    for parent, children in inverted_taxonomy.items():
        print(f"Class [{parent}] children: {children}")

    # Identify root nodes (nodes that are not children of any other node)
    # all_children = set(item for sublist in inverted_taxonomy.values() for item in sublist)
    # roots = [n for n in inverted_taxonomy.keys() if n not in all_children]
    # for root in roots:
    #     print_taxonomy(inverted_taxonomy, root)

    #display_pretty_taxonomy(inverted_taxonomy)
    print_taxonomy_iterative_with_predicates(inverted_taxonomy)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taxonomy Constructor: Construct a taxonomy from a knowledge base.")
    parser.add_argument('csv_file', type=_csv_file_path, help='Knowledge base CSV file path.')
    args = parser.parse_args()

    main(args.csv_file)