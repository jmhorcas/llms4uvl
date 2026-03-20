import argparse
import os
import sys
import io
import csv
from pathlib import Path
from contextlib import contextmanager

from kb import UVLComparator


CSV_FILE = 'comparison_model_results.csv'
CSV_HEADER = [
    'llm',
    'model', 
    'levenshtein_distance', 
    'levenshtein_similarity_ratio', 
    'syntax_errors', 
    'semantics_errors', 
    'language_level', 
    'feature_similarity', 
    'constraint_similarity', 
    'attribute_similarity', 
    'configurations_model2', 
    'jaccard_similarity', 
    'precision', 
    'recall', 
    'f1_score', 
    'global_similarity'
]


def _uvl_path(path):
    """Accepts a .uvl file or directory."""
    if os.path.isfile(path):
        if not path.endswith('.uvl'):
            raise argparse.ArgumentTypeError(f"'{path}' is not a valid .uvl file.")
        return path
    elif os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid file or directory.")


def write_csv_row(csv_filepath: str, row: dict) -> None:
    """Escribe una fila en el CSV, creando el fichero con cabecera si no existe."""
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_uvl_files(path):
    """Returns a list of .uvl files given a file or directory."""
    if os.path.isfile(path):
        return [path]
    # It's a directory: search recursively
    uvl_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.uvl'):
                uvl_files.append(os.path.join(root, file))
    return sorted(uvl_files)


def search_original_model(uvl_filepath: str) -> str:
    "Search for the original model associated with the given filepath of the generated model."
    original_models = get_uvl_files('../resources/models/uvl_models')
    for original in original_models:
        name = Path(original).stem.lower()
        generated_name = Path(uvl_filepath).stem.lower()
        if name in generated_name:
            return original
    return None


def main(uvl_filepath: str) -> None:
    uvl_files = get_uvl_files(uvl_filepath)
    if not uvl_files:
        print("No .uvl files found.")
        exit(1)
    
    total_models = len(uvl_files)
    for i, uvl_file in enumerate(uvl_files, 1):
        print(f"{i}/{total_models} ({round(i/total_models * 100, 2)}%) Processing model {uvl_file}...")
        original_file = search_original_model(uvl_file)
        if original_file is None:
            print(f"⚠️ Skipping model {uvl_file}. Original model not found.")
            continue
        comparator = UVLComparator(original_file, uvl_file)
        results = comparator.compare()
        write_csv_row(CSV_FILE, results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UVL Model Count Errors: Count the number of syntax errors in UVL models.")
    parser.add_argument('uvl_filepath', type=_uvl_path, help='UVL model (.uvl) or directory containing .uvl files.')    
    args = parser.parse_args()

    main(args.uvl_filepath)
    