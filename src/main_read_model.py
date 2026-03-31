import argparse
from flamapy.metamodels.fm_metamodel.transformations import UVLReader

from kb.uvl_file_analysis import UVLFileAnalysis


def _uvl_file_path(value: str) -> str:
    if not value.lower().endswith('.uvl'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .uvl extension.')
    return value


def main(uvl_filepath: str):
    analysis = UVLFileAnalysis(uvl_filepath)
    print(f"Number of lines in the UVL file: {analysis.number_of_lines()}")
    print(f"Number of syntax errors: {analysis.num_syntax_errors}")
    print(f"Number of non-empty lines in the UVL file: {analysis.number_of_non_empty_lines()}")
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Read a UVL model.")
    parser.add_argument('uvl_filepath', type=_uvl_file_path, help='First UVL model (.uvl).')
    args = parser.parse_args()
    main(args.uvl_filepath)