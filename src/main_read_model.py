import argparse
from flamapy.metamodels.fm_metamodel.transformations import UVLReader


def _uvl_file_path(value: str) -> str:
    if not value.lower().endswith('.uvl'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .uvl extension.')
    return value


def main(uvl_filepath: str):
    fm = UVLReader(uvl_filepath).transform()
    
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Read a UVL model.")
    parser.add_argument('uvl_filepath', type=_uvl_file_path, help='First UVL model (.uvl).')
    args = parser.parse_args()
    main(args.uvl_filepath)