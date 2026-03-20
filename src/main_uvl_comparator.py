import argparse
from multiprocessing import Process, Queue
from contextlib import contextmanager

from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.fm_metamodel.operations import FMLanguageLevel, MajorLevel
from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat
from flamapy.metamodels.pysat_metamodel.operations import PySATConfigurations
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3
from flamapy.metamodels.z3_metamodel.operations import Z3Configurations

from kb import KnowledgeBase, KnowledgeComparator, NaturalLanguageProcessor


CONFIGURATIONS_TIMEOUT = 60  # Timeout in seconds for computing configurations
DECIMAL_PRECISION = 4  # Decimal precision for similarity scores in the report


def _uvl_file_path(value: str) -> str:
    if not value.lower().endswith('.uvl'):
        raise argparse.ArgumentTypeError('The knowledge base file must have .uvl extension.')
    return value


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0  # Both sets are empty, consider them identical
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


def precision(set1: set, set2: set) -> float:
    if not set1:
        return 1.0 if not set2 else 0.0  # If set1 is empty, precision is 1 if set2 is also empty, otherwise 0
    intersection = set1.intersection(set2)
    return len(intersection) / len(set2) if set2 else 0.0


def recall(set1: set, set2: set) -> float:
    if not set2:
        return 1.0 if not set1 else 0.0  # If set2 is empty, recall is 1 if set1 is also empty, otherwise 0
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1) if set1 else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compare_attributes(fm_model1, fm_model2) -> tuple[float, dict, dict]:
    attributes1 = {feature.name + '.' + attr.name: attr.default_value for feature in fm_model1.get_features() for attr in feature.get_attributes()}
    attributes2 = {feature.name + '.' + attr.name: attr.default_value for feature in fm_model2.get_features() for attr in feature.get_attributes()}
    if not attributes1 and not attributes2:
        return 1.0, {}  # Both models have no attributes, consider them identical
    
    all_keys = set(attributes1.keys()).union(set(attributes2.keys()))
    score_similarity = 0
    for key in all_keys:
        value1 = attributes1.get(key, None)
        value2 = attributes2.get(key, None)
        if value1 == value2:
            score_similarity += 1

    score_similarity /= len(all_keys) if all_keys else 1

    return score_similarity, attributes1, attributes2

def compare_features(fm_model1, fm_model2) -> tuple[float, set, set]:
    features1 = set(feature.name for feature in fm_model1.get_features())
    features2 = set(feature.name for feature in fm_model2.get_features())
    similarity_score = jaccard_similarity(features1, features2)
    return similarity_score, features1, features2

def compare_constraints(fm_model1, fm_model2) -> tuple[float, set, set]:
    constraints1 = set(constraint.ast.pretty_str() for constraint in fm_model1.get_constraints())
    constraints2 = set(constraint.ast.pretty_str() for constraint in fm_model2.get_constraints())
    similarity_score = jaccard_similarity(constraints1, constraints2)
    return similarity_score, constraints1, constraints2

def execute_z3configurations(op, model, queue):
    configurations = set(op().execute(model).get_result())
    queue.put(configurations)


def compute_configurations(op, model) -> set:
    queue = Queue()
    p = Process(target=execute_z3configurations, args=(op, model, queue,))
    p.start()

    # Esperar 60 segundos
    p.join(timeout=CONFIGURATIONS_TIMEOUT)

    if p.is_alive():
        print("  ⚠️ Timeout of 60 seconds. The model is too complex to compute configurations within the time limit.")
        p.terminate()
        p.join()
        configurations = None
    else:
        configurations = queue.get()
    return configurations


def global_score(feature_similarity_score: float, constraint_similarity_score: float, attribute_similarity_score: float, jaccard_similarity_score: float) -> float:
    weights = {
        'features': 0.2,
        'constraints': 0.2,
        'attributes': 0.2,
        'configurations': 0.40
    }
    global_score = (
        feature_similarity_score * weights['features'] +
        constraint_similarity_score * weights['constraints'] +
        attribute_similarity_score * weights['attributes'] +
        jaccard_similarity_score * weights['configurations']
    )
    
    return round(global_score, DECIMAL_PRECISION)


def main(uvl_filepath1: str, uvl_filepath2: str) -> None:
    try:
        print('📥 Reading UVL model 1...')
        fm_model1 = UVLReader(uvl_filepath1).transform()
    except Exception as e:
        print(f'  ❌ Error reading UVL model 1 ({uvl_filepath1}). The model contains syntax errors: {e}')
        return

    try:
        print('📥 Reading UVL model 2...')
        fm_model2 = UVLReader(uvl_filepath2).transform()
    except Exception as e:
        print(f'  ❌ Error reading UVL model 2 ({uvl_filepath2}). The model contains syntax errors: {e}')
        return

    print('🔍 Analyzing language levels...')
    fm1_languageLevel = FMLanguageLevel().execute(fm_model1).get_result()
    minor_levels = ', '.join(level.name.capitalize().replace('_', ' ') for level in fm1_languageLevel.minors)
    minor_levels = f' ({minor_levels})' if minor_levels else ''
    fm1_languageLevel_str = f'{fm1_languageLevel.major.name.capitalize()}{minor_levels}'

    fm2_languageLevel = FMLanguageLevel().execute(fm_model2).get_result()
    minor_levels = ', '.join(level.name.capitalize().replace('_', ' ') for level in fm2_languageLevel.minors)
    minor_levels = f' ({minor_levels})' if minor_levels else ''
    fm2_languageLevel_str = f'{fm2_languageLevel.major.name.capitalize()}{minor_levels}'
    
    if fm1_languageLevel.major == MajorLevel.BOOLEAN:
        try:
            print('🔄 Transforming UVL model 1 to SAT...')
            sat_model1 = FmToPysat(fm_model1).transform()
            print(f'⚙️  Extracting configurations from UVL model 1 with timeout of {CONFIGURATIONS_TIMEOUT} seconds...')
            configurations1 = compute_configurations(PySATConfigurations, sat_model1)
        except Exception as e:
            print(f'  ❌ Error transforming UVL model 1 ({uvl_filepath1}) to SAT: {e}')
            return
    else:
        try:
            print('🔄 Transforming UVL model 1 to SMT...')
            z3_model1 = FmToZ3(fm_model1).transform()
            print(f'⚙️  Extracting configurations from UVL model 1 with timeout of {CONFIGURATIONS_TIMEOUT} seconds...')
            configurations1 = compute_configurations(Z3Configurations, z3_model1)
        except Exception as e:
            print(f'  ❌ Error transforming UVL model 1 ({uvl_filepath1}) to SMT: {e}')
            return

    if fm2_languageLevel.major == MajorLevel.BOOLEAN:
        try:
            print('🔄 Transforming UVL model 2 to SAT...')
            sat_model2 = FmToPysat(fm_model2).transform()
            print(f'⚙️  Extracting configurations from UVL model 2 with timeout of {CONFIGURATIONS_TIMEOUT} seconds...')
            configurations2 = compute_configurations(PySATConfigurations, sat_model2)
        except Exception as e:
            print(f'  ❌ Error transforming UVL model 2 ({uvl_filepath2}) to SAT: {e}')
            return
    else:
        try:
            print('🔄 Transforming UVL model 2 to SMT...')
            z3_model2 = FmToZ3(fm_model2).transform()
            print(f'⚙️  Extracting configurations from UVL model 2 with timeout of {CONFIGURATIONS_TIMEOUT} seconds...')
            configurations2 = compute_configurations(Z3Configurations, z3_model2)
        except Exception as e:
            print(f'  ❌ Error transforming UVL model 2 ({uvl_filepath2}) to SMT: {e}')
            return

    print('🔍 Comparing features...')
    feature_similarity_score, features1, features2 = compare_features(fm_model1, fm_model2)

    print('🔍 Comparing constraints...')
    constraint_similarity_score, constraints1, constraints2 = compare_constraints(fm_model1, fm_model2)

    print('🔍 Comparing attributes...')
    attribute_similarity_score, attributes1, attributes2 = compare_attributes(fm_model1, fm_model2)

    print('🔍 Comparing configurations...')
    if configurations1 is None or configurations2 is None:
        print('  ⚠️ Skipping configuration comparison due to timeout in computing configurations.')
        jaccard_similarity_score = None
        precision_score = None
        recall_score = None
        f1_score_value = None
    else:
        jaccard_similarity_score = round(jaccard_similarity(configurations1, configurations2), DECIMAL_PRECISION)
        precision_score = round(precision(configurations1, configurations2), DECIMAL_PRECISION)
        recall_score = round(recall(configurations1, configurations2), DECIMAL_PRECISION)
        f1_score_value = round(f1_score(precision_score, recall_score), DECIMAL_PRECISION)

    global_similarity_score = global_score(feature_similarity_score, constraint_similarity_score, attribute_similarity_score, jaccard_similarity_score if jaccard_similarity_score is not None else 0.0)

    print('📊 Report:')
    print(f'  - Language Level of Model 1: {fm1_languageLevel_str}')
    print(f'  - Language Level of Model 2: {fm2_languageLevel_str}')
    print(f'  - Number of features in Model 1: {len(features1)}')
    print(f'  - Number of features in Model 2: {len(features2)}')
    print(f'    - Feature Similarity: {feature_similarity_score:.4f}')
    print(f'  - Number of constraints in Model 1: {len(constraints1)}')
    print(f'  - Number of constraints in Model 2: {len(constraints2)}')
    print(f'    - Constraint Similarity: {constraint_similarity_score:.4f}') 
    print(f'  - Number of attributes in Model 1: {len(attributes1)}')
    print(f'  - Number of attributes in Model 2: {len(attributes2)}')
    print(f'    - Attribute Similarity: {attribute_similarity_score:.4f}')
    print(f'  - Number of configurations in Model 1: {len(configurations1) if configurations1 is not None else "N/A"}')
    print(f'  - Number of configurations in Model 2: {len(configurations2) if configurations2 is not None else "N/A"}')
    print(f'    - Jaccard Similarity: {jaccard_similarity_score if jaccard_similarity_score is not None else "N/A"}')
    print(f'    - Precision: {precision_score if precision_score is not None else "N/A"}')
    print(f'    - Recall: {recall_score if recall_score is not None else "N/A"}')
    print(f'    - F1 Score: {f1_score_value if f1_score_value is not None else "N/A"}')
    print(f'  - Global Similarity Score: {global_similarity_score:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UVL Model Comparator: Compare two UVL models.")
    parser.add_argument('uvl_filepath1', type=_uvl_file_path, help='First UVL model (.uvl).')
    parser.add_argument('uvl_filepath2', type=_uvl_file_path, help='Second UVL model (.uvl).')
    args = parser.parse_args()

    main(args.uvl_filepath1, args.uvl_filepath2)