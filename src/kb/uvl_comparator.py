import io
import sys
import re
import logging
from pathlib import Path
from contextlib import contextmanager
from multiprocessing import Process, Queue

import Levenshtein

from flamapy.metamodels.fm_metamodel.models import FeatureModel
from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.fm_metamodel.operations import FMLanguageLevel, MajorLevel
from flamapy.metamodels.pysat_metamodel.transformations import FmToPysat
from flamapy.metamodels.pysat_metamodel.operations import PySATConfigurations
from flamapy.metamodels.z3_metamodel.transformations import FmToZ3
from flamapy.metamodels.z3_metamodel.operations import Z3Configurations


CONFIGURATIONS_TIMEOUT = 60  # Timeout in seconds for computing configurations
DECIMAL_PRECISION = 4  # Decimal precision for similarity scores in the report


class StderrErrorCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.error_lines = []

    def emit(self, record):
        self.error_lines.append(self.format(record))

    def count(self) -> int:
        return len(self.error_lines)

    def reset(self):
        self.error_lines = []


class UVLComparator:

    def __init__(self, original_model: str, generated_model: str) -> None:
        self.uvl_path1 = original_model
        self.uvl_path2 = generated_model
        self._error_capture = StderrErrorCapture()
        logging.getLogger().addHandler(self._error_capture)

    def _read_model(self, uvl_path: str) -> tuple[FeatureModel, int]:
        """Reads a UVL file and returns the feature model and the number of syntax errors."""
        self._error_capture.reset()
        fm_model = None
        try:
            fm_model = UVLReader(uvl_path).transform()
        except Exception:
            pass
        
        return fm_model, self._error_capture.count()

    def compare(self) -> None:
        model1, syntax_errors1 = self._read_model(self.uvl_path1)
        model2, syntax_errors2 = self._read_model(self.uvl_path2)
        levenshtein_metrics = get_normalized_metrics(self.uvl_path1, self.uvl_path2)
        levenshtein_distance = levenshtein_metrics['distance']
        levenshtein_similarity_ratio = round(levenshtein_metrics['similarity_ratio'], 2)
        path2 = Path(self.uvl_path2)
        llm = [x for x in ['GPT', 'Claude', 'Deepseek', 'Gemini'] if x in self.uvl_path2][0]
        if model1 is None or model2 is None:
            results = {
                'llm': llm,
                'model': path2.stem,
                'levenshtein_distance': levenshtein_distance,
                'levenshtein_similarity_ratio': levenshtein_similarity_ratio,
                'syntax_errors': syntax_errors2,
                'semantics_errors': None,
                'language_level': None,
                'feature_similarity': None,
                'constraint_similarity': None,
                'attribute_similarity': None,
                'configurations_model2': None,
                'jaccard_similarity': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'global_similarity': None,
            }
            return results
        else:
            language_level1 = get_language_level(model1)
            language_level2 = get_language_level(model2)
            feature_similarity_score, features1, features2 = compare_features(model1, model2)
            constraint_similarity_score, constraints1, constraints2 = compare_constraints(model1, model2)
            attribute_similarity_score, attributes1, attributes2 = compare_attributes(model1, model2)
            configurations1 = get_configurations(model1)
            configurations2 = get_configurations(model2)
            if configurations1 is None or configurations2 is None:
                semantics_errors1 = 1
                semantics_errors2 = 1
                jaccard_similarity_score = None
                precision_score = None
                recall_score = None
                f1_score_value = None
            else:
                semantics_errors1 = 0
                semantics_errors2 = 0
                jaccard_similarity_score = round(jaccard_similarity(configurations1, configurations2), DECIMAL_PRECISION)
                precision_score = round(precision(configurations1, configurations2), DECIMAL_PRECISION)
                recall_score = round(recall(configurations1, configurations2), DECIMAL_PRECISION)
                f1_score_value = round(f1_score(precision_score, recall_score), DECIMAL_PRECISION)
                global_similarity_score = global_score(feature_similarity_score, constraint_similarity_score, attribute_similarity_score, jaccard_similarity_score if jaccard_similarity_score is not None else 0.0)
                num_configurations2 = len(configurations2) if configurations2 is not None else 0
                if configurations2 is not None and num_configurations2 >= 1e6:
                    num_configurations2 = int_to_scientific_notation(len(configurations2))
        results = {
                'llm': llm,
                'model': path2.stem,
                'levenshtein_distance': levenshtein_distance,
                'levenshtein_similarity_ratio': levenshtein_similarity_ratio,
                'syntax_errors': syntax_errors2,
                'semantics_errors': semantics_errors2,
                'language_level': language_level2,
                'feature_similarity': round(feature_similarity_score, DECIMAL_PRECISION),
                'constraint_similarity': round(constraint_similarity_score, DECIMAL_PRECISION),
                'attribute_similarity': round(attribute_similarity_score, DECIMAL_PRECISION),
                'configurations_model2': num_configurations2,
                'jaccard_similarity': jaccard_similarity_score,
                'precision': precision_score,
                'recall': recall_score,
                'f1_score': f1_score_value,
                'global_similarity': global_similarity_score,
        }
        return results

    
def get_language_level(fm: FeatureModel) -> str:
    language_level = _get_language_level(fm)
    minor_levels = ', '.join(level.name.capitalize().replace('_', ' ') for level in language_level.minors)
    minor_levels = f' ({minor_levels})' if minor_levels else ''
    return f'{language_level.major.name.capitalize()}{minor_levels}'


def _get_language_level(fm: FeatureModel) -> FMLanguageLevel:
        return FMLanguageLevel().execute(fm).get_result()


def get_configurations(fm: FeatureModel) -> set | None:
    if fm is None:
        return None
    language_level = _get_language_level(fm)
    configurations = None
    if language_level.major == MajorLevel.BOOLEAN:
        try:
            print('🔄 Transforming UVL model to SAT...')
            sat_model = FmToPysat(fm).transform()
            print(f'⚙️  Extracting configurations from UVL model with timeout of {CONFIGURATIONS_TIMEOUT} seconds...')
            configurations = compute_configurations(PySATConfigurations, sat_model, CONFIGURATIONS_TIMEOUT)
        except Exception as e:
            print(f'  ❌ Error transforming UVL model to SAT: {e}')
            return None
    else:
        try:
            print('🔄 Transforming UVL model to SMT...')
            z3_model = FmToZ3(fm).transform()
            print(f'⚙️  Extracting configurations from UVL model with timeout of {CONFIGURATIONS_TIMEOUT} seconds...')
            configurations = compute_configurations(Z3Configurations, z3_model, CONFIGURATIONS_TIMEOUT)
        except Exception as e:
            print(f'  ❌ Error transforming UVL model to SMT: {e}')
            return None
    return configurations


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


def execute_configurations_operation(op, model, queue):
    configurations = set(op().execute(model).get_result())
    queue.put(configurations)


def compute_configurations(op, model, timeout) -> set:
    queue = Queue()
    p = Process(target=execute_configurations_operation, args=(op, model, queue,))
    p.start()

    # Esperar 60 segundos
    p.join(timeout=timeout)

    if p.is_alive():
        print(f"  ⚠️ Timeout of {timeout} seconds. The model is too complex to compute configurations within the time limit.")
        p.terminate()
        p.join()
        configurations = None
    else:
        configurations = queue.get()
    return configurations


def compare_attributes(fm_model1, fm_model2) -> tuple[float, dict, dict]:
    attributes1 = {feature.name + '.' + attr.name: attr.default_value for feature in fm_model1.get_features() for attr in feature.get_attributes()}
    attributes2 = {feature.name + '.' + attr.name: attr.default_value for feature in fm_model2.get_features() for attr in feature.get_attributes()}
    if not attributes1 and not attributes2:
        return 1.0, {}, {}  # Both models have no attributes, consider them identical
    
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


def get_clean_uvl_content(text):
    """
    Limpia el contenido de un modelo UVL para evaluación:
    1. Elimina comentarios multilínea /* ... */
    2. Elimina comentarios de línea //
    3. Elimina líneas en blanco y espacios innecesarios
    """
    # 1. Eliminar comentarios multilínea: /* cualquier cosa */
    # re.DOTALL permite que el '.' incluya saltos de línea
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # 2. Eliminar comentarios de una sola línea: // resto de la línea
    text = re.sub(r'//.*', '', text)
    
    # 3. Eliminar líneas en blanco y espacios en blanco en los extremos
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    return "\n".join(lines)


def get_normalized_metrics(original_filepath, generated_filepath):
    """
    Calcula la distancia de edición sobre el contenido puro (sin ruido visual).
    """
    original_model = read_file(original_filepath)
    generate_model = read_file(generated_filepath)
    clean_gt = get_clean_uvl_content(original_model)
    clean_gen = get_clean_uvl_content(generate_model)

    distance = Levenshtein.distance(clean_gt, clean_gen)
    ratio = Levenshtein.ratio(clean_gt, clean_gen)
    
    return {
        "distance": distance,
        "similarity_ratio": ratio,
        "clean_gen": clean_gen
    }

def read_file(model_path: str) -> str:
    with open(model_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def int_to_scientific_notation(n: int, precision: int = 2) -> str:
    """Convert a large int into scientific notation.
    
    It is required for large numbers that Python cannot convert to float,
    solving the error `OverflowError: int too large to convert to float`.
    """
    str_n = str(n)
    decimal = str_n[1:precision+1]
    exponent = str(len(str_n) - 1)
    return str_n[0] + '.' + decimal + 'e' + exponent