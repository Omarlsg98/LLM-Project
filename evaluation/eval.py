from typing import Dict, List, Optional, Tuple

import numpy as np

from data.loader import Dataset


def evaluate_generation(
    dataset: Dataset,
    model_ouput: List[str],
) -> Tuple[List[bool], float]:
    """Evaluate the generation results against the dataset and return the accuracy"""
    results_float = []
    for result in model_ouput:
        try:
            results_float.append(float(result.replace(",", "")))
        except (ValueError, AttributeError):
            results_float.append(None)

    expected = list(dataset.get_base_answer())
    correct = np.array(expected) == np.array(results_float)
    accuracy = sum(correct) / len(correct)

    return correct, accuracy


def evaluate_classification(
    dataset: Dataset,
    model_ouput: List[str],
    to_choice_mapping: Dict[str, str],
):
    """Evaluate the classification results against the dataset and return several metrics"""
    expected = list(dataset.get_type(to_choice_mapping))
    correct = np.array(expected) == np.array(model_ouput)

    accuracy = sum(correct) / len(correct)
    return correct, accuracy
