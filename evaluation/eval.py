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


def get_confusion_matrix(predict, original, unknown_value="unknown"):
    """Get the confusion matrix of the results
    Outputs a 3x2 matrix categorizing the results into correct_numerical_answer, incorrect_numerical_answer, and unknown
    Args:
        predict: The predicted results, they can be a number or unknown
        original: The original results, they can be a number or unknown
    Returns
        A 3x2 matrix of the results"""
    confusion_matrix = np.zeros((3, 2))
    none_values = 0
    for i in range(len(predict)):
        if predict[i] is None:
            none_values += 1
        elif original[i] == predict[i]:
            if predict[i] == unknown_value:
                # correct unknown
                confusion_matrix[1, 1] += 1
            else:
                # correct number
                confusion_matrix[0, 0] += 1
        elif original[i] != predict[i]:
            if predict[i] == unknown_value:
                # incorrect unknown it was a number
                confusion_matrix[1, 0] += 1
            elif original[i] == unknown_value:
                # incorrect number it was unknown
                confusion_matrix[2, 1] += 1
            else:
                # both numbers but not equal
                confusion_matrix[2, 0] += 1

    return confusion_matrix, none_values
