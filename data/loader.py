import json
import os
import random
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np


class Dataset:
    def __init__(
        self, data: List[Dict[str, Any]], main_field: str = "question"
    ) -> None:
        self.data = data
        self.main_field = main_field

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index][self.main_field]

    def get_base_answer(self) -> Generator[int, None, None]:
        index = 0
        while index < len(self.data):
            yield self.data[index]["base answer"]
            index += 1

    def get_type(
        self, to_choice_mapping: Optional[Dict] = None
    ) -> Generator[str, None, None]:
        index = 0
        while index < len(self.data):
            if to_choice_mapping is not None:
                yield to_choice_mapping[self.data[index]["type"]]
            else:
                yield self.data[index]["type"]
            index += 1


def load_file(file_name: str):
    # read the json records from data/gsm8k folder
    full_path = os.path.join("data", "gsm8k", file_name)
    with open(full_path, "r") as f:
        data = json.load(f)
    return data


def get_dataset(
    n_examples: int,
    split_percentage: float = 0.8,
    percent_files: Tuple[float, ...] = (0.34, 0.33, 0.33),
    file_names: Tuple[str, ...] = (
        "base.json",
        "replace.json",
        "which.json",
    ),
    shuffle_data: bool = True,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """returns a dataset with the given number of examples and the given percent of base, replace and which

    Args:
        n_examples: number of examples to return
        percent_base_replace_which: a tuple of 3 integers representing the percent of base, replace and which (could be -1 to be defined later)

    Returns:
        a tuple of two lists, the first one is the train set and the second one is the test set
    """
    random.seed(seed)

    # check the percent_files
    count_not_defined = sum(
        [percent for percent in percent_files if percent < 0]
    )
    total_percent = sum(percent_files) + count_not_defined

    if total_percent > 1:
        raise Exception("The sum of the percents must be 1")
    elif total_percent < 1:
        # assign the remaining percent to the not defined uniformly
        percent_files = tuple(
            [
                percent
                if percent >= 0
                else 1 - total_percent / count_not_defined
                for percent in percent_files
            ]
        )

    examples_per_file = [
        int(n_examples * percent) for percent in percent_files
    ]
    if sum(examples_per_file) < n_examples:
        # assign the remaining examples to the biggest percent
        examples_per_file[np.argmax(percent_files)] += n_examples - sum(
            examples_per_file
        )

    dataset = []
    for i, file_name in enumerate(file_names):
        data = load_file(file_name)
        if shuffle_data:
            random.shuffle(data)

        if len(data) < examples_per_file[i]:
            print(
                f"WARNING: File {file_name} has less than {examples_per_file[i]} examples"
            )

        # add file_origin field to the data
        for j in range(len(data)):
            data[j]["base answer"] = float(
                data[j]["base answer"].replace(",", "")
            )
            data[j]["file_origin"] = file_name

        dataset.extend(data[: examples_per_file[i]])

    # split the dataset into train and test
    random.shuffle(dataset)
    split_index = int(len(dataset) * split_percentage)

    return Dataset(dataset[:split_index]), Dataset(dataset[split_index:])
