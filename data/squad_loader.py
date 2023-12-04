"""
This is the squad_loader.py file. It is used to load the data from the json files and to create the dataset for squad2.

"""

import json
import os
import random
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from data.loader import Dataset


class SquadDataset(Dataset):
    def __init__(
        self, data: List[Dict[str, Any]], main_field: str = "question"
    ) -> None:
        self.data = data
        self.main_field = main_field

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]

    def get_column(self, column: str) -> Generator[Any, None, None]:
        index = 0
        while index < len(self.data):
            yield self.data[index].get(column)
            index += 1

    def get_base_answer(self) -> Generator[List, None, None]:
        return self.get_column("answers")

    def get_type(
        self, to_choice_mapping: Optional[Dict] = None
    ) -> Generator[str, None, None]:
        raise NotImplementedError

    @staticmethod
    def format_element(format, item):
        return format.format(context=item["context"], question=item["question"])


def prepare_squad(squad_data, get_no_answer: bool):
    reshaped_data = []
    for article in squad_data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                # exclude questions with no modified context
                if qa.get("modified") == "N/A":
                    continue
                # filter out questions with the undesired answer type
                if get_no_answer != qa["is_impossible"]:
                    continue
                reshaped_data.append({"context": paragraph["context"], **qa})
    return reshaped_data


def get_squad_dataset(
    n_examples: int,
    split_percentage: float = 0.8,
    percent_types: Tuple[float, ...] = (0.34, 0.33, 0.33),
    types: Tuple[str, ...] = (
        "answerable",
        "unanswerable_base",
        "unanswerable_ner",
    ),
    shuffle_data: bool = True,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """returns a dataset with the given number of examples and the given percent of asnwerable, base and ner (replaced named entities)

    Args:
        n_examples: number of examples to return
        percent_base_replace_which: a tuple of 3 integers representing the percent of base, replace and which (could be -1 to be defined later)

    Returns:
        a tuple of two lists, the first one is the train set and the second one is the test set
    """
    random.seed(seed)

    # check the percent_types
    count_not_defined = sum([percent for percent in percent_types if percent < 0])
    total_percent = sum(percent_types) + count_not_defined

    if total_percent > 1:
        raise Exception("The sum of the percents must be 1")
    elif total_percent < 1:
        # assign the remaining percent to the not defined uniformly
        percent_files = tuple(
            [
                percent if percent >= 0 else 1 - total_percent / count_not_defined
                for percent in percent_types
            ]
        )

    examples_per_type = [int(n_examples * percent) for percent in percent_types]
    if sum(examples_per_type) < n_examples:
        # assign the remaining examples to the biggest percent
        examples_per_type[np.argmax(percent_types)] += n_examples - sum(
            examples_per_type
        )

    squad2_ner = json.load(open("data/squad2/modified_context_NER.json", "r"))
    squad2_base = json.load(open("data/squad2/train-v2.0.json", "r"))["data"]

    datasets = {}
    datasets["unanswerable_ner"] = prepare_squad(squad2_ner, True)
    datasets["answerable"] = prepare_squad(squad2_base, False)
    datasets["unanswerable_base"] = prepare_squad(squad2_base, True)

    final_dataset = []
    for i, data_type in enumerate(types):
        data = datasets[data_type]

        if shuffle_data:
            random.shuffle(data)

        if len(data) < examples_per_type[i]:
            print(
                f"WARNING: File {data_type} has less than {examples_per_type[i]} examples"
            )

        final_dataset.extend(data[: examples_per_type[i]])

    # split the dataset into train and test
    random.shuffle(final_dataset)
    split_index = int(len(final_dataset) * split_percentage)

    return (
        SquadDataset(final_dataset[:split_index]),
        SquadDataset(final_dataset[split_index:]),
    )
