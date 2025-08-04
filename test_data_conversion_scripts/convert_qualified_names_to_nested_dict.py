"""
This script converts qualified variable names in the test files to the regular tree
expected by GETTSIM.
"""

from pathlib import Path

import dags.tree as dt
import yaml
from _gettsim_tests import TEST_DIR


def collect_all_yaml_files() -> list[Path]:
    return list((TEST_DIR / "test_data").glob("**/*.yaml"))


def read_one_yaml_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_to_yaml(sorted_dict: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(sorted_dict, file, allow_unicode=True)


def sort_one_test_dict_alphabetically(path: Path) -> None:
    test_dict = read_one_yaml_file(path)

    assumed_inputs = test_dict["inputs"].get("assumed", {})
    provided_inputs = test_dict["inputs"].get("provided", {})

    sorted_dict = {}
    sorted_dict["info"] = test_dict["info"]
    sorted_dict["inputs"] = {}
    if provided_inputs:
        sorted_dict["inputs"]["provided"] = sort_dict(provided_inputs)
    else:
        sorted_dict["inputs"]["provided"] = {}
    if assumed_inputs:
        sorted_dict["inputs"]["assumed"] = sort_dict(assumed_inputs)
    else:
        sorted_dict["inputs"]["assumed"] = {}
    sorted_dict["outputs"] = sort_dict(test_dict["outputs"])
    save_to_yaml(sorted_dict, path)


def sort_dict(unsorted_dict: dict) -> dict:
    return dict(sorted(unsorted_dict.items()))


def convert_qualified_names_to_tree(path: Path) -> None:
    test_dict = read_one_yaml_file(path)
    provided_inputs = test_dict["inputs"].get("provided", {})
    assumed_inputs = test_dict["inputs"].get("assumed", {})

    unflattened_dict = {}
    unflattened_dict["inputs"] = {}
    unflattened_dict["outputs"] = {}
    if provided_inputs:
        unflattened_dict["inputs"]["provided"] = dt.unflatten_from_qual_names(
            provided_inputs
        )
    else:
        unflattened_dict["inputs"]["provided"] = {}
    if assumed_inputs:
        unflattened_dict["inputs"]["assumed"] = dt.unflatten_from_qual_names(
            assumed_inputs
        )
    else:
        unflattened_dict["inputs"]["assumed"] = {}

    unflattened_dict["outputs"] = dt.unflatten_from_qual_names(test_dict["outputs"])
    save_to_yaml(unflattened_dict, path)


if __name__ == "__main__":
    _all_yaml_files = collect_all_yaml_files()
    for path in _all_yaml_files:
        if path.name.startswith("skip_"):
            # Skip draft tests that are not properly formatted
            continue
        sort_one_test_dict_alphabetically(path)
        convert_qualified_names_to_tree(path)
