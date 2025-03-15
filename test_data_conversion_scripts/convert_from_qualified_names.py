"""
This script converts qualified variable names in the test files to the regular tree
expected by GETTSIM.
"""

from _gettsim_tests import TEST_DATA_DIR
import yaml
from pathlib import Path
import flatten_dict
from _gettsim.shared import qualified_name_splitter


def collect_all_yaml_files() -> list[Path]:
    return list(TEST_DATA_DIR.glob("**/*.yaml"))


def read_one_yaml_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    

def save_to_yaml(sorted_dict: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(sorted_dict, file)


def sort_one_test_dict_alphabetically(path: Path) -> None:
    test_dict = read_one_yaml_file(path)
    sorted_dict = {}
    sorted_dict["info"] = test_dict["info"]
    sorted_dict["inputs"] = {}
    sorted_dict["inputs"]["provided"] = sort_dict(test_dict["inputs"]["provided"])
    sorted_dict["inputs"]["assumed"] = sort_dict(test_dict["inputs"]["assumed"])
    sorted_dict["outputs"] = sort_dict(test_dict["outputs"])
    save_to_yaml(sorted_dict, path)


def sort_dict(unsorted_dict: dict) -> dict:
    return dict(sorted(unsorted_dict.items()))
    

def convert_qualified_names_to_tree(path: Path) -> None:
    test_dict = read_one_yaml_file(path)
    test_dict["inputs"]["provided"] = flatten_dict.unflatten(
        test_dict["inputs"]["provided"], splitter=qualified_name_splitter
    )
    test_dict["inputs"]["assumed"] = flatten_dict.unflatten(
        test_dict["inputs"]["assumed"], splitter=qualified_name_splitter
    )
    test_dict["outputs"] = flatten_dict.unflatten(
        test_dict["outputs"], splitter=qualified_name_splitter
    )
    save_to_yaml(test_dict, path)


if __name__ == "__main__":
    for path in collect_all_yaml_files():
        sort_one_test_dict_alphabetically(path)
        convert_qualified_names_to_tree(path)
