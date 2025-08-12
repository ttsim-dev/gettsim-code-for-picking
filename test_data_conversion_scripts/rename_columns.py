"""
This script helps you to rename variables in GETTSIM's test files.

It is useful when you
- Rename the leaf name of a variable that is quite common in other parts of the code
base
- Want to change the namespace of a variable

To do this, this script
1. Flattens all input data dicts and expected output dicts in the test files to qnames
2. Renames the variables in the flattened dicts
3. Unflattens back to paths

After running this script, run yamlfix to fix the formatting of the test files.
"""

import re
from pathlib import Path

import dags.tree as dt
import yaml
from _gettsim_tests import TEST_DIR


def process_text_content(text: str) -> str:
    """Process text content to ensure proper formatting for YAML block scalars."""
    # Remove extra whitespace at the beginning and end
    text = text.strip()
    # Replace multiple consecutive spaces with single spaces (but preserve line breaks)
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple consecutive newlines with single newline
    text = re.sub(r"\n+", "\n", text)

    if len(text) > 80:
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= 80:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)
        return "\n".join(lines)

    return text


def collect_all_yaml_files() -> list[Path]:
    return list((TEST_DIR / "test_data").glob("**/*.yaml"))


def read_one_yaml_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def represent_str(dumper, data):
    """Custom YAML representer for strings to use block scalars for multi-line text."""
    if "\n" in data or len(data) > 80:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    else:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def save_to_yaml(sorted_dict: dict, path: Path) -> None:
    # Create a custom YAML dumper with our string representer
    class BlockScalarDumper(yaml.SafeDumper):
        pass

    BlockScalarDumper.add_representer(str, represent_str)

    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(
            sorted_dict,
            file,
            Dumper=BlockScalarDumper,
            allow_unicode=True,
            default_flow_style=False,
        )


def sort_dict(unsorted_dict: dict) -> dict:
    return dict(sorted(unsorted_dict.items()))


def rename_one_variable_in_one_yaml_file(
    path: Path, old_qname: str, new_qname: str
) -> None:
    test_dict = read_one_yaml_file(path)
    provided_inputs = test_dict["inputs"].get("provided", {})
    assumed_inputs = test_dict["inputs"].get("assumed", {})
    expected_outputs = test_dict.get("outputs", {})

    flat_provided_inputs = dt.flatten_to_qnames(provided_inputs)
    flat_assumed_inputs = dt.flatten_to_qnames(assumed_inputs)

    if old_qname in flat_provided_inputs:
        flat_provided_inputs[new_qname] = flat_provided_inputs.pop(old_qname)
    if old_qname in flat_assumed_inputs:
        flat_assumed_inputs[new_qname] = flat_assumed_inputs.pop(old_qname)
    if old_qname in expected_outputs:
        expected_outputs[new_qname] = expected_outputs.pop(old_qname)

    unflattened_provided_inputs = dt.unflatten_from_qnames(
        sort_dict(flat_provided_inputs)
    )
    unflattened_assumed_inputs = dt.unflatten_from_qnames(
        sort_dict(flat_assumed_inputs)
    )
    unflattened_expected_outputs = dt.unflatten_from_qnames(sort_dict(expected_outputs))

    out = {}
    # Process info section to ensure proper text formatting
    info = test_dict["info"].copy()
    for key, value in info.items():
        if isinstance(value, str):
            info[key] = process_text_content(value)
    out["info"] = info
    out["inputs"] = {}
    out["inputs"]["provided"] = unflattened_provided_inputs
    out["inputs"]["assumed"] = unflattened_assumed_inputs
    out["outputs"] = unflattened_expected_outputs
    save_to_yaml(out, path)


OLD_QNAME = "p_id"
NEW_QNAME = "p_id_new"


if __name__ == "__main__":
    _all_yaml_files = collect_all_yaml_files()
    for path in _all_yaml_files:
        if path.name.startswith("skip_"):
            # Skip draft tests that are not properly formatted
            continue
        rename_one_variable_in_one_yaml_file(
            path=path,
            old_qname=OLD_QNAME,
            new_qname=NEW_QNAME,
        )
