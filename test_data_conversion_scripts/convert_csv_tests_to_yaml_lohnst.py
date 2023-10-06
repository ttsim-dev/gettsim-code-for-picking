from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
import yaml
from _gettsim_tests import TEST_DATA_DIR

from pathlib import Path

note_columns = [
    "note",
    "Note",
    "notes",
    "comment",
    "Comment",
    "Notes on Entgeltpunkte",
    "Notes on Regelaltersgrenze",
]
source_columns = ["source", "Source", "Quelle Arbeitgeber"]

roles = {
    "lohnst": {
        "in_provided": [
            "hh_id",
            "tu_id",
            "p_id",
            "wohnort_ost",
            "steuerklasse",
            "bruttolohn_m",
            "alter",
            "hat_kinder",
            "arbeitsstunden_w",
            "in_ausbildung",
            "ges_krankenv_zusatzbeitr_satz",
            "ges_pflegev_zusatz_kinderlos",
            "regulär_beschäftigt",
        ],
        "in_assumed": [
        ],
        "out": [
            "lohnst_m",
            "soli_st_lohnst_m",
        ],
    },
}


def list_csv_files() -> list[Path]:
    test_data_list = [
        *list(TEST_DATA_DIR.glob("*.csv")),
        *list((Path(__file__).parent.parent / "original_testfaelle").glob("*.csv")),
    ]
    return test_data_list


def read_file(path_: Path) -> pd.DataFrame:
    return (
        pd.read_csv(TEST_DATA_DIR / path_, header=0, index_col=0, encoding="utf-8")
        .squeeze("columns")
        .reset_index()
    )


def unique_years(df: pd.DataFrame, column_name: str = "jahr") -> list[int]:
    return sorted(df[column_name].unique())


def grouped_by_year(
    df: pd.DataFrame, column_name: str = "jahr"
) -> dict[int, pd.DataFrame]:
    return {year: df[df[column_name] == year] for year in unique_years(df, column_name)}


def columns_by_role(
    df: pd.DataFrame, name: str
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    out_cols = roles[name]["out"] if name in roles and "out" in roles[name] else []
    in_cols_assumed = (
        roles[name]["in_assumed"]
        if name in roles and "in_assumed" in roles[name]
        else []
    )
    in_cols_provided = (
        roles[name]["in_provided"]
        if name in roles and "in_provided" in roles[name]
        else []
    )
    note_cols = [col for col in df if col in note_columns]
    source_cols = [col for col in df if col in source_columns]

    return in_cols_provided, in_cols_assumed, out_cols, note_cols, source_cols


def create_yaml(df: pd.DataFrame, name: str) -> dict[str, dict]:
    (
        in_cols_provided,
        in_cols_assumed,
        out_cols,
        note_cols,
        source_cols,
    ) = columns_by_role(df, name)

    df.replace(to_replace=np.nan, value=None, inplace=True)

    out = {}

    def df_to_dict(df: pd.DataFrame) -> dict:
        source = "\n\n".join(
            value_to_string(df[source_column].iloc[0])
            for source_column in source_cols
            if value_to_string(df[source_column].iloc[0]) != ""
        )
        note = "\n\n".join(
            value_to_string(df[note_column].iloc[0])
            for note_column in note_cols
            if value_to_string(df[note_column].iloc[0]) != ""
        )
        specs = {"note": note, "source": source}
        inputs = {
            "provided": df[in_cols_provided].to_dict("list"),
            "assumed": df[in_cols_assumed].to_dict("list"),
        }
        outputs = df[out_cols].to_dict("list")
        return {"info": specs, "inputs": inputs, "outputs": outputs}

    if "hh_id" in df:
        for hh_id in sorted(df["hh_id"].unique()):
            df_hh = df.loc[df["hh_id"] == hh_id]
            out[f"hh_id_{hh_id}"] = df_to_dict(df_hh)
    else:
        out["hh_id_unknown"] = df_to_dict(df)
    return out


def value_to_string(value: Any) -> str:
    if pd.isnull(value):
        return ""
    else:
        return str(value)


def write_yaml_to_file(
    out: dict[str, dict], name: str, year: Optional[int] = None
) -> None:
    for hh_id, hh_dict in out.items():
        text = yaml.dump(hh_dict, sort_keys=False, allow_unicode=True, indent=2, width=88)
        if year is None:
            path = TEST_DATA_DIR / name / f"{hh_id}.yaml"
        else:
            path = TEST_DATA_DIR / name / str(year) / f"{hh_id}.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing to {path}")
        with open(path, "w", encoding="utf-8") as text_file:
            text_file.write(text)


def convert_test_data() -> None:
    for path_ in list_csv_files():
        df = read_file(path_)
        name = path_.stem

        if "jahr" not in df:
            yaml_out = create_yaml(df, name)
            write_yaml_to_file(yaml_out, name)
        else:
            for year, year_df in grouped_by_year(df).items():
                yaml_out = create_yaml(year_df, name)
                write_yaml_to_file(yaml_out, name, year)


if __name__ == "__main__":
    for path in list_csv_files():
        print(f'"{path.stem}": {"{}"},')  # noqa: T201
    convert_test_data()
