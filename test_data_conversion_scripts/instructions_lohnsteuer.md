# Converting test data in xlsx from BMF to the YAML file in gettsim tests

## Source of the test data
The test data is sent out on a mailing list by Referat IA6 together with the new Programmablaufplan. To be added to the mailing list, best contact Steuerrechner@bmf.bund.de

## Instructions for new test data
1. Save test data for year 20AB in `gettsim-code-for-picking/original_testfaelle` with the name `lohnsteuer_bmf_20AB.xlsx`.
2. In `convert_xlsx_tests_csv_lohnst.py` replace the line `lst_data["year"] = 20CD` with the correct year 20AB.
3. Run `convert_xlsx_tests_csv_lohnst.py`. This generates the file `lohnst_converted.csv`.
4. Run `convert_cvs_tests_to_yaml.py`. This generated the test files in _gettsim_tests in the gettsim repo.
