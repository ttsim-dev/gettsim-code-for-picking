# Converting test data in xlsx from BMF to the YAML file in gettsim tests

## Source of the test data
The test data is sent out on a mailing list by Referat IA6 together with the new Programmablaufplan. To be added to the mailing list, best contact Steuerrechner@bmf.bund.de

## Instructions for new test data
1. Save test data for year `year` in `\gettsim-code-for-picking\original_testfaelle` with the name `lohnsteuer_bmf_`year`.xlsx`.
2. In `convert_lohn_st.py` replace the line `lst_data["year"] = `year`` with the correct year.
3. Run `convert_lohn_st.py`. This generates the file `lohn_st_converted.csv`.
4. Run `convert_cvs_tests_to_yamk.py`. This generated the test files in _gettsim_tests in the gettsim repo.