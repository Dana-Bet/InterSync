import os
import pandas as pd
import openpyxl

from data.target import Target


def target_vectorframe_data_export_to_excel(target_for_export: Target, export_path: str, csv=False):

    df = target_for_export.get_vector_data_as_dataframe()

    # check if the DataFrame is empty
    if df.empty or df.shape[1] == 0:
        print(f"target_vectorframe_data_export_to_excel: DataFrame for target {target_for_export.get_id()} is empty or has no columns.")
        return

    # verification to make sure column count matches data count
    expected_columns = len(df.columns)
    actual_columns = df.shape[1]

    if expected_columns != actual_columns:
        print(f"target_vectorframe_data_export_to_excel: mismatch between column names ({expected_columns}) and data columns ({actual_columns}) for target {target_for_export.get_id()}.")
        return

    if not csv:
        filename = f'target_id_{target_for_export.get_id()}_vectorframe_data.xlsx'
        output_file = os.path.join(export_path, filename)
        df.to_excel(output_file, index=False)
    else:
        filename = f'target_id_{target_for_export.get_id()}_vectorframe_data.csv'
        output_file = os.path.join(export_path, filename)
        df.to_csv(output_file, index=False)
    print(f"data exported successfully to {output_file}")
