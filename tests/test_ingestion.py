import pytest, pandas
from etl/ingestion.py import load_data

'''Puts all output dfs into one csv to check content'''
def test_load_data:
    dfs = load_data()

    #compile all dfs into one
    compiled_df = pd.concat(dfs, ignore_index=True)
    #reorder so sheet_name is first 
    cols = compiled_df.columns.tolist()
    cols.remove("sheet_name")
    compiled_df = compiled_df[["sheet_name"] + cols]
    #output to csv for manual checking
    compiled_df.to_csv('data/output.csv', index=False) #make sure you don't have csv open

    return compiled_df