'''Tests corresponding to ingestion.py methods - use python -m pytest to run'''
import pytest, pandas as pd
from etl.ingestion import load_data

'''Puts all output dfs into one csv to check content'''
def test_load_data():
    dfs = load_data()
    first_df = dfs[0]
    first_df.to_csv("tests/output/first_output.csv", index=False)
    #compile all dfs into one
    compiled_df = pd.concat(dfs, axis=1) #no ignore index for axis=1
    #reorder so sheet_name is first 
    cols = compiled_df.columns.tolist()
    #output to csv for manual checking
    compiled_df.to_csv("tests/output/output.csv", index=False)

    return compiled_df

if __name__ == "__main__":
    test_load_data()