import pandas as pd

def load_csv():
    df = pd.read_csv("sample.csv") #has some empty cells 
    df.fillna("", inplace=True)
    return df