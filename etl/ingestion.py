import pandas as pd

def load_csv():
    df = pd.read_csv("sample.csv") #has some empty cells 
    df.fillna("", inplace=True)
    print("hello...")
    print(df.head())
    return df

if __name__ == "__main__":
    load_csv()