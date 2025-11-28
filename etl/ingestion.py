import pandas as pd
import os
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = None
LEMMATIZER = None

def load_data(folder = "data"):
    dfs = []
    #sheets w long names will only match substring of length ~30
    sheets_to_exclude = ["large company list (accumulated", "keywords", "ideal tableau format"]

    #loops through xlsx files in data folder
    for file in os.listdir(folder):
        if file.lower().endswith(".xlsx"):
            file_path = os.path.join(folder, file)
            #get sheets w data
            data = pd.ExcelFile(file_path)
            #print(data.sheet_names)
            sheets_to_read = [sheet for sheet in data.sheet_names if sheet not in sheets_to_exclude]
            #create df for sheet + track which sheet it came from
            for sheet_name in sheets_to_read:
                df = pd.read_excel(data, sheet_name=sheet_name).fillna("") #prevent na + type float64
                df["sheet_name"] = sheet_name
                dfs.append(df)
            data.close()

    #compile all dfs into one
    compiled_df = pd.concat(dfs, ignore_index=True)
    #reorder so sheet_name is first 
    cols = compiled_df.columns.tolist()
    cols.remove("sheet_name")
    compiled_df = compiled_df[["sheet_name"] + cols]
    #output to csv for manual checking
    compiled_df.to_csv('data/output.csv', index=False) #make sure you don't have csv open

    return compiled_df

def _ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

def preprocess_text(text: str, stop_words: set, lemmatizer) -> str: #preprocess a single text entry
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    #basic preprocessing with lowercase, splitting, stopword removal, punctuation removal, lemmatization (task 7)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    table = str.maketrans("", "", string.punctuation)
    tokens = [t.translate(table) for t in tokens]
    tokens = [t for t in tokens if t]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_cols: List[str] = None, sample_n: int = 30) -> pd.DataFrame:
    global STOP_WORDS, LEMMATIZER #reference global vars
    
    if STOP_WORDS is None or LEMMATIZER is None:
        _ensure_nltk()
        STOP_WORDS = set(stopwords.words("english"))
        LEMMATIZER = WordNetLemmatizer()

    if text_cols is None:
        text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]

    n = min(len(df), sample_n)
    working = df.iloc[:n].copy()

    for col in text_cols:
        working[f"{col}_clean"] = working[col].astype(str).apply(
            lambda t: preprocess_text(t, STOP_WORDS, LEMMATIZER)
        )

    for col in text_cols:
        new_col = f"{col}_clean"
        if new_col not in df.columns:
            df[new_col] = ""
        df.loc[:n-1, new_col] = working[new_col].values

    return df


def save_processed(df: pd.DataFrame, out_path: str = "data/processed/cleaned.csv"): #following naming conventions mentioned in task 7
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    df = load_data()
    print("Preprocessing sample rows (up to 30) and saving to data/processed/cleaned.csv")
    df2 = preprocess_dataframe(df, sample_n=30)
    save_processed(df2, os.path.join("data", "processed", "cleaned.csv"))
    print("Saved cleaned sample to data/processed/cleaned.csv")