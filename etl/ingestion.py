import pandas as pd
import os
import string
import regex as re
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = None
LEMMATIZER = None

'''Extracts each sheet from source into df, returned as list of dfs'''
def load_data(folder = "data"):
    dfs = []
    #sheets w long names will only match substring of length ~30
    sheets_to_exclude = ["large company list (accumulated", "keywords", "ideal tableau format"]

    #loops through xlsx files in data folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        #handle xlsx files
        if file.lower().endswith(".xlsx"):
            #get sheets w data
            data = pd.ExcelFile(file_path)
            sheets_to_read = [sheet for sheet in data.sheet_names if sheet not in sheets_to_exclude]
            
            for sheet_name in sheets_to_read:
                df = pd.read_excel(data, sheet_name=sheet_name)
                df["sheet_name"] = sheet_name
                df = df[["sheet_name"] + [c for c in df.columns if c != "sheet_name"]] #sheet name 1st col
                dfs.append(df)
            data.close()
        
        #handle csv files
        elif file.lower().endswith(".csv"):
            df = pd.read_csv(file_path).fillna("")
            df["sheet_name"] = os.path.splitext(file)[0] #use file name as sheet name
            dfs.append(df)
    return dfs

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
    text = re.sub(r'\p{P}+', ' ', text, flags=re.UNICODE)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_cols: List[str] = None) -> pd.DataFrame:
    global STOP_WORDS, LEMMATIZER
    
    if STOP_WORDS is None or LEMMATIZER is None:
        _ensure_nltk()
        STOP_WORDS = set(stopwords.words("english"))
        LEMMATIZER = WordNetLemmatizer()

    if text_cols is None:
        # auto-detect likely text columns: columns containing any alphabetic character
        detected: list[str] = []
        for c in df.columns:
            # skip obviously non-text columns
            if c.lower() in {"survey_id", "question_id", "response_id", "timestamp", "sheet_name"}:
                continue
            s = df[c].astype(str)
            if s.str.contains(r"[A-Za-z]", regex=True).any():
                detected.append(c)
        text_cols = detected

    cleaned_df = df.copy()

    for col in text_cols:
        # Replace original column instead of creating new one
        cleaned_df[col] = cleaned_df[col].astype(str).apply(
            lambda t: preprocess_text(t, STOP_WORDS, LEMMATIZER)
        )

    return cleaned_df



def save_processed(df: pd.DataFrame, out_path: str = "data/processed/cleaned.csv"): #following naming conventions mentioned in task 7
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    df = load_data()
    print("Preprocessing entire dataset and saving to data/processed/cleaned.csv")
    df2 = preprocess_dataframe(df)
    save_processed(df2, os.path.join("data", "processed", "cleaned.csv"))
    print("Saved cleaned dataset to data/processed/cleaned.csv")
