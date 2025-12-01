from typing import List, Optional
import os

import pandas as pd
import nltk
from rake_nltk import Rake


def _ensure_nltk_for_rake() -> None:

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def extract_keywords(text: Optional[str], max_phrases: int = 5) -> List[str]:
    
    #extract up to `max_phrases` keyword phrases from a single text string
    

    
    if text is None:
        return []

    text = str(text).strip()
    if not text:
        return []

    
    _ensure_nltk_for_rake()

    rake = Rake()  #default English stopwords + punkt

    rake.extract_keywords_from_text(text)
    phrases = rake.get_ranked_phrases()

    keywords: List[str] = []
    for p in phrases:
        p = p.strip()
        if len(p) < 3:
            continue  #ignore super short junk
        keywords.append(p)
        if len(keywords) >= max_phrases:
            break

    return keywords




NON_TEXT_LIKE_COLUMNS = {
    "survey_id",
    "question_id",
    "response_id",
    "Timestamp",
    "sheet_name",
}


def detect_text_columns(df: pd.DataFrame) -> List[str]:
    
    #auto-detect text columns to use for keyword extraction + ignores obvious non-text columns
    detected: List[str] = []
    for c in df.columns:
        if c in NON_TEXT_LIKE_COLUMNS:
            continue
        s = df[c].astype(str)
        if s.str.contains(r"[A-Za-z]", regex=True).any():
            detected.append(c)
    return detected


def _combine_row_text(row: pd.Series, text_cols: List[str]) -> str:
    
   #combine multiple text columns into a single string for keyword extraction.
    
    parts: List[str] = []
    for c in text_cols:
        val = str(row[c]).strip()
        if val and val.lower() != "nan":
            parts.append(val)
    return " ".join(parts)


def run_keyword_extraction(
    input_path: str = os.path.join("data", "processed", "cleaned.csv"),
    output_path: str = os.path.join("data", "processed", "with_keywords.csv"),
) -> None:
    
    
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find cleaned data at {input_path}. "
            "Run ingestion/preprocessing first."
        )

    df = pd.read_csv(input_path)

    text_cols = detect_text_columns(df)
    if not text_cols:
        raise RuntimeError("No text-like columns detected for keyword extraction.")

    print(f"[INFO] Using text columns for keyword extraction: {text_cols}")

    #combine text columns into a single string per row
    df["__combined_text"] = df.apply(
        lambda row: _combine_row_text(row, text_cols), axis=1
    )

    
    df["keywords"] = df["__combined_text"].apply(
        lambda t: ", ".join(extract_keywords(t))
    )

    df.drop(columns=["__combined_text"], inplace=True)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"[OK] Saved keyword-augmented data to {output_path}")


if __name__ == "__main__":
    print("Running keyword extraction on cleaned dataset")
    run_keyword_extraction()
    print("Done. Output saved to data/processed/with_keywords.csv")
