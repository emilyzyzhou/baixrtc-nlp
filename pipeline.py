import pandas as pd
import os
import regex as re
from typing import List, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# vaderSentiment for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("[WARN] vaderSentiment not installed. Using basic sentiment analysis.")
    print("       Install with: pip install vaderSentiment")


# =============================================================================
# CONFIG
# =============================================================================

# sheets to exclude from processing
SHEETS_TO_EXCLUDE = [
    "large company list (accumulated",
    "keywords",
    "ideal tableau format"
]

# Columns that should not be treated as text
NON_TEXT_LIKE_COLUMNS = {
    "survey_id", "question_id", "response_id", "timestamp", "sheet_name",
    "sentiment_score", "sentiment_label", "response_length", "word_count",
    "question_text_original", "response_text_original", "source_file"
}

# global vars
STOP_WORDS = None
LEMMATIZER = None
SENTIMENT_ANALYZER = None
KEYWORD_EXTRACTION_ERROR_SHOWN = False
RAKE_INSTANCE = None

# =============================================================================
# STEP 0: setup helpers
# =============================================================================

def _ensure_nltk(): #fixed this to check for all required resources at once
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            downloaded = nltk.download(resource_name, quiet=True)
            if not downloaded:
                raise RuntimeError(
                    f"NLTK resource download failed: {resource_name}. "
                    "Check your network or NLTK data path."
                )


def _get_sentiment_analyzer():
    """Get or initialize sentiment analyzer."""
    global SENTIMENT_ANALYZER
    if SENTIMENT_ANALYZER is None and VADER_AVAILABLE:
        SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
    return SENTIMENT_ANALYZER


def _get_rake():
    """Get or initialize RAKE keyword extractor."""
    global RAKE_INSTANCE
    if RAKE_INSTANCE is None:
        _ensure_nltk()
        RAKE_INSTANCE = Rake()  # default English stopwords + punkt
    return RAKE_INSTANCE


# =============================================================================
# STEP 1: INGEST - load and transform data
# =============================================================================

'''Extracts each sheet from source into df, returned as list of dfs'''
def load_and_transform_data(folder: str = "data") -> pd.DataFrame:
    """
    Load survey data and transform from wide to long format.
     - long format: each row is a single response to a single question, easier for sentiment analysis
    
    folder: path to data folder containing excel/CSV files
    returns: long-format df with standardized columns
    """
    print("\n[STEP 1/5] INGEST - loading and transforming data...")
    all_responses = []
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")
    
    response_id = 1
    
    # process all xlsx files in data folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        # skipping directories and processed files
        if os.path.isdir(file_path) or "processed" in file.lower():
            continue
        
        # process excel files
        if file.lower().endswith(".xlsx"):
            try:
                xl_file = pd.ExcelFile(file_path)
                
                for sheet_name in xl_file.sheet_names:
                    # skip excluded sheets
                    if any(excl in sheet_name.lower() for excl in SHEETS_TO_EXCLUDE):
                        print(f"  [SKIP] Excluded sheet: {sheet_name}")
                        continue
                    
                    df = pd.read_excel(xl_file, sheet_name=sheet_name)
                    
                    # update: transforming each column from wide to long format for easier sentiment analysis
                    for col in df.columns:
                        normalized_col = re.sub(r"\s+", " ", str(col)).strip().lower()
                        if normalized_col in {"sheet name", "sheetname", "sheet_name"}: #this question is metadata not a real question (skews the number of neutral responses)
                            continue
                        # skip empty columns
                        if df[col].isna().all():
                            continue
                        
                        # each non-empty cell becomes a response
                        for idx, value in df[col].items():
                            if pd.notna(value) and str(value).strip():
                                all_responses.append({
                                    'response_id': response_id,
                                    'question_text': col,
                                    'response_text': str(value).strip(),
                                    'sheet_name': sheet_name,
                                    'source_file': file
                                })
                                response_id += 1
                
                print(f"  [OK] Loaded {len(xl_file.sheet_names)} sheets from {file}")
                xl_file.close()
                
            except Exception as e:
                print(f"  [ERROR] Failed to load {file}: {e}")
        
        # process csv files
        elif file.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                
                # transform from wide to long
                for col in df.columns:
                    normalized_col = re.sub(r"\s+", " ", str(col)).strip().lower()
                    if normalized_col in {"sheet name", "sheetname", "sheet_name"}:
                        continue
                    if df[col].isna().all():
                        continue
                    
                    for idx, value in df[col].items():
                        if pd.notna(value) and str(value).strip():
                            all_responses.append({
                                'response_id': response_id,
                                'question_text': col,
                                'response_text': str(value).strip(),
                                'sheet_name': os.path.splitext(file)[0],
                                'source_file': file
                            })
                            response_id += 1
                
                print(f"  [OK] Loaded {file}")
                
            except Exception as e:
                print(f"  [ERROR] Failed to load {file}: {e}")
    
    if not all_responses:
        raise ValueError(f"No data loaded from {folder}")
    
    # create long-format dataframe
    long_df = pd.DataFrame(all_responses)
    
    # adding ids for questions and surveys
    unique_questions = long_df['question_text'].unique()
    question_id_map = {q: i+1 for i, q in enumerate(unique_questions)}
    long_df['question_id'] = long_df['question_text'].map(question_id_map)
    
    unique_sheets = long_df['sheet_name'].unique()
    survey_id_map = {s: i+1 for i, s in enumerate(unique_sheets)}
    long_df['survey_id'] = long_df['sheet_name'].map(survey_id_map)
    
    # add processing timestamp
    long_df['timestamp'] = datetime.now()
    
    # reorder columns for clarity
    long_df = long_df[[
        'response_id', 'survey_id', 'question_id', 
        'question_text', 'response_text', 
        'sheet_name', 'source_file', 'timestamp'
    ]]
    
    print(f"  Ingestion complete:")
    print(f"    • total responses: {len(long_df):,}")
    print(f"    • unique questions: {len(unique_questions)}")
    print(f"    • unique surveys/sheets: {len(unique_sheets)}")
    
    return long_df


# =============================================================================
# STEP 2: PREPROCESS - clean and normalize text
# =============================================================================

def preprocess_text(text: str, stop_words: set, lemmatizer) -> str:
    """
    - convert to lowercase
    - remove punctuation
    - remove stopwords
    - lemmatize words
    
    text: input text
    stop_words: set of stopwords to remove
    lemmatizer: NLTK lemmatizer
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # lowercase
    text = text.lower()
    
    # remove punctuation (Unicode-aware)
    text = re.sub(r'\p{P}+', ' ', text, flags=re.UNICODE)
    
    # tokenize, remove stopwords, and lemmatize
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    preprocessing all text columns in df
    """
    print("\n[STEP 2/5] PREPROCESS - Cleaning and normalizing text...")
    
    global STOP_WORDS, LEMMATIZER
    
    if STOP_WORDS is None or LEMMATIZER is None:
        _ensure_nltk()
        STOP_WORDS = set(stopwords.words("english"))
        LEMMATIZER = WordNetLemmatizer()
    
    cleaned_df = df.copy()
    
    # store original text before cleaning
    cleaned_df['question_text_original'] = cleaned_df['question_text']
    cleaned_df['response_text_original'] = cleaned_df['response_text']
    
    # store metrics before cleaning
    cleaned_df['response_length'] = cleaned_df['response_text'].str.len()
    
    # clean text
    cleaned_df['question_text'] = cleaned_df['question_text'].apply(
        lambda t: preprocess_text(t, STOP_WORDS, LEMMATIZER)
    )
    cleaned_df['response_text'] = cleaned_df['response_text'].apply(
        lambda t: preprocess_text(t, STOP_WORDS, LEMMATIZER)
    )
    
    # add word count (after cleaning)
    cleaned_df['word_count'] = cleaned_df['response_text'].str.split().str.len()
    
    print(f"  Preprocessed {len(cleaned_df):,} responses")
    print(f"    • avg response length: {cleaned_df['response_length'].mean():.1f} chars")
    print(f"    • avg word count: {cleaned_df['word_count'].mean():.1f} words")
    
    return cleaned_df


# =============================================================================
# STEP 3: SENTIMENT - analyze sentiment
# =============================================================================

def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    returns tuple of (sentiment_score, sentiment_label)
        - score: -1 (negative) to +1 (positive)
        - label: "positive", "negative", or "neutral"
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return 0.0, "neutral"
    
    if VADER_AVAILABLE:
        analyzer = _get_sentiment_analyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # classifying based on compound score
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return compound, label
    else:
        # if vader fails, use simple keyword matching
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'love', 'best', 
            'happy', 'thank', 'amazing', 'wonderful', 'helpful', 'useful'
        }
        negative_words = {
            'bad', 'terrible', 'poor', 'negative', 'hate', 'worst', 
            'sad', 'problem', 'difficult', 'hard', 'confusing', 'frustrating'
        }
        
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            return 0.5, "positive"
        elif neg_count > pos_count:
            return -0.5, "negative"
        else:
            return 0.0, "neutral"


def add_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    updates df with sentiment_score and sentiment_label columns
    """
    print("\n[STEP 3/5] SENTIMENT - analyzing sentiment...")
    
    df_with_sentiment = df.copy()
    
    # using (uncleaned) text for better sentiment analysis
    sentiment_results = df_with_sentiment["response_text_original"].apply(analyze_sentiment)
    df_with_sentiment["sentiment_score"] = sentiment_results.apply(lambda x: x[0])
    df_with_sentiment["sentiment_label"] = sentiment_results.apply(lambda x: x[1])
    
    print("  Sentiment distribution:")
    for label, count in df_with_sentiment["sentiment_label"].value_counts().items():
        pct = count / len(df_with_sentiment) * 100
        print(f"    • {label}: {count:,} ({pct:.1f}%)")
    
    return df_with_sentiment


# =============================================================================
# STEP 4: KEYWORDS - extract key phrases
# =============================================================================

def extract_keywords(text: Optional[str], max_phrases: int = 5) -> List[str]:
    """
    extracting keyword phrases from text using RAKE algorithm.
    text: Input text
    max_phrases: Maximum number of phrases to extract
    """
    if text is None or not isinstance(text, str):
        return []
    
    text = str(text).strip()
    if not text or len(text) < 3:
        return []
    
    try:
        rake = _get_rake()
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        
        keywords = []
        for p in phrases:
            p = p.strip()
            if len(p) < 3:
                continue  # ignore super short junk
            keywords.append(p)
            if len(keywords) >= max_phrases:
                break
        
        return keywords
    except Exception as e:
        global KEYWORD_EXTRACTION_ERROR_SHOWN, STOP_WORDS, LEMMATIZER
        if not KEYWORD_EXTRACTION_ERROR_SHOWN:
            print(f"[WARN] Keyword extraction failed; using fallback. Error: {e}")
            KEYWORD_EXTRACTION_ERROR_SHOWN = True

        if STOP_WORDS is None or LEMMATIZER is None:
            STOP_WORDS = set(stopwords.words("english"))
            LEMMATIZER = WordNetLemmatizer()

        cleaned = preprocess_text(text, STOP_WORDS, LEMMATIZER)
        tokens = [t for t in cleaned.split() if len(t) > 2]
        if not tokens:
            return []
        counts = Counter(tokens)
        return [w for w, _ in counts.most_common(max_phrases)]


def add_keyword_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    add keywords column to df
    """
    print("\n[STEP 4/5] KEYWORDS - Extracting key phrases...")
    
    df_with_keywords = df.copy()
    
    # using uncleaned text for better keyword extraction
    df_with_keywords["keywords"] = df_with_keywords["response_text_original"].apply(
        lambda t: ", ".join(extract_keywords(t))
    )
    
    # count non-empty keyword extractions
    non_empty = (df_with_keywords["keywords"].str.len() > 0).sum()
    print(f"  extracted keywords from {non_empty:,} responses ({non_empty/len(df_with_keywords)*100:.1f}%)")
    
    return df_with_keywords


# =============================================================================
# STEP 5: EXPORT - Generate summaries and save
# =============================================================================

def generate_summary_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    adding summary statistics grouped by question:
    - response count
    - average metrics
    - sentiment distribution
    - top keywords
    """
    print("\n[STEP 5/5] EXPORT - generating summaries...")
    
    summary_data = []
    
    # grouping by question
    grouped = df.groupby(["question_id", "question_text_original"])
    
    for (qid, qtext), group in grouped:
        summary = {
            "question_id": qid,
            "question_text": qtext,
            "total_responses": len(group),
            "avg_response_length": group["response_length"].mean(),
            "avg_word_count": group["word_count"].mean(),
        }
        
        # sentiment statistics
        if "sentiment_score" in group.columns:
            summary["avg_sentiment_score"] = group["sentiment_score"].mean()
            summary["sentiment_std"] = group["sentiment_score"].std()
            summary["positive_responses"] = (group["sentiment_label"] == "positive").sum()
            summary["negative_responses"] = (group["sentiment_label"] == "negative").sum()
            summary["neutral_responses"] = (group["sentiment_label"] == "neutral").sum()
        
        # top keywords (aggregated across all responses)
        if "keywords" in group.columns:
            all_keywords = []
            for kw_str in group["keywords"].dropna():
                if kw_str:
                    all_keywords.extend([k.strip() for k in str(kw_str).split(",")])
            
            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]
            summary["top_keywords"] = ", ".join(top_keywords[:5])
            summary["all_top_keywords"] = ", ".join(top_keywords)
        
        # survey sources
        summary["survey_sources"] = ", ".join(group["sheet_name"].unique())
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    # sort by total responses (most answered questions first)
    summary_df = summary_df.sort_values("total_responses", ascending=False)
    
    print(f"  Generated summary for {len(summary_df)} unique questions")
    
    return summary_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    data_folder: str = "data",
    output_folder: str = "data/final"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    data_folder: path to input data folder
    output_folder: path to output folder

    return: tuple of (responses_with_features_df, summary_by_question_df)
    """
    print("=" * 80)
    print("BAI SURVEY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # execute pipeline steps
    df = load_and_transform_data(data_folder)
    df = preprocess_dataframe(df)
    df = add_sentiment_analysis(df)
    df = add_keyword_extraction(df)
    summary_df = generate_summary_by_question(df)
    
    # create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # save outputs
    responses_path = os.path.join(output_folder, "responses_with_features.csv")
    summary_path = os.path.join(output_folder, "summary_by_question.csv")
    
    try:
        df.to_csv(responses_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        responses_path = os.path.join(output_folder, f"responses_with_features_{ts}.csv")
        print(f"[WARN] Permission denied for responses file. Writing to: {responses_path}")
        df.to_csv(responses_path, index=False)

    try:
        summary_df.to_csv(summary_path, index=False)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(output_folder, f"summary_by_question_{ts}.csv")
        print(f"[WARN] Permission denied for summary file. Writing to: {summary_path}")
        summary_df.to_csv(summary_path, index=False)
    
    # final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutputs saved:")
    print(f"  1. {responses_path}")
    print(f"     → {len(df):,} responses with {len(df.columns)} features")
    print(f"  2. {summary_path}")
    print(f"     → {len(summary_df)} questions with aggregated statistics")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nReady for Tableau import!")
    print("=" * 80)
    
    return df, summary_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    responses_df, summary_df = run_pipeline()
    
    # displaying sample output for now
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT - responses_with_features.csv (first 3 rows):")
    print("=" * 80)
    sample_cols = ['question_id', 'response_text_original', 'sentiment_label', 'keywords']
    print(responses_df[sample_cols].head(3).to_string())
    
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT - summary_by_question.csv (top 5 questions):")
    print("=" * 80)
    summary_cols = ['question_text', 'total_responses', 'avg_sentiment_score', 'top_keywords']
    print(summary_df[summary_cols].head(5).to_string())
