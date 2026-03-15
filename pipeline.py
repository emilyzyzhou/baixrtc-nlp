import pandas as pd
import os
import json
import regex as re
from typing import List, Optional, Tuple, Dict
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

# Keywords to exclude from analysis (generic responses)
EXCLUDED_KEYWORDS = {
    "yes", "no", "maybe", "perhaps", "sure", "okay", "ok", "alright", "fine", "good",
    "bad", "great", "excellent", "poor", "terrible", "awesome", "amazing", "horrible",
    "wonderful", "nice", "well", "very", "really", "so", "too", "quite", "pretty",
    "kind", "sort", "bit", "lot", "lots", "much", "many", "some", "any", "all",
    "none", "every", "each", "both", "either", "neither", "one", "two", "three",
    "four", "five", "first", "second", "third", "last", "next", "new", "old",
    "big", "small", "large", "little", "long", "short", "high", "low", "right",
    "wrong", "true", "false", "able", "unable", "possible", "impossible"
}

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

def extract_keywords(text: Optional[str], max_phrases: int = 5) -> Dict[str, int]:
    """
    extracting keyword phrases from text using RAKE algorithm.
    text: Input text
    max_phrases: Maximum number of phrases to extract
    returns: Dictionary mapping keywords to their frequency counts
    """
    if text is None or not isinstance(text, str):
        return {}
    
    text = str(text).strip()
    if not text or len(text) < 3:
        return {}
    
    try:
        rake = _get_rake()
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        
        keyword_dict = {}
        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) < 3:
                continue  # ignore super short junk
            
            # Skip excluded generic keywords
            phrase_lower = phrase.lower()
            if phrase_lower in EXCLUDED_KEYWORDS:
                continue
            
            # count how many times this phrase appears in the text
            count = text.lower().count(phrase.lower())
            if count > 0:
                keyword_dict[phrase] = count
            if len(keyword_dict) >= max_phrases:
                break
        
        return keyword_dict
    except Exception as e:
        global KEYWORD_EXTRACTION_ERROR_SHOWN, STOP_WORDS, LEMMATIZER
        if not KEYWORD_EXTRACTION_ERROR_SHOWN:
            print(f"[WARN] Keyword extraction failed; using fallback. Error: {e}")
            KEYWORD_EXTRACTION_ERROR_SHOWN = True

        if STOP_WORDS is None or LEMMATIZER is None:
            STOP_WORDS = set(stopwords.words("english"))
            LEMMATIZER = WordNetLemmatizer()

        cleaned = preprocess_text(text, STOP_WORDS, LEMMATIZER)
        tokens = [t for t in cleaned.split() if len(t) > 2 and t.lower() not in EXCLUDED_KEYWORDS]
        if not tokens:
            return {}
        counts = Counter(tokens)
        return dict(counts.most_common(max_phrases))


def add_keyword_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    add keywords column to df (stored as JSON with frequency counts)
    """
    print("\n[STEP 4/5] KEYWORDS - Extracting key phrases...")
    
    df_with_keywords = df.copy()
    
    # using uncleaned text for better keyword extraction
    df_with_keywords["keywords"] = df_with_keywords["response_text_original"].apply(
        lambda t: json.dumps(extract_keywords(t))
    )
    
    # count non-empty keyword extractions
    non_empty = (df_with_keywords["keywords"].str.len() > 2).sum()  # {} is 2 chars
    print(f"  extracted keywords from {non_empty:,} responses ({non_empty/len(df_with_keywords)*100:.1f}%)")
    
    return df_with_keywords


# =============================================================================
# STEP 5: EXPORT - Generate summaries and save
# =============================================================================

def generate_summary_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics grouped by question with keywords expanded to separate rows.

    To allow Tableau to better aggregate keywords, creates one row per keyword per question.

    Output format:
    - One row per question-keyword combination
    - Tableau can easily count, filter, and visualize keywords across questions
    """
    print("\n[STEP 5/5] EXPORT - generating summaries...")

    summary_data = []

    # Group by question
    grouped = df.groupby(["question_id", "question_text_original"])

    for (qid, qtext), group in grouped:
        # Calculate base question statistics (same for all keywords in this question)
        base_stats = {
            "question_id": qid,
            "question_text": qtext,
            "total_responses": len(group),
            "avg_response_length": round(group["response_length"].mean(), 2),
            "avg_word_count": round(group["word_count"].mean(), 2),
        }

        # Add sentiment statistics
        if "sentiment_score" in group.columns:
            base_stats.update({
                "avg_sentiment_score": round(group["sentiment_score"].mean(), 2),
                "sentiment_std": round(group["sentiment_score"].std(), 2),
                "positive_responses": (group["sentiment_label"] == "positive").sum(),
                "negative_responses": (group["sentiment_label"] == "negative").sum(),
                "neutral_responses": (group["sentiment_label"] == "neutral").sum(),
            })

        # Survey sources (same for all keywords in this question)
        base_stats["survey_sources"] = ", ".join(group["sheet_name"].unique())

        # Extract and aggregate keywords across all responses for this question
        all_keywords = Counter()
        for text in group["response_text_original"].dropna():
            kw_dict = extract_keywords(text)
            all_keywords.update(kw_dict)

        # Get top 10 keywords with frequencies
        top_keywords = all_keywords.most_common(10)

        # Create one row per keyword (long format for Tableau)
        if top_keywords:
            for keyword, frequency in top_keywords:
                keyword_row = base_stats.copy()
                keyword_row["keyword"] = keyword
                keyword_row["frequency"] = frequency
                summary_data.append(keyword_row)
        else:
            # If no keywords found, still create one row with empty keyword
            keyword_row = base_stats.copy()
            keyword_row["keyword"] = ""
            keyword_row["frequency"] = 0
            summary_data.append(keyword_row)

    summary_df = pd.DataFrame(summary_data)

    # Sort by question_id, then by frequency (most frequent keywords first)
    summary_df = summary_df.sort_values(["question_id", "frequency"], ascending=[True, False])

    print(f"  Generated {len(summary_df):,} keyword rows for {len(summary_df['question_id'].unique())} questions")
    print(f"  Average {len(summary_df)/len(summary_df['question_id'].unique()):.1f} keywords per question")

    return summary_df


def generate_keywords_csv(df: pd.DataFrame, output_folder: str = "data/final") -> None:
    """
    Generate a simple keywords CSV where each keyword appears as many times as its frequency.
    This creates a single-column CSV perfect for simple word cloud tools.

    Filters out generic keywords and responses from yes/no questions for more meaningful topics.
    """
    print("\n[STEP 5.5] EXPORT - generating keywords CSV...")
    
    # Filter out responses that are likely from yes/no questions
    # Exclude questions with very short average responses (likely yes/no)
    question_stats = df.groupby('question_id').agg({
        'response_length': 'mean',
        'response_text_original': 'count'
    }).reset_index()
    
    # Keep only questions with average response length > 10 characters
    meaningful_questions = question_stats[question_stats['response_length'] > 10]['question_id']
    df_filtered = df[df['question_id'].isin(meaningful_questions)]
    
    print(f"  Filtered to {len(df_filtered):,} responses from {len(meaningful_questions)} meaningful questions")
    
    # Aggregate keywords across filtered responses
    all_keywords = Counter()
    
    for text in df_filtered["response_text_original"].dropna():
        kw_dict = extract_keywords(text)
        all_keywords.update(kw_dict)
    
    # Get top 10 keywords
    top_keywords = all_keywords.most_common(10)
    
    # Create DataFrame where each keyword appears frequency times
    keyword_rows = []
    for keyword, frequency in top_keywords:
        # Add 'frequency' number of rows for this keyword
        for _ in range(frequency):
            keyword_rows.append({"keyword": keyword})

    keywords_df = pd.DataFrame(keyword_rows)

    # Save to CSV
    keywords_path = os.path.join(output_folder, "keywords.csv")
    try:
        keywords_df.to_csv(keywords_path, index=False)
        print(f"  Saved {len(keywords_df):,} keyword instances to {keywords_path}")
        print(f"  Top keywords: {', '.join([f'{kw}({freq})' for kw, freq in top_keywords])}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        keywords_path = os.path.join(output_folder, f"keywords_{ts}.csv")
        print(f"[WARN] Permission denied for keywords file. Writing to: {keywords_path}")
        keywords_df.to_csv(keywords_path, index=False)


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
    df = df.drop(columns=["keywords"])
    summary_df = generate_summary_by_question(df)
    generate_keywords_csv(df, output_folder)  # Create keywords.csv in output folder
    
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
    print(f"  3. {os.path.join(output_folder, 'keywords.csv')}")
    print(f"     → Simple keyword list for word clouds")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nReady for Tableau import!")
    print("=" * 80)
    
    return df, summary_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    responses_df, summary_df = run_pipeline()
    print("Done!")
