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

# Suppress Hugging Face Hub warnings
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

SEED_KEYWORDS = {
    
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


def load_rtc_keywords_from_excel(
    file_path: str,
    sheet_name: str = "keywords",
    first_col_index: int = 0
) -> List[str]:
    """
    grabbing RTC keyword list from the "keywords" sheet (first column).
    returns a cleaned, deduplicated list of keywords (og_rtc_kw).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Keyword file not found: {file_path}")

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    if first_col_index not in df.columns:
        raise ValueError(f"Column index {first_col_index} not found in sheet {sheet_name}")

    # split multi-keyword cells by comma and/or semicolon + whitespace
    og_rtc_kw = []
    for raw_cell in df.iloc[:, first_col_index].dropna().astype(str):
        raw_cell = raw_cell.strip()
        if not raw_cell:
            continue

        parts = re.split(r",\s*|;\s*", raw_cell)
        for part in parts:
            kw = part.strip()
            if kw:
                og_rtc_kw.append(kw)

    # maintaining unique order (case insensitive dedupe)
    seen = set()
    og_rtc_kw_unique = []
    for kw in og_rtc_kw:
        norm_kw = kw.lower()
        if norm_kw not in seen:
            seen.add(norm_kw)
            og_rtc_kw_unique.append(kw)
    og_rtc_kw = og_rtc_kw_unique

    return og_rtc_kw


def find_present_rtc_keywords(
    df: pd.DataFrame,
    rtc_keywords: List[str],
    text_column: str = "response_text_original"
) -> List[str]:
    """return keywords from rtc_keywords that appear in the data text column"""
    if text_column not in df.columns:
        raise ValueError(f"Dataframe does not have expected column: {text_column}")

    text_series = df[text_column].dropna().astype(str).str.lower()

    normalized_kw = [kw.strip().lower() for kw in rtc_keywords if kw.strip()]
    present_rtc_kw = []

    for kw in normalized_kw:
        if not kw:
            continue
        pattern = r"\b" + re.escape(kw) + r"\b"
        if text_series.str.contains(pattern, regex=True).any():
            present_rtc_kw.append(kw)

    return present_rtc_kw


# =============================================================================
# STEP 2: PREPROCESS - clean and normalize text (need to skip now since sentiment analysis + keyword extraction needs more to work with)
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

def get_seed_keywords():
    """
    Returns a set of seed keywords to prioritize in extraction.
    This can be expanded based on domain knowledge or common themes in the data.
    """
    
    return SEED_KEYWORDS   

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
    
    # need to be using nltk stopwords + lemmatizer here since the keyword extraction relies on them for cleaner results
    try:
        rake = _get_rake()
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        
        keyword_dict = {}
        for phrase in phrases:
            phrase = phrase.strip()
            # if len(phrase) < 3:
            #     continue 
            
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
# STEP 5: TOPIC MODELING - Extract and infer topics
# =============================================================================

def load_provided_topics(
    file_path: str,
    sheet_name: str = "keywords"
) -> List[str]:
    """
    Load provided topics from the keywords sheet (first column).
    Uses load_rtc_keywords_from_excel() which handles comma-separated keywords.

    file_path: path to Excel file containing keywords sheet
    sheet_name: sheet name containing keywords data
    returns: deduplicated list of provided topics (individual keywords/phrases)
    """
    if not os.path.exists(file_path):
        print(f"[WARN] Keywords file not found: {file_path}. Skipping provided topics.")
        return []

    try:
        # Use existing load_rtc_keywords_from_excel with first column (index 0)
        provided_topics = load_rtc_keywords_from_excel(file_path, sheet_name, first_col_index=0)
        print(f"  Loaded {len(provided_topics)} provided topics from keywords sheet")
        return provided_topics
    except Exception as e:
        print(f"[WARN] Error loading provided topics: {e}")
        return []


def perform_topic_modeling(
    texts: List[str],
    num_topics: int = 5,
    min_topic_size: int = 2,
    language: str = "english",
    max_texts: int = None
) -> Tuple:
    """
    Perform topic modeling using TF-IDF (replaces BERTopic for better results on short responses).

    texts: List of text documents (responses)
    num_topics: Target number of topics to infer (returns num_topics * 3 individual terms)
    min_topic_size: Unused (kept for backward compatibility)
    language: Language for stopwords
    max_texts: Unused (kept for backward compatibility; processes all texts)
    returns: Tuple of (None, inferred_topics, None) where inferred_topics is a list of individual keywords
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        print("[INFO] Using TF-IDF for topic modeling...")

        # Filter out empty texts
        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]

        if len(valid_texts) < 2:
            print(f"[WARN] Not enough text samples ({len(valid_texts)}) for topic modeling. Skipping.")
            return None, [], None

        print(f"  Fitting TF-IDF on {len(valid_texts):,} texts...")

        # Initialize TF-IDF vectorizer with specified parameters
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.85,
            stop_words='english'
        )

        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(valid_texts)

        # Sum TF-IDF scores across all documents to rank terms by global importance
        tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Convert to 1D array

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Create list of (term, score) tuples and sort by score
        term_scores = list(zip(feature_names, tfidf_scores))
        term_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter terms: remove <5 chars, stopwords, excluded keywords, numeric-only
        # Also remove terms appearing in >30% of responses (too common/filler)
        filtered_terms = []
        nltk_stopwords = set(stopwords.words("english")) if STOP_WORDS is None else STOP_WORDS

        # Calculate appearance threshold (30% of documents)
        appearance_threshold = len(valid_texts) * 0.30

        for term, score in term_scores:
            # Skip if too short (5+ character requirement)
            if len(term) < 5:
                continue
            # Skip if in stopwords or excluded keywords
            if term.lower() in nltk_stopwords or term.lower() in EXCLUDED_KEYWORDS:
                continue
            # Skip if purely numeric
            if term.replace(' ', '').isdigit():
                continue

            # Count how many documents contain this term
            term_appears = sum(1 for text in valid_texts if term.lower() in text.lower())
            # Skip if too common (appears in >60% of responses)
            if term_appears > appearance_threshold:
                continue

            filtered_terms.append(term)

        # Return top num_topics * 3 terms as individual topics
        target_count = num_topics * 3
        inferred_topics = filtered_terms[:target_count]

        print(f"  Topic modeling complete. Found {len(inferred_topics)} inferred topics.")

        return None, inferred_topics, None

    except Exception as e:
        print(f"[ERROR] Topic modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return None, [], None


def extract_inferred_topics(
    inferred_topics: List[str],
    num_topics: int = 5
) -> List[str]:
    """
    Extract individual inferred topics (no longer using BERTopic model).
    Each topic string is returned as-is (no grouping, no prefix).

    inferred_topics: List of inferred topic strings from TF-IDF
    num_topics: Unused (kept for backward compatibility)
    returns: List of topic strings (already individual, no prefixes)
    """
    if inferred_topics is None or not inferred_topics:
        return []

    # Topics are already individual strings from perform_topic_modeling
    # No further processing needed
    return inferred_topics


def match_responses_to_topics(
    df: pd.DataFrame,
    provided_topics: List[str],
    inferred_topics: List[str],
    model=None,
    topics_array=None,
    probabilities=None
) -> pd.DataFrame:
    """
    Match each response to relevant topics using keyword presence (both provided and inferred).
    Deduplicates topics case-insensitively (provided topics take priority).

    df: DataFrame with responses
    provided_topics: List of provided topics (individual keywords/phrases)
    inferred_topics: List of inferred topics (individual keywords/phrases)
    model: Unused (kept for backward compatibility)
    topics_array: Unused (kept for backward compatibility)
    probabilities: Unused (kept for backward compatibility)
    returns: DataFrame with topic assignments for each response
    """
    df_topics = df.copy()

    # Deduplicate topics case-insensitively (provided topics take priority)
    all_topics = list(provided_topics)
    seen_lower = {t.lower() for t in provided_topics}

    for topic in inferred_topics:
        if topic.lower() not in seen_lower:
            all_topics.append(topic)
            seen_lower.add(topic.lower())

    # Initialize topic columns
    for topic in all_topics:
        df_topics[topic] = ""

    # Process each response
    for idx, row in df_topics.iterrows():
        text = str(row['response_text_original']).lower() if pd.notna(row['response_text_original']) else ""

        if not text:
            continue

        # Match all topics using word boundary regex matching
        for topic in all_topics:
            # Use word boundary matching for each topic term
            pattern = r"\b" + re.escape(topic.lower()) + r"\b"
            if re.search(pattern, text):
                df_topics.at[idx, topic] = row['response_id']

    return df_topics


# =============================================================================
# STEP 5.5: EXPORT - Additional topic-based outputs
# =============================================================================

def generate_topics_csv(
    df: pd.DataFrame,
    provided_topics: List[str],
    inferred_topics: List[str],
    output_folder: str = "data/final"
) -> None:
    """
    Generate topics.csv where each column is a topic and contains
    all response_ids that relate to that topic.
    
    df: DataFrame with topic assignments (from match_responses_to_topics)
    provided_topics: List of provided topics
    inferred_topics: List of inferred topics
    output_folder: Output directory path
    """
    print("\n[STEP 5.5] EXPORT - Generating topics.csv...")
    
    all_topics = provided_topics + inferred_topics
    
    if not all_topics:
        print("[WARN] No topics found. Skipping topics.csv generation.")
        return
    
    # Create topics CSV with clean response data
    topic_data = {}

    for topic in all_topics:
        if topic in df.columns:
            # Get all non-empty response IDs for this topic
            responses = df[topic].dropna()
            responses = responses[responses != ""]
            # Filter out topics with zero responses
            if len(responses) > 0:
                topic_data[topic] = responses.tolist()
    
    # Find max length for consistent row count
    max_responses = max(len(v) for v in topic_data.values()) if topic_data else 0
    
    # Create DataFrame with topics as columns and response IDs as values
    topics_output = {}
    for topic, response_ids in topic_data.items():
        # Pad with empty strings to match max length
        padded = response_ids + [""] * (max_responses - len(response_ids))
        topics_output[topic] = padded
    
    topics_df = pd.DataFrame(topics_output)
    
    # Save to CSV
    topics_path = os.path.join(output_folder, "topics.csv")
    try:
        topics_df.to_csv(topics_path, index=False)
        print(f"  Saved {len(topics_df.columns)} topics to {topics_path}")
        print(f"    • Topics: {len(provided_topics)} provided + {len(inferred_topics)} inferred")
        for topic in all_topics:
            if topic in topics_df.columns:
                count = (topics_df[topic] != "").sum()
                print(f"    • {topic}: {count} responses")
    except Exception as e:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        topics_path = os.path.join(output_folder, f"topics_{ts}.csv")
        print(f"[WARN] Error writing topics file. Trying alternative path: {topics_path}")
        topics_df.to_csv(topics_path, index=False)


# =============================================================================
# STEP 6: EXPORT - Generate summaries and save
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
    output_folder: str = "data/final",
    excel_file: str = "data/UVABAIData.xlsx",
    num_topics: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    data_folder: path to input data folder
    output_folder: path to output folder
    excel_file: path to Excel file containing keywords/topics sheet
    num_topics: number of topics to infer with BERTopic

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
    
    # Task 21: Load provided topics and perform topic modeling
    print("\n[TASK 21] TOPIC MODELING - Inferring topics from responses...")
    provided_topics = load_provided_topics(excel_file)

    # Perform TF-IDF topic modeling on response texts
    model, inferred_topics_raw, probabilities = perform_topic_modeling(
        texts=df['response_text_original'].fillna("").tolist(),
        num_topics=num_topics
    )

    # Extract individual inferred topics
    inferred_topics = extract_inferred_topics(inferred_topics_raw, num_topics=num_topics)
    if inferred_topics:
        print(f"  Inferred {len(inferred_topics)} topics from response texts")

    # Match responses to topics (Task 21 & 22 setup)
    print("\n[TASK 21/22] TOPIC ASSIGNMENT - Matching responses to topics...")
    df_with_topics = match_responses_to_topics(
        df,
        provided_topics,
        inferred_topics
    )
    
    summary_df = generate_summary_by_question(df)
    generate_keywords_csv(df, output_folder)
    
    # Task 22: Generate topics.csv
    print("\n[TASK 22] TOPIC AGGREGATION - Creating topics.csv...")
    generate_topics_csv(df_with_topics, provided_topics, inferred_topics, output_folder)
    
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
    print(f"  4. {os.path.join(output_folder, 'topics.csv')}")
    print(f"     → Topic assignments ({len(provided_topics)} provided + {len(inferred_topics)} inferred topics)")
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
