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
import numpy as np
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

# ── Embedding libraries (sentence-transformers primary, GloVe fallback) ──
EMBEDDING_METHOD = None  # set during init: "sbert" | "glove" | None

try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import gensim.downloader as gensim_api
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


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

# ── Embedding globals ──
SBERT_MODEL = None
GLOVE_MODEL = None
RTC_EMBEDDINGS = None        # precomputed embeddings for RTC keywords
RTC_KEYWORDS_LIST = None     # flat list of RTC keywords (parallel to RTC_EMBEDDINGS)
SIMILARITY_THRESHOLD = 0.35  # cosine similarity cutoff (tunable)
KEYWORD_CACHE = {}               # text -> filtered keyword dict (avoids recomputation)

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

def _ensure_nltk():
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
        RAKE_INSTANCE = Rake()
    return RAKE_INSTANCE


# =============================================================================
# EMBEDDING HELPERS  (NEW)
# =============================================================================

def _init_embedding_model():
    """
    Try sentence-transformers first (best for multi-word phrases),
    fall back to GloVe via gensim, or disable similarity filtering.
    """
    global EMBEDDING_METHOD, SBERT_MODEL, GLOVE_MODEL

    if EMBEDDING_METHOD is not None:
        return  # already initialised

    # ── attempt 1: sentence-transformers ──
    if SBERT_AVAILABLE:
        try:
            print("  [EMBED] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            EMBEDDING_METHOD = "sbert"
            print("  [EMBED] sentence-transformers ready.")
            return
        except Exception as e:
            print(f"  [EMBED] sentence-transformers failed: {e}")

    # ── attempt 2: GloVe via gensim ──
    if GENSIM_AVAILABLE:
        try:
            print("  [EMBED] Loading GloVe vectors (glove-wiki-gigaword-100)...")
            GLOVE_MODEL = gensim_api.load("glove-wiki-gigaword-100")
            EMBEDDING_METHOD = "glove"
            print("  [EMBED] GloVe ready.")
            return
        except Exception as e:
            print(f"  [EMBED] GloVe failed: {e}")

    # ── nothing available ──
    EMBEDDING_METHOD = None
    print("  [EMBED] No embedding model available. Keyword filtering will use exact match only.")
    print("          Install with: pip install sentence-transformers   (recommended)")
    print("          Or:           pip install gensim")


def _embed_text(text: str) -> Optional[np.ndarray]:
    """
    Return a normalised embedding vector for *text*.
    Uses whichever model was loaded by _init_embedding_model().
    """
    if EMBEDDING_METHOD == "sbert":
        vec = SBERT_MODEL.encode(text, convert_to_numpy=True)
        return vec / (np.linalg.norm(vec) + 1e-10)

    if EMBEDDING_METHOD == "glove":
        # average word vectors for the phrase
        tokens = text.lower().split()
        vecs = [GLOVE_MODEL[t] for t in tokens if t in GLOVE_MODEL]
        if not vecs:
            return None
        avg = np.mean(vecs, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-10)

    return None

def _embed_batch(texts: List[str]) -> np.ndarray:
    """
    Batch-embed a list of texts. Returns matrix of shape (N, dim).
    Falls back to one-by-one for GloVe.
    """
    if EMBEDDING_METHOD == "sbert":
        vecs = SBERT_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms

    # GloVe path (or None path — returns zeros)
    results = []
    for t in texts:
        v = _embed_text(t)
        if v is not None:
            results.append(v)
        else:
            results.append(np.zeros(100))  # GloVe-100d zero vec
    return np.array(results)


def precompute_rtc_embeddings(rtc_keywords: List[str]):
    """
    Build the embedding matrix for the RTC keyword list once.
    Called after loading the RTC keywords and after the embedding model is ready.
    """
    global RTC_EMBEDDINGS, RTC_KEYWORDS_LIST

    _init_embedding_model()

    if EMBEDDING_METHOD is None:
        RTC_KEYWORDS_LIST = rtc_keywords
        RTC_EMBEDDINGS = None
        return

    RTC_KEYWORDS_LIST = rtc_keywords
    RTC_EMBEDDINGS = _embed_batch(rtc_keywords)
    print(f"  [EMBED] Precomputed embeddings for {len(rtc_keywords)} RTC keywords "
          f"(method={EMBEDDING_METHOD}, dim={RTC_EMBEDDINGS.shape[1]})")


def keyword_similarity_to_rtc(keyword: str) -> Tuple[float, str]:
    """
    Return (max_cosine_similarity, closest_rtc_keyword) for *keyword*
    against the precomputed RTC embeddings.
    """
    if RTC_EMBEDDINGS is None or EMBEDDING_METHOD is None:
        return 0.0, ""

    kw_vec = _embed_text(keyword)
    if kw_vec is None:
        return 0.0, ""

    # cosine similarities (vectors are already normalised)
    sims = RTC_EMBEDDINGS @ kw_vec
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), RTC_KEYWORDS_LIST[best_idx]


def filter_keywords_by_rtc_similarity(
    keyword_dict: Dict[str, int],
    threshold: Optional[float] = None
) -> Dict[str, int]:
    """
    Keep only keywords whose embedding similarity to at least one RTC keyword
    meets the threshold.  Uses batch embedding for speed.
    If no embedding model is loaded, passes through unchanged.
    """
    if RTC_EMBEDDINGS is None or EMBEDDING_METHOD is None:
        return keyword_dict  # no filtering possible

    if not keyword_dict:
        return keyword_dict

    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    keywords = list(keyword_dict.keys())

    # batch-embed all RAKE keywords at once (much faster than one-by-one)
    kw_embeddings = _embed_batch(keywords)  # shape (N, dim)

    # cosine sims: each keyword against all RTC keywords → take max per keyword
    sim_matrix = kw_embeddings @ RTC_EMBEDDINGS.T  # shape (N, M)
    max_sims = sim_matrix.max(axis=1)              # shape (N,)

    filtered = {}
    for i, kw in enumerate(keywords):
        if max_sims[i] >= threshold:
            filtered[kw] = keyword_dict[kw]

    return filtered


# =============================================================================
# STEP 1: INGEST - load and transform data
# =============================================================================

def load_and_transform_data(folder: str = "data") -> pd.DataFrame:
    """
    Load survey data and transform from wide to long format.
    """
    print("\n[STEP 1/5] INGEST - loading and transforming data...")
    all_responses = []
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Data folder not found: {folder}")
    
    response_id = 1
    
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        if os.path.isdir(file_path) or "processed" in file.lower():
            continue
        
        if file.lower().endswith(".xlsx"):
            try:
                xl_file = pd.ExcelFile(file_path)
                
                for sheet_name in xl_file.sheet_names:
                    if any(excl in sheet_name.lower() for excl in SHEETS_TO_EXCLUDE):
                        print(f"  [SKIP] Excluded sheet: {sheet_name}")
                        continue
                    
                    df = pd.read_excel(xl_file, sheet_name=sheet_name)
                    
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
                                    'sheet_name': sheet_name,
                                    'source_file': file
                                })
                                response_id += 1
                
                print(f"  [OK] Loaded {len(xl_file.sheet_names)} sheets from {file}")
                xl_file.close()
                
            except Exception as e:
                print(f"  [ERROR] Failed to load {file}: {e}")
        
        elif file.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path)
                
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
    
    long_df = pd.DataFrame(all_responses)
    
    unique_questions = long_df['question_text'].unique()
    question_id_map = {q: i+1 for i, q in enumerate(unique_questions)}
    long_df['question_id'] = long_df['question_text'].map(question_id_map)
    
    unique_sheets = long_df['sheet_name'].unique()
    survey_id_map = {s: i+1 for i, s in enumerate(unique_sheets)}
    long_df['survey_id'] = long_df['sheet_name'].map(survey_id_map)
    
    long_df['timestamp'] = datetime.now()
    
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

def load_rtc_keywords_from_csv(file_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load RTC keywords from the keywords CSV.
    Left column = keywords (comma-separated), right column = category.

    Returns:
        rtc_keywords: flat deduplicated list of individual keywords
        keyword_to_category: mapping from each keyword to its RTC category
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Keyword file not found: {file_path}")

    df = pd.read_csv(file_path, header=None, names=["keywords_raw", "category"])

    rtc_keywords = []
    keyword_to_category = {}
    seen = set()

    for _, row in df.iterrows():
        raw = str(row["keywords_raw"]).strip()
        category = str(row.get("category", "")).strip()
        if not raw or raw.lower() == "nan":
            continue

        # split on comma/semicolon
        parts = re.split(r"[,;]\s*", raw)
        for part in parts:
            kw = part.strip()
            # strip parenthetical notes like "(resume)" and trailing markers like "etc."
            kw = re.sub(r"\(.*?\)", "", kw).strip()
            kw = re.sub(r"\betc\.?\s*$", "", kw).strip()
            kw = re.sub(r"\?\?+\s*$", "", kw).strip()
            if not kw:
                continue
            norm = kw.lower()
            if norm not in seen:
                seen.add(norm)
                rtc_keywords.append(kw)
                if category and category.lower() != "nan":
                    keyword_to_category[norm] = category

    return rtc_keywords, keyword_to_category


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
# STEP 2: PREPROCESS - clean and normalize text
# =============================================================================

def preprocess_text(text: str, stop_words: set, lemmatizer) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    text = text.lower()
    text = re.sub(r'\p{P}+', ' ', text, flags=re.UNICODE)
    
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 2/5] PREPROCESS - Cleaning and normalizing text...")
    
    global STOP_WORDS, LEMMATIZER
    
    if STOP_WORDS is None or LEMMATIZER is None:
        _ensure_nltk()
        STOP_WORDS = set(stopwords.words("english"))
        LEMMATIZER = WordNetLemmatizer()
    
    cleaned_df = df.copy()
    
    cleaned_df['question_text_original'] = cleaned_df['question_text']
    cleaned_df['response_text_original'] = cleaned_df['response_text']
    
    cleaned_df['response_length'] = cleaned_df['response_text'].str.len()
    
    cleaned_df['question_text'] = cleaned_df['question_text'].apply(
        lambda t: preprocess_text(t, STOP_WORDS, LEMMATIZER)
    )
    cleaned_df['response_text'] = cleaned_df['response_text'].apply(
        lambda t: preprocess_text(t, STOP_WORDS, LEMMATIZER)
    )
    
    cleaned_df['word_count'] = cleaned_df['response_text'].str.split().str.len()
    
    print(f"  Preprocessed {len(cleaned_df):,} responses")
    print(f"    • avg response length: {cleaned_df['response_length'].mean():.1f} chars")
    print(f"    • avg word count: {cleaned_df['word_count'].mean():.1f} words")
    
    return cleaned_df


# =============================================================================
# STEP 3: SENTIMENT - analyze sentiment
# =============================================================================

def analyze_sentiment(text: str) -> Tuple[float, str]:
    if not text or not isinstance(text, str) or text.strip() == "":
        return 0.0, "neutral"
    
    if VADER_AVAILABLE:
        analyzer = _get_sentiment_analyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return compound, label
    else:
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
    print("\n[STEP 3/5] SENTIMENT - analyzing sentiment...")
    
    df_with_sentiment = df.copy()
    
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
    return SEED_KEYWORDS  

# using this helped instead of find present rtc kw since it rebuilds a df every time
def get_rtc_keywords_in_text(text: str, rtc_keywords: List[str]) -> List[str]:
    """
    Return RTC keywords that appear in the given text.
    Uses simple regex word-boundary matching.
    """
    if not text or not rtc_keywords:
        return []
    
    text_lower = text.lower()
    found = []

    for kw in rtc_keywords:
        kw_norm = kw.strip().lower()
        if not kw_norm:
            continue

        pattern = r"\b" + re.escape(kw_norm) + r"\b"
        if re.search(pattern, text_lower):
            found.append(kw_norm)

    return found 

def extract_keywords(text: Optional[str], max_phrases: int = 5) -> Dict[str, int]:
    """
    Extract keyword phrases from text using RAKE, then filter by embedding
    similarity to the RTC keyword list (if embeddings are available).
    Results are cached so repeated calls with the same text are instant.
    """
    if text is None or not isinstance(text, str):
        return {}
    
    text = str(text).strip()
    if not text or len(text) < 3:
        return {}

    # ── cache hit → skip RAKE + embedding entirely ──
    if text in KEYWORD_CACHE:
        return KEYWORD_CACHE[text]
    
    try:
        rake = _get_rake()
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        
        keyword_dict = {}
        for phrase in phrases:
            phrase = phrase.strip()
            
            phrase_lower = phrase.lower()
            if phrase_lower in EXCLUDED_KEYWORDS:
                continue
            
            count = text.lower().count(phrase.lower())
            if count > 0:
                keyword_dict[phrase] = count
            if len(keyword_dict) >= max_phrases:
                break

        # ── filter RAKE keywords by similarity to RTC keywords ──
        keyword_dict = filter_keywords_by_rtc_similarity(keyword_dict)
        
        KEYWORD_CACHE[text] = keyword_dict

        # ── add RTC keywords found directly in text ──
        rtc_keywords_in_text = get_rtc_keywords_in_text(text, RTC_KEYWORDS_LIST)

        for kw in rtc_keywords_in_text:
            keyword_dict[kw] = keyword_dict.get(kw, 0) + 1

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
            KEYWORD_CACHE[text] = {}
            return {}
        counts = Counter(tokens)
        raw = dict(counts.most_common(max_phrases))

        result = filter_keywords_by_rtc_similarity(raw)
        KEYWORD_CACHE[text] = result
        return result


def add_keyword_extraction(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 4/5] KEYWORDS - Extracting key phrases...")
    
    df_with_keywords = df.copy()
    
    df_with_keywords["keywords"] = df_with_keywords["response_text_original"].apply(
        lambda t: json.dumps(extract_keywords(t))
    )
    
    non_empty = (df_with_keywords["keywords"].str.len() > 2).sum()
    print(f"  extracted keywords from {non_empty:,} responses ({non_empty/len(df_with_keywords)*100:.1f}%)")
    
    return df_with_keywords


# =============================================================================
# STEP 5: EXPORT - Generate summaries and save
# =============================================================================

def generate_summary_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics grouped by question with keywords expanded to separate rows.
    Keywords are already filtered by RTC similarity during extraction.
    """
    print("\n[STEP 5/5] EXPORT - generating summaries...")

    summary_data = []

    grouped = df.groupby(["question_id", "question_text_original"])

    for (qid, qtext), group in grouped:
        base_stats = {
            "question_id": qid,
            "question_text": qtext,
            "total_responses": len(group),
            "avg_response_length": round(group["response_length"].mean(), 2),
            "avg_word_count": round(group["word_count"].mean(), 2),
        }

        if "sentiment_score" in group.columns:
            base_stats.update({
                "avg_sentiment_score": round(group["sentiment_score"].mean(), 2),
                "sentiment_std": round(group["sentiment_score"].std(), 2),
                "positive_responses": (group["sentiment_label"] == "positive").sum(),
                "negative_responses": (group["sentiment_label"] == "negative").sum(),
                "neutral_responses": (group["sentiment_label"] == "neutral").sum(),
            })

        base_stats["survey_sources"] = ", ".join(group["sheet_name"].unique())

        # ── Reuse keywords already computed in Step 4 (from the "keywords" column) ──
        all_keywords = Counter()
        for kw_json in group["keywords"].dropna():
            try:
                kw_dict = json.loads(kw_json)
                all_keywords.update(kw_dict)
            except (json.JSONDecodeError, TypeError):
                continue

        top_keywords = all_keywords.most_common(10)

        if top_keywords:
            for keyword, frequency in top_keywords:
                keyword_row = base_stats.copy()
                keyword_row["keyword"] = keyword
                keyword_row["frequency"] = frequency
                summary_data.append(keyword_row)
        else:
            keyword_row = base_stats.copy()
            keyword_row["keyword"] = ""
            keyword_row["frequency"] = 0
            summary_data.append(keyword_row)

    summary_df = pd.DataFrame(summary_data)

    summary_df = summary_df.sort_values(["question_id", "frequency"], ascending=[True, False])

    print(f"  Generated {len(summary_df):,} keyword rows for {len(summary_df['question_id'].unique())} questions")
    print(f"  Average {len(summary_df)/len(summary_df['question_id'].unique()):.1f} keywords per question")

    return summary_df


def generate_keywords_csv(
    df: pd.DataFrame,
    output_folder: str = "data/final",
    keyword_pool: Counter = None
) -> None:
    """
    Generate a simple keywords CSV where each keyword appears as many times as its frequency.

    If keyword_pool is provided (Task 20), use the combined filtered_rake_kw + present_rtc_kw.
    Otherwise (legacy), extract and use top 10 keywords from df["keywords"].
    """
    print("\n[STEP 5.5] EXPORT - generating keywords CSV...")

    if keyword_pool is not None:
        # Task 20: Use provided combined keyword pool (capped at top 200 for Tableau/word clouds)
        top_keywords = keyword_pool.most_common(200)
        print(f"  Using combined keyword pool (top 200 of {len(keyword_pool)} unique keywords)")
    else:
        # Legacy: Extract from df["keywords"] (top 10 only)
        question_stats = df.groupby('question_id').agg({
            'response_length': 'mean',
            'response_text_original': 'count'
        }).reset_index()

        meaningful_questions = question_stats[question_stats['response_length'] > 10]['question_id']
        df_filtered = df[df['question_id'].isin(meaningful_questions)]

        print(f"  Filtered to {len(df_filtered):,} responses from {len(meaningful_questions)} meaningful questions")

        all_keywords = Counter()

        # ── Reuse keywords already computed in Step 4 ──
        for kw_json in df_filtered["keywords"].dropna():
            try:
                kw_dict = json.loads(kw_json)
                all_keywords.update(kw_dict)
            except (json.JSONDecodeError, TypeError):
                continue

        top_keywords = all_keywords.most_common(10)

    keyword_rows = []
    for keyword, frequency in top_keywords:
        for _ in range(frequency):
            keyword_rows.append({"keyword": keyword})

    keywords_df = pd.DataFrame(keyword_rows)

    keywords_path = os.path.join(output_folder, "keywords.csv")
    try:
        keywords_df.to_csv(keywords_path, index=False)
        print(f"  Saved {len(keywords_df):,} keyword instances to {keywords_path}")
        print(f"  Top keywords: {', '.join([f'{kw}({freq})' for kw, freq in top_keywords[:10]])}")
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
    rtc_keywords_file: Optional[str] = None,
    similarity_threshold: float = 0.35
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    data_folder: path to input data folder
    output_folder: path to output folder
    rtc_keywords_file: path to CSV with RTC keywords (left column).
                       If None, looks for 'keywords.csv' in data_folder or
                       tries to load from the first .xlsx 'keywords' sheet.
    similarity_threshold: cosine similarity cutoff for filtering RAKE keywords
                          against RTC keywords. Lower = more permissive.
                          Recommended range: 0.25–0.50. Default 0.35.

    return: tuple of (responses_with_features_df, summary_by_question_df)
    """
    global SIMILARITY_THRESHOLD
    SIMILARITY_THRESHOLD = similarity_threshold

    print("=" * 80)
    print("BAI SURVEY ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
    
    # ── Load RTC keywords ──
    rtc_keywords = []
    rtc_categories = {}

    if rtc_keywords_file and os.path.exists(rtc_keywords_file):
        if rtc_keywords_file.lower().endswith(".csv"):
            rtc_keywords, rtc_categories = load_rtc_keywords_from_csv(rtc_keywords_file)
        else:
            rtc_keywords = load_rtc_keywords_from_excel(rtc_keywords_file)
        print(f"\n  [RTC] Loaded {len(rtc_keywords)} RTC keywords from {rtc_keywords_file}")
    else:
        # auto-discover: check for keywords.csv in data folder
        csv_candidate = os.path.join(data_folder, "keywords.csv")
        if os.path.exists(csv_candidate):
            rtc_keywords, rtc_categories = load_rtc_keywords_from_csv(csv_candidate)
            print(f"\n  [RTC] Auto-loaded {len(rtc_keywords)} RTC keywords from {csv_candidate}")
        else:
            # try to load from first xlsx file's keywords sheet
            for f in os.listdir(data_folder):
                if f.lower().endswith(".xlsx"):
                    try:
                        rtc_keywords = load_rtc_keywords_from_excel(
                            os.path.join(data_folder, f), sheet_name="keywords"
                        )
                        print(f"\n  [RTC] Auto-loaded {len(rtc_keywords)} RTC keywords from {f}")
                        break
                    except Exception:
                        continue

    # ── Precompute RTC embeddings ──
    if rtc_keywords:
        precompute_rtc_embeddings(rtc_keywords)
    else:
        print("\n  [RTC] No RTC keywords found — skipping embedding similarity filter.")

    # ── Create output directory before any saving ──
    os.makedirs(output_folder, exist_ok=True)

    # ── Execute pipeline steps ──
    df = load_and_transform_data(data_folder)
    df = preprocess_dataframe(df)
    df = add_sentiment_analysis(df)
    df = add_keyword_extraction(df)

    # ── Task 20: Aggregate filtered RAKE keywords and combine with present RTC keywords 
    # STUB: filtered_rake_kw aggregated from df["keywords"] below
    filtered_rake_kw = Counter()
    for kw_json in df["keywords"].dropna():
        try:
            kw_dict = json.loads(kw_json)
            filtered_rake_kw.update(kw_dict)
        except (json.JSONDecodeError, TypeError):
            continue

    # Get RTC keywords that are actually present in the data
    present_rtc_kw = find_present_rtc_keywords(df, rtc_keywords, text_column="response_text_original")

    # Combine both keyword lists
    combined_keywords = Counter(filtered_rake_kw)  # Start with RAKE keywords
    seen_lower = {k.lower() for k in filtered_rake_kw.keys()}

    # Add present_rtc_kw that aren't already in RAKE pool
    for rtc_kw in present_rtc_kw:
        if rtc_kw.lower() not in seen_lower:
            # Count actual occurrences in response text
            pattern = r"\b" + re.escape(rtc_kw.lower()) + r"\b"
            count = sum(1 for text in df["response_text_original"].dropna().astype(str)
                       if re.search(pattern, text.lower()))
            if count > 0:
                combined_keywords[rtc_kw] = count

    summary_df = generate_summary_by_question(df)    # reuses df["keywords"]
    generate_keywords_csv(df, output_folder, keyword_pool=combined_keywords)  # pass combined pool

    # ── Task 23: Convert keywords JSON → "theme" column, then select final columns ──
    df["theme"] = df["keywords"].apply(
        lambda x: ", ".join(json.loads(x).keys()) if x and x != "{}" else ""
    )
    df = df.drop(columns=["keywords"])

    df = df.rename(columns={
        "response_text_original": "response_text",
        "question_text_original": "question_text",
    })
    df = df[[
        "response_id",
        "survey_id",
        "question_id",
        "response_text",
        "sentiment_label",
        "sentiment_score",
        "theme",
        "question_text",
        "sheet_name",
        "source_file",
    ]]

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
    print(f"\nEmbedding method: {EMBEDDING_METHOD or 'none (exact match only)'}")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
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