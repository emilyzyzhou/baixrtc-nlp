# baixrtc-nlp
NLP Project by Business and AI Institute at UVA(BAI) for Rewriting the Code

PROCESSING FLOW:
1. INGEST      → Load Excel/CSV files and transform from wide to long format
2. PREPROCESS  → Clean and normalize text (lowercase, remove stopwords, lemmatize)
3. SENTIMENT   → Analyze sentiment using VADER (positive/negative/neutral)
4. KEYWORDS    → Extract key phrases using RAKE algorithm
5. EXPORT      → Generate two CSV files ready for Tableau

================================================================================
OUTPUT FILES & FIELD DEFINITIONS
================================================================================

FILE 1: responses_with_features.csv
------------------------------------
Row-level data - each row represents one response to one question
Total columns: 14

IDENTIFIERS:
  response_id          Unique ID for each response (1, 2, 3...)
  survey_id            ID of the survey/sheet (1-6 for your data)
  question_id          ID of the question (1-40 for your data)

RAW TEXT (ORIGINAL):
  question_text_original    Original question text before cleaning
  response_text_original    Original response text before cleaning
                           (Used for sentiment & keywords - preserves nuance)

PROCESSED TEXT (CLEANED):
  question_text        Preprocessed question (lowercase, no stopwords/punct)
  response_text        Preprocessed response (lowercase, no stopwords/punct)
                      (Available for advanced text analysis if needed)

METRICS:
  response_length      Character count of original response
  word_count          Word count after preprocessing

SENTIMENT ANALYSIS:
  sentiment_score      Continuous score from -1.0 (negative) to +1.0 (positive)
                      Calculated using VADER sentiment analyzer
  sentiment_label      Categorical: "positive" | "negative" | "neutral"
                      Based on sentiment_score thresholds (±0.05)

METADATA:
  sheet_name          Original Excel sheet name (survey source)
  source_file         Original Excel/CSV filename
  timestamp           When the pipeline processed this data


FILE 2: summary_by_question.csv
--------------------------------
Question-level aggregated statistics - each row represents one question-keyword combination
Total columns: 13

IDENTIFIERS:
  question_id          Unique question identifier (matches responses file)
  question_text        Full original question text

RESPONSE STATISTICS:
  total_responses      Number of responses to this question
  avg_response_length  Average character count across all responses
  avg_word_count      Average word count (after preprocessing)

SENTIMENT STATISTICS:
  avg_sentiment_score      Mean sentiment score for this question
  sentiment_std           Standard deviation of sentiment scores
  positive_responses      Count of positive responses
  negative_responses      Count of negative responses
  neutral_responses       Count of neutral responses

KEYWORD ANALYSIS:
  keyword             Individual keyword extracted from responses to this question
  frequency           How many times this keyword appeared across all responses

METADATA:
  survey_sources      Comma-separated list of sheets containing this question

FILE 3: keywords.csv
--------------------
Simple keyword frequency list - each keyword appears as many times as its frequency
Total columns: 1

KEYWORD:
  keyword             Individual keyword repeated by frequency count
                     Top 10 keywords across ALL meaningful responses (not per question)
                     Filters out generic words (yes, no, maybe, good, bad, etc.) and
                     excludes responses from yes/no questions (avg response length < 10 chars)


================================================================================
USAGE NOTES FOR TABLEAU
================================================================================

LOADING DATA:
- Import both CSV files as separate data sources
- Use question_id to join them if needed (one-to-many relationship)

RECOMMENDED VISUALIZATIONS:

responses_with_features.csv:
  • Sentiment distribution histogram (sentiment_score)
  • Response detail table filtered by question_id or sentiment_label
  • Time series if you have timestamp variations
  • Word count vs sentiment score scatter plot

summary_by_question.csv:
  • Questions ranked by avg_sentiment_score
  • Bar charts showing positive/negative/neutral counts
  • Top questions by total_responses
  • Word clouds using keyword and frequency columns (no JSON parsing needed!)
  • Keyword frequency analysis across questions
  • Filter keywords by minimum frequency threshold

keywords.csv:
  • Simple word clouds (each word appears by frequency)
  • Basic keyword frequency visualization
  • Import into simple word cloud tools

FILTERING:
- Filter by survey_sources to compare surveys
- Filter by sentiment_label for positive-only or negative-only analysis
- Use question_text contains "keyword" for topic-specific analysis

