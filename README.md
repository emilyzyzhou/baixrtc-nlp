# baixrtc-nlp
NLP Project by Business and AI Institute at UVA(BAI) for Rewriting the Code

PROCESSING FLOW:
1. INGEST      → Load Excel/CSV files and transform from wide to long format
2. PREPROCESS  → Clean and normalize text (lowercase, remove stopwords, lemmatize)
3. SENTIMENT   → Analyze sentiment using VADER (positive/negative/neutral)
4. KEYWORDS    → Extract key phrases using RAKE algorithm
5. TOPICS      → Infer topics using TF-IDF and match against provided/inferred topics
6. FEATURES    → Add keywords, topics, and theme to each response (Tasks 24-25)
7. EXPORT      → Generate four CSV files ready for Tableau

**OUTPUT FILES & FIELD DEFINITIONS
**
FILE 1: responses_with_features.csv
------------------------------------
Row-level data - each row represents one response to one question
Total columns: 16

IDENTIFIERS:
  response_id          Unique ID for each response (1, 2, 3...)
  survey_id            ID of the survey/sheet (1-6 for your data)
  question_id          ID of the question (1-40 for your data)

RAW TEXT:
  response_text        Original response text before cleaning
                       (Used for sentiment, keywords, and topics)
  question_text        Original question text

SENTIMENT ANALYSIS:
  sentiment_score      Continuous score from -1.0 (negative) to +1.0 (positive)
                      Calculated using VADER sentiment analyzer
  sentiment_label      Categorical: "positive" | "negative" | "neutral"
                      Based on sentiment_score thresholds (±0.05)

KEYWORDS & TOPICS (Task 24-25):
  keywords             Comma-separated list of key phrases extracted from this response
                      Extracted using RAKE algorithm and filtered against RTC keywords
  topics               Comma-separated list of topics relevant to this response
                      Topics matched using provided list + inferred topics from TF-IDF/BERTopic
  theme                Short summary derived from top keywords in response
                      (Kept for backward compatibility)

METADATA:
  sheet_name          Original Excel sheet name (survey source)
  source_file         Original Excel/CSV filename


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



**USAGE NOTES FOR TABLEAU
**
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

FILE 4: topics.csv (Task 22)
----------------------------
Topic assignment matrix - each column is a topic, each cell contains matching response_id (or empty)
Total columns: number of unique topics

STRUCTURE:
  Each column header is a topic (provided or inferred)
  Cell values are response_id (e.g., 1, 42, 156) if that response matches the topic
  Empty cells indicate no match

USAGE:
  • Identify which responses belong to each topic
  • Count responses per topic
  • Cross-reference topics with responses_with_features.csv

FILTERING:
- Filter by survey_sources to compare surveys
- Filter by sentiment_label for positive-only or negative-only analysis
- Filter by keywords or topics columns for detailed analysis
- Use question_text contains "keyword" for topic-specific analysis

