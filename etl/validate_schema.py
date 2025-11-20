"""
Schema / content validator for survey responses.

Required columns (schema v1):
- survey_id (int-like)
- question_id (int-like)
- question_text (str, can be missing)
- response_id (int-like)
- response_text (str, can be empty)
- Timestamp (datetime-like)

Optional columns:
- survey_source (str): identifies which survey / platform the response came from
- email (str): responder email, may be missing
"""

import pandas as pd


REQUIRED_COLUMNS = [
    "survey_id",
    "question_id",
    "question_text",
    "response_id",
    "response_text",
    "Timestamp",
]

OPTIONAL_COLUMNS = [
    "survey_source",
    "email",
]


def _check_required_columns(df: pd.DataFrame) -> bool:
    ok = True
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            print(f"[ERROR] Missing required column: {col}")
            ok = False
    return ok


def _check_types(df: pd.DataFrame) -> bool:
    """need to be a bit forgiving on dtypes: we only check that columns are *compatible*.

    - ids -> integer / numeric
    - Timestamp -> datetime or convertible to datetime
    - text fields -> object/string
    """
    ok = True

    # integer-ish id columns
    for col in ["survey_id", "question_id", "response_id"]:
        if col not in df.columns:
            continue
        if not pd.api.types.is_integer_dtype(df[col]):
            if pd.api.types.is_numeric_dtype(df[col]):
                # pandas will make columns floats if there's any NaNs - importnt to keep in mind
                print(f"[WARN] {col} is numeric but not integer dtype ({df[col].dtype}).")
            else:
                print(f"[ERROR] {col} is not numeric (got {df[col].dtype}).")
                ok = False

    # timestamp
    if "Timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
            try:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
                print("[INFO] Converted Timestamp column to datetime.")
            except Exception as e:
                print(f"[ERROR] Could not parse Timestamp as datetime: {e}")
                ok = False

    # text-ish fields – just make sure they’re not purely numeric
    for col in ["question_text", "response_text", "survey_source", "email"]:
        if col in df.columns:
            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
                print(f"[WARN] {col} is not an object/string dtype (got {df[col].dtype}).")

    return ok


def _check_content(df: pd.DataFrame) -> None:
    """content-level checks that *do not* cause the whole schema to fail.
    This way downstream code can make decisions (drop rows, etc.)
    but validate_schema still returns True as long as structure is okay
    """
    # missing question_text
    if "question_text" in df.columns:
        missing_q = df["question_text"].isna().sum()
        if missing_q > 0:
            print(f"[WARN] {missing_q} rows have missing question_text.")

    # empty / missing response_text
    if "response_text" in df.columns:
        missing_resp = df["response_text"].isna().sum()
        empty_resp = (df["response_text"].astype(str).str.strip() == "").sum()
        if missing_resp > 0:
            print(f"[WARN] {missing_resp} rows have missing response_text (NaN).")
        if empty_resp > 0:
            print(f"[WARN] {empty_resp} rows have empty response_text strings.")

    # survey_source – multiple sources are expected/allowed
    if "survey_source" in df.columns:
        vc = df["survey_source"].value_counts(dropna=False)
        print("[INFO] survey_source distribution:")
        print(vc)

    # optional email field
    if "email" in df.columns:
        missing_email = df["email"].isna().sum()
        total = len(df)
        print(f"[INFO] email column present for {total - missing_email}/{total} rows.")


def validate_schema(df: pd.DataFrame, verbose: bool = True) -> bool:
    """validate the dataframe against our schema v1

    returns:
    1. True if the structural schema is ok (required columns + compatible types),
    2. False otherwise

    !! content issues like missing question_text or empty response_text do NOT cause
    validation to fail; they are reported as warnings so the ingestion / cleaning
    step can decide how to handle them
    """
    if verbose:
        print(f"[INFO] Validating schema for DataFrame with columns: {list(df.columns)}")

    has_columns = _check_required_columns(df)
    has_types = _check_types(df)

    if not (has_columns and has_types):
        if verbose:
            print("[ERROR] Schema validation failed.")
        return False

    # content-level checks (warnings/info only)
    _check_content(df)

    if verbose:
        print("[OK] Schema validation passed.")
    return True
