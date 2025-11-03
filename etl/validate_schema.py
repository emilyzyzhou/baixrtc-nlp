# survey_id (int)
# question_id (int)
# question_text (str)
# response_id (int)
# response_text (str)
# (Timestamp) (datetime)

import pandas as pd

def validate_schema(df: pd.DataFrame) -> bool:
    expected_columns = {
        "survey_id": int,
        "question_id": int,
        "question_text": str
        # add remaining expected columns here
    }
    
    # current error handling is harsh, can be modified to be more lenient
    for column, dtype in expected_columns.items():
        if column not in df.columns:
            print(f"Missing column: {column}")
            return False 
        if not pd.api.types.is_dtype_equal(df[column].dtype, dtype):
            print(f"Incorrect type for column: {column}. Expected {dtype}, got {df[column].dtype}")
            return False
            
    print("Schema validation passed.")
    return True