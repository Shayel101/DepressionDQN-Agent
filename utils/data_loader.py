import sys
import os
import pandas as pd
import numpy as np
from config import SCORES_PATH, CONDITIONS_DIR, CONTROLS_DIR, SEQUENCE_LENGTH

def preprocess_static_features(df):
    # Convert relevant columns to numeric
    for col in ['age', 'IQ_Raven', 'MADRS']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Map gender to numeric (example: m -> 1, f -> 0)
    df['gender_numeric'] = df['gender'].str.lower().map({'m': 1, 'f': 0})
    # Derive binary label from 'group': depr -> 1, control -> 0
    df['label'] = df['group'].str.lower().map({'depr': 1, 'control': 0})
    return df

def load_scores():
    """Load the participants CSV with static patient data."""
    # Assuming participants.csv is comma-separated. Adjust 'sep' if needed.
    scores_df = pd.read_csv(SCORES_PATH)
    scores_df = preprocess_static_features(scores_df)
    return scores_df

def load_time_series_data(file_name, group):
    """
    Load a single subject's time-series CSV file.
    Choose folder based on group: if group is 'depr', use CONDITIONS_DIR; if 'control', use CONTROLS_DIR.
    """
    if group == "depr":
        file_path = os.path.join(CONDITIONS_DIR, file_name)
    elif group == "control":
        file_path = os.path.join(CONTROLS_DIR, file_name)
    else:
        raise ValueError("Group must be 'depr' or 'control'")
    
    df = pd.read_csv(file_path)
    activity = df['activity'].values
    if len(activity) < SEQUENCE_LENGTH:
        activity = np.pad(activity, (0, SEQUENCE_LENGTH - len(activity)), 'constant')
    else:
        activity = activity[:SEQUENCE_LENGTH]
    return activity

def create_dataset():
    """
    Combine static features and time-series data.
    Uses the participant_id to locate the corresponding CSV in the appropriate folder.
    """
    scores_df = load_scores()
    
    # Select static features (customize as needed)
    X_static = scores_df[['gender_numeric', 'age', 'IQ_Raven', 'MADRS']].copy()
    X_time = []
    y = []
    
    for idx, row in scores_df.iterrows():
        participant_id = row['participant_id']
        group = row['group'].strip().lower()  # Ensure group is in lowercase without extra spaces
        file_name = participant_id + ".csv"  # e.g., "sub-01.csv"
        try:
            time_series = load_time_series_data(file_name, group)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue
        X_time.append(time_series)
        y.append(row['label'])
    
    X_time = np.array(X_time)
    y = np.array(y)
    return X_static, X_time, y

if __name__ == '__main__':
    X_static, X_time, y = create_dataset()
    print("Static Features shape:", X_static.shape)
    print("Time-Series Features shape:", X_time.shape)
    print("Labels shape:", y.shape)
