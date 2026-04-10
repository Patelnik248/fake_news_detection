# =============================================================
# preprocess.py — Member 1: Data Preprocessing
# Loads True.csv and Fake.csv, cleans, and prepares the data
# =============================================================

import re
import pandas as pd
from sklearn.model_selection import train_test_split

TRUE_CSV = "data/True.csv"
FAKE_CSV = "data/Fake.csv"

def load_data():
    """Load True.csv and Fake.csv and combine into one DataFrame."""
    true_df = pd.read_csv(TRUE_CSV)
    fake_df = pd.read_csv(FAKE_CSV)

    # Assign labels: 1 = Real, 0 = Fake
    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    print(f"[INFO] Loaded {len(df)} articles ({len(true_df)} real, {len(fake_df)} fake)")
    return df


def clean_text(text):
    """
    Basic text cleaning steps:
    - Lowercase
    - Remove URLs, HTML tags, punctuation, extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"<.*?>", "", text)                # remove HTML
    text = re.sub(r"[^a-z\s]", " ", text)            # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df):
    """Merge title+text, clean, and drop empty rows."""
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.strip() != ""].reset_index(drop=True)
    print(f"[INFO] After cleaning: {len(df)} articles")
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Split into train/test sets."""
    X = df["content"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def get_dataset():
    """One-call helper: load -> preprocess -> split."""
    df = load_data()
    df = preprocess(df)
    return split_data(df)
