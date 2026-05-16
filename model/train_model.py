# =============================================================================
# Customer Support Ticket Classification System
# File: model/train_model.py
# =============================================================================

import pandas as pd
import re
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "dataset", "tickets.csv")
)

MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_OUTPUT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

TEXT_COLUMN = "issue_description"
LABEL_COLUMN = "category"

TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset(filepath):
    """
    Load dataset using pandas.
    """

    print("\n[INFO] Loading dataset...")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}\n")

    return df


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text(text):
    """
    Clean and normalize ticket text.
    """

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove punctuation only
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df):
    """
    Apply preprocessing to dataframe.
    """

    print("[INFO] Preprocessing text data...")

    # Remove null values
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    # Apply preprocessing
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)

    # Remove empty rows
    df = df[df[TEXT_COLUMN] != ""]

    print(f"[INFO] Remaining samples: {len(df)}\n")

    return df


# =============================================================================
# DEBUG DATASET QUALITY
# =============================================================================

def show_sample_tickets(df):
    """
    Print sample tickets from each category.
    """

    print("=" * 60)
    print(" SAMPLE TICKETS ")
    print("=" * 60)

    for category in df[LABEL_COLUMN].unique():

        sample = df[df[LABEL_COLUMN] == category][TEXT_COLUMN].iloc[0]

        print(f"\nCATEGORY: {category}")
        print(f"TICKET : {sample[:200]}")


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def split_data(df):

    X = df[TEXT_COLUMN]
    y = df[LABEL_COLUMN]

    print("\n[INFO] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Testing samples : {len(X_test)}\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# TF-IDF VECTORIZATION
# =============================================================================

def vectorize_text(X_train, X_test):

    print("[INFO] Applying TF-IDF vectorization...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"[INFO] Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"[INFO] Training matrix shape: {X_train_vec.shape}")
    print(f"[INFO] Testing matrix shape : {X_test_vec.shape}\n")

    return X_train_vec, X_test_vec, vectorizer


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(X_train_vec, y_train):

    print("[INFO] Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(X_train_vec, y_train)

    print("[INFO] Model training completed.\n")

    return model


# =============================================================================
# EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X_test_vec, y_test):

    print("[INFO] Evaluating model...")

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f" MODEL ACCURACY: {accuracy * 100:.2f}% ")
    print("=" * 60)

    print("\nCLASSIFICATION REPORT:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


# =============================================================================
# SAVE MODEL + VECTORIZER
# =============================================================================

def save_artifacts(model, vectorizer):

    print("\n[INFO] Saving model artifacts...")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_PATH)

    print(f"[INFO] Model saved to      : {MODEL_OUTPUT_PATH}")
    print(f"[INFO] Vectorizer saved to : {VECTORIZER_OUTPUT_PATH}")

    print("\n[SUCCESS] Artifacts saved successfully!")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("\n" + "=" * 60)
    print(" CUSTOMER SUPPORT TICKET CLASSIFIER ")
    print("=" * 60)

    # Load dataset
    df = load_dataset(DATASET_PATH)

    # Preprocess dataset
    df = preprocess_dataframe(df)

    # Show sample tickets
    show_sample_tickets(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Vectorize text
    X_train_vec, X_test_vec, vectorizer = vectorize_text(
        X_train,
        X_test
    )

    # Train model
    model = train_model(X_train_vec, y_train)

    # Evaluate
    evaluate_model(model, X_test_vec, y_test)

    # Save model
    save_artifacts(model, vectorizer)

    print("\n[DONE] Phase 2 completed successfully.")
    print("[NEXT] Ready for Flask API integration.\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()