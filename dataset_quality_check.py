import pandas as pd
from collections import Counter
import re

# ==========================
# CONFIGURATION
# ==========================

DATASET_PATH = "dataset/tickets.csv"

TEXT_COLUMN = "issue_description"
LABEL_COLUMN = "category"

# ==========================
# LOAD DATASET
# ==========================

print("=" * 60)
print("DATASET QUALITY ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"\nDataset Shape: {df.shape}")

# ==========================
# MISSING VALUES
# ==========================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df[[TEXT_COLUMN, LABEL_COLUMN]].isnull().sum())

# ==========================
# CATEGORY DISTRIBUTION
# ==========================

print("\n" + "=" * 60)
print("CATEGORY DISTRIBUTION")
print("=" * 60)

print(df[LABEL_COLUMN].value_counts())

# ==========================
# DUPLICATE TICKETS
# ==========================

duplicates = df[TEXT_COLUMN].duplicated().sum()

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

print(f"Duplicate Tickets: {duplicates}")

duplicate_percentage = (duplicates / len(df)) * 100

print(f"Duplicate Percentage: {duplicate_percentage:.2f}%")

# ==========================
# VOCABULARY ANALYSIS
# ==========================

all_words = []

for text in df[TEXT_COLUMN].dropna():

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    all_words.extend(words)

unique_words = set(all_words)

print("\n" + "=" * 60)
print("VOCABULARY ANALYSIS")
print("=" * 60)

print(f"Total Words: {len(all_words):,}")
print(f"Unique Words: {len(unique_words):,}")

# ==========================
# TOP WORDS
# ==========================

print("\nTop 20 Most Frequent Words:")

for word, count in Counter(all_words).most_common(20):

    print(f"{word:<20} {count}")

# ==========================
# TICKET LENGTH ANALYSIS
# ==========================

df["ticket_length"] = df[TEXT_COLUMN].astype(str).apply(
    lambda x: len(x.split())
)

print("\n" + "=" * 60)
print("TICKET LENGTH ANALYSIS")
print("=" * 60)

print(f"Average Length : {df['ticket_length'].mean():.2f} words")
print(f"Minimum Length : {df['ticket_length'].min()} words")
print(f"Maximum Length : {df['ticket_length'].max()} words")

# ==========================
# SAMPLE TICKETS
# ==========================

print("\n" + "=" * 60)
print("SAMPLE TICKETS")
print("=" * 60)

for category in df[LABEL_COLUMN].unique():

    sample = df[df[LABEL_COLUMN] == category][TEXT_COLUMN].iloc[0]

    print(f"\nCATEGORY: {category}")
    print(f"TICKET: {sample}")

# ==========================
# CATEGORY CONSISTENCY TEST
# ==========================

print("\n" + "=" * 60)
print("CATEGORY CONSISTENCY CHECK")
print("=" * 60)

for category in df[LABEL_COLUMN].unique():

    print(f"\n{category}")

    samples = (
        df[df[LABEL_COLUMN] == category][TEXT_COLUMN]
        .sample(min(3, len(df[df[LABEL_COLUMN] == category])))
        .tolist()
    )

    for i, sample in enumerate(samples, 1):

        print(f"{i}. {sample}")

# ==========================
# QUALITY SCORE
# ==========================

score = 100

if duplicate_percentage > 20:
    score -= 20

if len(unique_words) < 1000:
    score -= 20

if df["ticket_length"].mean() < 5:
    score -= 20

print("\n" + "=" * 60)
print("DATASET QUALITY SCORE")
print("=" * 60)

print(f"Quality Score: {score}/100")

if score >= 80:
    print("Excellent dataset for NLP projects")

elif score >= 60:
    print("Usable dataset with some limitations")

elif score >= 40:
    print("Weak dataset - consider replacement")

else:
    print("Poor dataset - strongly recommend replacement")