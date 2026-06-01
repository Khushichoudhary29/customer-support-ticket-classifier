import sqlite3

DATABASE = "database/predictions.db"


# --------------------------------------------------
# Database Connection
# --------------------------------------------------
def get_connection():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# --------------------------------------------------
# Create Database Table
# --------------------------------------------------
def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            ticket_text TEXT NOT NULL,

            prediction TEXT NOT NULL,

            confidence REAL NOT NULL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    conn.commit()

    conn.close()

    print("Database initialized successfully.")


# --------------------------------------------------
# Save Prediction
# --------------------------------------------------
def save_prediction(ticket_text,
                    prediction,
                    confidence):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

        INSERT INTO predictions
        (
            ticket_text,
            prediction,
            confidence
        )

        VALUES (?, ?, ?)

    """,
    (
        ticket_text,
        prediction,
        confidence
    ))

    conn.commit()

    conn.close()


# --------------------------------------------------
# Prediction History
# --------------------------------------------------
def get_prediction_history():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *
        FROM predictions

        ORDER BY id DESC

        LIMIT 50

    """)

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


# --------------------------------------------------
# Dashboard Statistics
# --------------------------------------------------
def get_statistics():

    conn = get_connection()

    cursor = conn.cursor()

    # Total Predictions
    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    total_predictions = cursor.fetchone()[0]

    # Average Confidence
    cursor.execute(
        "SELECT AVG(confidence) FROM predictions"
    )

    avg_confidence = cursor.fetchone()[0]

    if avg_confidence is None:
        avg_confidence = 0

    # Most Common Category
    cursor.execute("""
        SELECT prediction,
               COUNT(*) as count

        FROM predictions

        GROUP BY prediction

        ORDER BY count DESC

        LIMIT 1
    """)

    result = cursor.fetchone()

    most_common = result[0] if result else "N/A"

    conn.close()

    return {
        "total_predictions": total_predictions,
        "average_confidence": round(avg_confidence, 2),
        "most_common_category": most_common
    }