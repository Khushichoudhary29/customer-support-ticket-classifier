import sqlite3

DATABASE = "database/predictions.db"


def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


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

    print("Database initialized successfully.")
    
    
