
import sqlite3
from typing import Any, Dict, Optional

def create_connection(db_file: str) -> Optional[sqlite3.Connection]:
    """Create a database connection to the SQLite database specified by db_file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table(conn: sqlite3.Connection):
    """Create the missions table."""
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS missions (
                thread_id TEXT PRIMARY KEY,
                mission_state TEXT NOT NULL
            )
        ''')
    except sqlite3.Error as e:
        print(e)

