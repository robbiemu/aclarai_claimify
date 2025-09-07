
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

def save_mission_state(conn: sqlite3.Connection, thread_id: str, mission_state: str):
    """Save the mission state for a given thread ID."""
    sql = ''' INSERT OR REPLACE INTO missions(thread_id, mission_state)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, (thread_id, mission_state))
    conn.commit()

def load_mission_state(conn: sqlite3.Connection, thread_id: str) -> Optional[str]:
    """Load the mission state for a given thread ID."""
    cur = conn.cursor()
    cur.execute("SELECT mission_state FROM missions WHERE thread_id=?", (thread_id,))
    rows = cur.fetchall()
    if rows:
        return rows[0][0]
    return None
