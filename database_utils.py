import sqlite3
import time

class Database:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._connection = None
            cls._instance._cursor = None
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        self._connection = sqlite3.connect('watchly_ai.db', check_same_thread=False)
        self._cursor = self._connection.cursor()
        
        # Use when you to clear table contents
        # self._cursor.execute("DROP TABLE IF EXISTS crossing_events")
         
        # Initialize schema
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS crossing_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                source    TEXT,
                track_id  INTEGER,
                direction TEXT,
                timestamp TEXT,
                mode_type TEXT DEFAULT 'line'
            )
        ''')

        self._connection.commit()
    
    def get_connection(self):
        return self._connection, self._cursor
    
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
            self._cursor = None
            Database._instance = None


# Database insert function
def insert_to_db(thread_controller):
    db = Database()
    conn, cursor = db.get_connection()
    
    while not thread_controller.stop_event.is_set():
        time.sleep(0.5)  # Process every half second
        if thread_controller.pending_inserts:
            # Copy the current pending inserts and clear the list
            inserts_to_process = thread_controller.pending_inserts.copy()
            thread_controller.pending_inserts.clear()
            
            try:
                with conn:
                    cursor.executemany(
                        "INSERT INTO crossing_events (source, track_id, direction, timestamp, mode_type) VALUES (?, ?, ?, ?, ?)", 
                        inserts_to_process
                    )
            except sqlite3.Error as e:
                print(f"Database error: {e}")
    
    print("Database thread stopped")
