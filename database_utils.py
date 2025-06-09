import sqlite3
import queue
import time
from config import get_database_path

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
        self._connection = sqlite3.connect(get_database_path(), check_same_thread=False)
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

    # def insert_events(self, list_of_tuples):
    #     try:
    #         conn, cursor = self.get_connection()
    #         with conn:
    #             cursor.executemany(
    #                 "INSERT INTO crossing_events (source, track_id, direction, timestamp, mode_type) VALUES (?, ?, ?, ?, ?)",
    #                 list_of_tuples
    #             )
    #     except sqlite3.Error as e:
    #         print(f"Database error: {e}")

    # # Database insert function
    # def insert_to_db(self, thread_controller):
    #     batch = []
    #     while not thread_controller.stop_event.is_set() or not thread_controller.pending_inserts.empty():
    #         try:
    #             item = thread_controller.pending_inserts.get(timeout=0.5)
    #             batch.append(item)
    #             # Keep draining until queue is empty
    #             while True:
    #                 try:
    #                     batch.append(thread_controller.pending_inserts.get_nowait())
    #                 except queue.Empty:
    #                     break
    #         except queue.Empty:
    #             pass

    #         if batch:
    #             self.insert_events(batch)
    #             batch.clear()



def insert_to_db(thread_controller):
    """
    Drains a thread-safe queue of pending inserts and writes them to SQLite every 0.5 seconds.
    Expects thread_controller.pending_inserts to be a Queue() containing tuples:
      (source, track_id, direction, timestamp, mode_type)
    """
    db = Database()
    conn, cursor = db.get_connection()
    INSERT_SQL = """
        INSERT INTO crossing_events (source, track_id, direction, timestamp, mode_type)
        VALUES (?, ?, ?, ?, ?)
    """

    while not thread_controller.stop_event.is_set():
        batch = []
        # Drain the queue without blocking
        while True:
            try:
                record = thread_controller.pending_inserts.get_nowait()
                batch.append(record)
            except queue.Empty:
                break

        if batch:
            try:
                with conn:
                    cursor.executemany(INSERT_SQL, batch)
            except sqlite3.Error as e:
                print(f"Database error during batch insert: {e}")

        time.sleep(0.5)

    # Before exiting, flush any remaining items
    remaining = []
    while True:
        try:
            record = thread_controller.pending_inserts.get_nowait()
            remaining.append(record)
        except queue.Empty:
            break

    if remaining:
        try:
            with conn:
                cursor.executemany(INSERT_SQL, remaining)
        except sqlite3.Error as e:
            print(f"Database error during final insert: {e}")

    print("Database insertion thread stopped")

def get_total_counts(start_ts, end_ts):
    db = Database()
    _, cursor = db.get_connection()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM crossing_events WHERE direction = 'enter' AND timestamp BETWEEN ? AND ?", 
                    (start_ts, end_ts))
    count = cursor.fetchone()[0]

    return count

def get_grouped_counts(start_ts, end_ts, groupby):
    db = Database()
    _, cursor = db.get_connection()

    query = f"""
            SELECT {groupby} as timeperiod, COUNT(*) 
            FROM crossing_events 
            WHERE direction = 'enter' AND timestamp BETWEEN ? AND ? 
            GROUP BY timeperiod
            ORDER BY timeperiod
        """
        
    cursor.execute(query, (start_ts, end_ts))
    data = cursor.fetchall()

    return data
        