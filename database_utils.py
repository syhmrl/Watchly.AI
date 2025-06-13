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
        
        self._cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_analysis (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name           TEXT    NOT NULL,
                video_width          INTEGER NOT NULL,
                video_height         INTEGER NOT NULL,
                video_fps            INTEGER NOT NULL,
                total_count          INTEGER NOT NULL,
                model_name           TEXT    NOT NULL,
                confidence           REAL    NOT NULL,
                iou                  REAL    NOT NULL,
                last_tracked_id      INTEGER,
                tracker_type         TEXT    NOT NULL,
                track_high_thresh    REAL    NOT NULL,
                track_low_thresh     REAL    NOT NULL,
                new_track_thresh     REAL    NOT NULL,
                track_buffer         INTEGER NOT NULL,
                match_thresh         REAL    NOT NULL,
                fuse_score           INTEGER NOT NULL,
                gmc_method           TEXT,
                proximity_thresh     REAL,
                appearance_thresh    REAL,
                with_reid            INTEGER,
                tracker_model        TEXT
            );
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
    
def insert_video_analysis(
    video_name, video_width, video_height, video_fps,
    total_count, model_name, confidence, iou,
    last_tracked_id, tracker_settings
):
    """
    Insert a single summary row into video_analysis.
    tracker_settings should be your dict loaded from YAML.
    """
    db = Database()
    conn, cur = db.get_connection()
    ts = tracker_settings
    cur.execute("""
        INSERT INTO video_analysis (
          video_name, video_width, video_height, video_fps,
          total_count, model_name, confidence, iou,
          last_tracked_id, tracker_type, track_high_thresh,
          track_low_thresh, new_track_thresh, track_buffer,
          match_thresh, fuse_score, gmc_method, proximity_thresh,
          appearance_thresh, with_reid, tracker_model
        ) VALUES (
          ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, (
        video_name,
        video_width,
        video_height,
        int(video_fps),
        total_count,
        model_name,
        confidence,
        iou,
        last_tracked_id,
        ts["tracker_type"],
        ts["track_high_thresh"],
        ts["track_low_thresh"],
        ts["new_track_thresh"],
        ts["track_buffer"],
        ts["match_thresh"],
        1 if ts["fuse_score"] else 0,
        ts.get("gmc_method"),
        ts.get("proximity_thresh"),
        ts.get("appearance_thresh"),
        1 if ts.get("with_reid") else 0,
        ts.get("model")
    ))
    conn.commit()

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
        
def get_total_counts_line_mode(start_ts, end_ts, mode):
    db = Database()
    _, cursor = db.get_connection()

    # Get total count
    cursor.execute("""SELECT direction, COUNT(*) FROM crossing_events 
                                WHERE timestamp BETWEEN ? AND ? AND mode_type = ? GROUP BY direction""",
                                (start_ts, end_ts, mode))
    data = cursor.fetchone()

    return data

def get_total_counts_crowd_mode(start_ts, end_ts, mode):
    db = Database()
    _, cursor = db.get_connection()
    
    # Get total count
    cursor.execute("""SELECT COUNT(*) FROM crossing_events 
                              WHERE timestamp BETWEEN ? AND ? AND mode_type = ?""",
                              (start_ts, end_ts, mode))

    count = cursor.fetchone()[0] or 0

    return count