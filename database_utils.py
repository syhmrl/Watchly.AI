import sqlite3
import queue
import time

from datetime import datetime
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
                start_timestamp      TEXT,
                end_timestamp        TEXT,
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
                tracker_model        TEXT,
                run_index            INTEGER DEFAULT 1,
                ground_truth_count   INTEGER,
                precision            REAL,
                recall               REAL,
                f1_score             REAL,
                processing_time_ms   REAL,
                frame_count          INTEGER,
                analysis_timestamp   TEXT
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
    start_timestamp, end_timestamp,
    total_count, model_name, confidence, iou,
    last_tracked_id, tracker_settings,
    run_index=1, ground_truth_count=None, precision=None, recall=None, f1_score=None,
    processing_time_ms=None, frame_count=None
):
    """
    Insert a single summary row into video_analysis.
    tracker_settings should be your dict loaded from YAML.
    """
    db = Database()
    conn, cur = db.get_connection()
    ts = tracker_settings
    
    # Generate analysis timestamp
    analysis_timestamp = datetime.now().isoformat()
    
    try:
        cur.execute("""
            INSERT INTO video_analysis (
            video_name, video_width, video_height, video_fps,
            start_timestamp, end_timestamp,
            total_count, model_name, confidence, iou,
            last_tracked_id, tracker_type, track_high_thresh,
            track_low_thresh, new_track_thresh, track_buffer,
            match_thresh, fuse_score, gmc_method, proximity_thresh,
            appearance_thresh, with_reid, tracker_model,
            run_index, ground_truth_count, precision, recall, f1_score,
            processing_time_ms, frame_count, analysis_timestamp
            ) VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            video_name,
            video_width,
            video_height,
            int(video_fps),
            start_timestamp,
            end_timestamp,
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
            ts.get("model"),
            run_index,
            ground_truth_count,
            precision,
            recall,
            f1_score,
            processing_time_ms,
            frame_count,
            analysis_timestamp
        ))
        
        conn.commit()
        
        print(f"Video analysis inserted for {video_name}, run {run_index}")
    except sqlite3.Error as e:
        print(f"Error inserting video analysis: {e}")
        raise

def get_analysis_comparison(video_name):
    """
    Get all analysis runs for a specific video for comparison
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        cursor.execute("""
            SELECT 
                run_index, total_count, ground_truth_count, precision, recall, f1_score,
                processing_time_ms, frame_count, analysis_timestamp,
                model_name, confidence, iou, tracker_type
            FROM video_analysis 
            WHERE video_name = ? 
            ORDER BY run_index, analysis_timestamp
        """, (video_name,))
        
        results = cursor.fetchall()
        columns = [
            'run_index', 'total_count', 'ground_truth_count', 'precision', 'recall', 'f1_score',
            'processing_time_ms', 'frame_count', 'analysis_timestamp',
            'model_name', 'confidence', 'iou', 'tracker_type'
        ]
        
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        print(f"Error getting analysis comparison: {e}")
        return []
    finally:
        db.close()

def get_next_run_index(video_name):
    """
    Get the next run index for a video
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        cursor.execute("""
            SELECT COALESCE(MAX(run_index), 0) + 1 
            FROM video_analysis 
            WHERE video_name = ?
        """, (video_name,))
        
        result = cursor.fetchone()
        return result[0] if result else 1
    except Exception as e:
        print(f"Error getting next run index: {e}")
        return 1
    finally:
        db.close()

# Required database utility functions to add to database_utils.py
def get_distinct_sources():
    """
    Get distinct source values from crossing_events table
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        cursor.execute("SELECT DISTINCT source FROM crossing_events WHERE source IS NOT NULL ORDER BY source")
        sources = [row[0] for row in cursor.fetchall()]
        return sources
    except Exception as e:
        print(f"Error getting distinct sources: {e}")
        return []
    finally:
        db.close()

def get_video_names():
    """
    Get distinct video_name values from video_analysis table
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        cursor.execute("SELECT DISTINCT video_name FROM video_analysis WHERE video_name IS NOT NULL ORDER BY video_name")
        videos = [row[0] for row in cursor.fetchall()]
        return videos
    except Exception as e:
        print(f"Error getting video names: {e}")
        return []
    finally:
        db.close()
        
def get_video_timestamps(video_name):
    """
    Get start and end timestamps for a specific video from video_analysis table
    Returns tuple of (start_datetime, end_datetime)
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        cursor.execute("""
            SELECT start_timestamp as start_time, end_timestamp as end_time 
            FROM video_analysis 
            WHERE video_name = ?
        """, (video_name,))
        
        result = cursor.fetchone()
        if result and result[0] and result[1]:
            from datetime import datetime
            # Parse the timestamps - adjust format as needed based on your timestamp format
            start_dt = datetime.fromisoformat(result[0].replace('T', ' ')) if 'T' in result[0] else datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
            end_dt = datetime.fromisoformat(result[1].replace('T', ' ')) if 'T' in result[1] else datetime.strptime(result[1], '%Y-%m-%d %H:%M:%S')
            return start_dt, end_dt
        return None, None
    except Exception as e:
        print(f"Error getting video timestamps: {e}")
        return None, None
    finally:
        db.close()
        


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

def get_video_run_indices(video_name):
    """
    Get all run indices for a specific video
    Returns a list of unique run_index values for the given video
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        
        query = """
            SELECT DISTINCT run_index
            FROM video_analysis 
            WHERE video_name = ?
            ORDER BY run_index
        """
        
        cursor.execute(query, (video_name,))
        results = cursor.fetchall()
        
        # Extract run_index values from tuples
        return [row[0] for row in results if row[0] is not None]
        
    except Exception as e:
        print(f"Error getting video run indices: {e}")
        return []
    finally:
        db.close()

def get_individual_timestamps_filtered(start_timestamp, end_timestamp, filters):
    """
    Get individual timestamps (not grouped) with filters applied
    Used for second-level resolution to preserve exact timing
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        
        # Build WHERE clause
        where_conditions = ["timestamp BETWEEN ? AND ?"]
        params = [start_timestamp, end_timestamp]
        
        if filters['mode_type']:
            where_conditions.append("mode_type = ?")
            params.append(filters['mode_type'])
        
        if filters['source']:
            where_conditions.append("source = ?")
            params.append(filters['source'])
        
        if filters['direction']:
            where_conditions.append("direction = ?")
            params.append(filters['direction'])
            
        if filters['run_index']:
            where_conditions.append("run_index = ?")
            params.append(int(filters['run_index']))
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
            SELECT timestamp
            FROM crossing_events 
            WHERE {where_clause}
            ORDER BY timestamp
        """
        
        cursor.execute(query, params)
        return cursor.fetchall()
        
    except Exception as e:
        print(f"Error getting individual timestamps: {e}")
        return []
    finally:
        db.close()

def get_total_counts_filtered(start_timestamp, end_timestamp, filters):
    """
    Get total count with filters applied
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        
        # Build WHERE clause
        where_conditions = ["timestamp BETWEEN ? AND ?"]
        params = [start_timestamp, end_timestamp]
        
        if filters['mode_type']:
            where_conditions.append("mode_type = ?")
            params.append(filters['mode_type'])
        
        if filters['source']:
            where_conditions.append("source = ?")
            params.append(filters['source'])
        
        if filters['direction']:
            where_conditions.append("direction = ?")
            params.append(filters['direction'])
            
        if filters['run_index']:
            where_conditions.append("run_index = ?")
            params.append(int(filters['run_index']))
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"SELECT COUNT(*) FROM crossing_events WHERE {where_clause}"
        cursor.execute(query, params)
        
        result = cursor.fetchone()
        return result[0] if result else 0
    except Exception as e:
        print(f"Error getting filtered total counts: {e}")
        return 0
    finally:
        db.close()


def get_grouped_counts_filtered(start_timestamp, end_timestamp, groupby, filters):
    """
    Get grouped counts with filters applied
    """
    db = Database()
    try:
        _, cursor = db.get_connection()
        
        # Build WHERE clause
        where_conditions = ["timestamp BETWEEN ? AND ?"]
        params = [start_timestamp, end_timestamp]
        
        if filters['mode_type']:
            where_conditions.append("mode_type = ?")
            params.append(filters['mode_type'])
        
        if filters['source']:
            where_conditions.append("source = ?")
            params.append(filters['source'])
        
        if filters['direction']:
            where_conditions.append("direction = ?")
            params.append(filters['direction'])
            
        if filters['run_index']:
            where_conditions.append("run_index = ?")
            params.append(int(filters['run_index']))
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
            SELECT {groupby} as time_period, COUNT(*) as count 
            FROM crossing_events 
            WHERE {where_clause}
            GROUP BY {groupby} 
            ORDER BY time_period
        """
        
        cursor.execute(query, params)
        return cursor.fetchall()
        
    except Exception as e:
        print(f"Error getting filtered grouped counts: {e}")
        return []
    finally:
        db.close()
        
if __name__ == "__main__":
    db = Database()
    
    try:
        conn, cursor = db.get_connection()
        
        query = """
            ALTER TABLE crossing_events RENAME TO counting_events
        """
        
        cursor.execute(query)
        conn.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        db.close()