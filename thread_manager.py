import threading

from ThreadController import ThreadController
from config import get_camera_sources
from database_utils import insert_to_db

CAMERA_SOURCES = get_camera_sources()

thread_controller = ThreadController()

# Start the tracking threads
def start_threads():
    thread_controller.reset()

    db_thread = threading.Thread(target=insert_to_db,args=(thread_controller,), daemon=True)
    db_thread.start()
    thread_controller.threads.append(db_thread)
