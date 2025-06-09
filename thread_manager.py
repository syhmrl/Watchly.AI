import threading

from ThreadController import ThreadController
from config import get_camera_sources
from database_utils import insert_to_db

CAMERA_SOURCES = get_camera_sources()

thread_controller = ThreadController()

# # Start the tracking threads
# def start_threads(mode, on_session_end=None):

#     from VideoProcessor import capture_frames, video_processing_dispatcher

#     # Reset the thread controller
#     thread_controller.reset()
    
#     # Create and start capture threads for each camera
#     for i, source in enumerate(CAMERA_SOURCES):
#         if source is not None:  # Only start threads for defined sources
#             # Capture thread
#             capture_thread = threading.Thread(target=capture_frames, args=(i,), daemon=True)
#             capture_thread.start()
#             thread_controller.threads.append(capture_thread)
            
#             # Processing thread for each camera - use correct mode
#             if mode == "LINE":
#                 video_thread = threading.Thread(target=video_processing_dispatcher, args=("LINE", i, on_session_end,), daemon=True)
#             elif mode == "CROWD":
#                 video_thread = threading.Thread(target=video_processing_dispatcher, args=("CROWD", i, on_session_end,), daemon=True)

#             video_thread.start()
#             thread_controller.threads.append(video_thread)

#     db_thread = threading.Thread(target=insert_to_db,args=(thread_controller,), daemon=True)
#     db_thread.start()
#     thread_controller.threads.append(db_thread)

# Start the tracking threads
def start_threads():
    thread_controller.reset()
    
    # Create and start capture threads for each camera
    # for i, source in enumerate(CAMERA_SOURCES):
    #     if source is not None:  # Only start threads for defined sources
    #         # Capture thread
    #         capture_thread = threading.Thread(target=capture_frames, args=(i,), daemon=True)
    #         capture_thread.start()
    #         thread_controller.threads.append(capture_thread)

    db_thread = threading.Thread(target=insert_to_db,args=(thread_controller,), daemon=True)
    db_thread.start()
    thread_controller.threads.append(db_thread)
