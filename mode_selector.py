import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import VideoProcessor
import os
import config

from datetime import datetime, date, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
from tkinter import messagebox, ttk
from database_utils import *
from EmbeddedFrame import EmbeddedFrame
from thread_manager import start_threads, thread_controller

COUNT_MODE = "LINE"
CAMERA_SOURCES = VideoProcessor.CAMERA_SOURCES

enter_count = [0 for _ in CAMERA_SOURCES]
exit_count = [0 for _ in CAMERA_SOURCES]
crowd_count = [0 for _ in CAMERA_SOURCES]
total_enter_count = 0
total_exit_count = 0
total_crowd_count = 0

def show_selection_window():
    """
    Creates and runs the selection dialog. When 'Start' is clicked,
    this window is destroyed and start_threads() is called.
    When the counter window later closes, it will re-invoke this function.
    """
    
    # Ensure any previous stop event is set
    thread_controller.stop_event.set()
    
    # Create the main window
    sel = tk.Tk()
    sel.title("Crowd Monitoring System and Analysis")
    sel.geometry("600x500")
    
    # Add padding and styling
    content_frame = tk.Frame(sel, padx=20, pady=20)
    content_frame.pack(fill=tk.BOTH, expand=True)

    # Count type selection
    count_type_frame = tk.LabelFrame(content_frame, text="Counting Type", padx=10, pady=10)
    count_type_frame.pack(fill=tk.X, pady=(0, 10))
    
    count_type_var = tk.StringVar(value="LINE")
    
    tk.Radiobutton(count_type_frame, text="Line Crossing (Standard)", variable=count_type_var, 
                  value="LINE", command=lambda: on_count_type_change("LINE")).pack(anchor='w')
    tk.Radiobutton(count_type_frame, text="Crowd Count (Person in Frame)", variable=count_type_var, 
                  value="CROWD", command=lambda: on_count_type_change("CROWD")).pack(anchor='w')
    
    # Mode selection
    mode_frame = tk.LabelFrame(content_frame, text="Select Mode", padx=10, pady=10)
    mode_frame.pack(fill=tk.X, pady=(0, 10))
    
    mode_var = tk.IntVar(value=0)

    tk.Radiobutton(mode_frame, text="Start Fresh", variable=mode_var, value=0).pack(anchor='w')
    tk.Radiobutton(mode_frame, text="Custom Date Range", variable=mode_var, value=2).pack(anchor='w')

    # Date and time selection for custom range
    date_time_frame = tk.Frame(content_frame)
    date_time_frame.pack(fill=tk.X, pady=10)

    # Start date/time frame
    start_frame = tk.LabelFrame(date_time_frame, text="Start", padx=5, pady=5)
    start_frame.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    
    # Start date
    start_date_entry = DateEntry(start_frame, date_pattern='yyyy-MM-dd')
    start_date_entry.grid(row=0, column=0, padx=5, pady=5)
    
    # Start time
    start_time_frame = tk.Frame(start_frame)
    start_time_frame.grid(row=0, column=1, padx=5, pady=5)
    
    start_hour = tk.Spinbox(start_time_frame, from_=0, to=23, width=2, format="%02.0f")
    start_hour.grid(row=0, column=0)
    start_hour.delete(0, tk.END)
    start_hour.insert(0, "00")
    
    tk.Label(start_time_frame, text=":").grid(row=0, column=1)
    
    start_min = tk.Spinbox(start_time_frame, from_=0, to=59, width=2, format="%02.0f")
    start_min.grid(row=0, column=2)
    start_min.delete(0, tk.END)
    start_min.insert(0, "00")
    
    tk.Label(start_time_frame, text=":").grid(row=0, column=3)
    
    start_sec = tk.Spinbox(start_time_frame, from_=0, to=59, width=2, format="%02.0f")
    start_sec.grid(row=0, column=4)
    start_sec.delete(0, tk.END)
    start_sec.insert(0, "00")
    
    # End date/time frame
    end_frame = tk.LabelFrame(date_time_frame, text="End", padx=5, pady=5)
    end_frame.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    
    # End date
    end_date_entry = DateEntry(end_frame, date_pattern='yyyy-MM-dd')
    end_date_entry.grid(row=0, column=0, padx=5, pady=5)
    
    # End time
    end_time_frame = tk.Frame(end_frame)
    end_time_frame.grid(row=0, column=1, padx=5, pady=5)
    
    end_hour = tk.Spinbox(end_time_frame, from_=0, to=23, width=2, format="%02.0f")
    end_hour.grid(row=0, column=0)
    end_hour.delete(0, tk.END)
    end_hour.insert(0, "23")
    
    tk.Label(end_time_frame, text=":").grid(row=0, column=1)
    
    end_min = tk.Spinbox(end_time_frame, from_=0, to=59, width=2, format="%02.0f")
    end_min.grid(row=0, column=2)
    end_min.delete(0, tk.END)
    end_min.insert(0, "59")
    
    tk.Label(end_time_frame, text=":").grid(row=0, column=3)
    
    end_sec = tk.Spinbox(end_time_frame, from_=0, to=59, width=2, format="%02.0f")
    end_sec.grid(row=0, column=4)
    end_sec.delete(0, tk.END)
    end_sec.insert(0, "59")
    
    # Initially hide the date entries
    date_time_frame.pack_forget()
    
    # Show/hide date entries based on mode selection
    def on_mode_change(*_):
        if mode_var.get() == 2:
            date_time_frame.pack(fill=tk.X, pady=10)
        else:
            date_time_frame.pack_forget()

    mode_var.trace_add('write', on_mode_change)

    # Handle count type change
    def on_count_type_change(mode):
        global COUNT_MODE
        COUNT_MODE = mode

    # Initialize count for custom date range
    def init_custom_counts():
        global enter_count, exit_count, total_enter_count, total_exit_count, crowd_count, total_crowd_count

        try:
            # Get start date and time
            start_date = start_date_entry.get_date()
            start_time = f"{start_hour.get().zfill(2)}:{start_min.get().zfill(2)}:{start_sec.get().zfill(2)}"
            
            # Get end date and time
            end_date = end_date_entry.get_date()
            end_time = f"{end_hour.get().zfill(2)}:{end_min.get().zfill(2)}:{end_sec.get().zfill(2)}"
            
            # Format the complete timestamps
            s = f"{start_date.isoformat()}T{start_time}"
            e = f"{end_date.isoformat()}T{end_time}"

            # Reset counts
            enter_count       = [0 for _ in CAMERA_SOURCES]
            exit_count        = [0 for _ in CAMERA_SOURCES]
            crowd_count       = [0 for _ in CAMERA_SOURCES]
            total_enter_count = 0
            total_exit_count = 0
            total_crowd_count = 0

            if COUNT_MODE == "LINE":
                data = get_total_counts_line_mode(s, e, COUNT_MODE.lower())
                
                for d, c in data:
                    if d == 'enter':
                        total_enter_count = c
                    else:
                        total_exit_count = c

                # Set the first camera's count to the total
                enter_count[0] = total_enter_count
                exit_count[0] = total_exit_count
            elif COUNT_MODE == "CROWD":
                total_crowd_count = get_total_counts_crowd_mode(s, e, COUNT_MODE.lower())
                
                # Set the first camera's count to the total
                crowd_count[0] = total_crowd_count
            
        except Exception as ex:
            messagebox.showerror("Error", f"Failed to get data: {str(ex)}")
            total_enter_count = total_exit_count = total_crowd_count = 0
            enter_count       = [0 for _ in CAMERA_SOURCES]
            exit_count        = [0 for _ in CAMERA_SOURCES]
            crowd_count       = [0 for _ in CAMERA_SOURCES]
        
        VideoProcessor.enter_count       = enter_count.copy()
        VideoProcessor.exit_count        = exit_count.copy()
        VideoProcessor.crowd_count       = crowd_count.copy()
        VideoProcessor.total_enter_count = total_enter_count
        VideoProcessor.total_exit_count  = total_exit_count
        VideoProcessor.total_crowd_count = total_crowd_count

    # Create and show the query window
    def open_query_window():
        query_win = tk.Toplevel(sel)
        query_win.title("Statistic Dashboard")
        query_win.geometry("1000x800")
        query_win.grab_set()  # Make window modal
        
        # Create frames for better organization
        control_frame = tk.Frame(query_win, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        result_frame = tk.Frame(query_win, padx=10)
        result_frame.pack(fill=tk.X)
        
        graph_frame = tk.Frame(query_win, padx=10, pady=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Filtering options
        filter_frame = tk.LabelFrame(control_frame, text="Filters", padx=10, pady=10)
        filter_frame.pack(fill=tk.X, pady=10)
        
        # Mode selection
        mode_frame = tk.Frame(filter_frame)
        mode_frame.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        tk.Label(mode_frame, text="Mode:").grid(row=0, column=0, sticky='w')
        mode_var = tk.StringVar(value="all")
        mode_combo = ttk.Combobox(mode_frame, textvariable=mode_var, values=["all", "crowd", "video"], 
                                 state="readonly", width=10)
        mode_combo.grid(row=0, column=1, padx=5)
        
        # Source selection
        source_frame = tk.Frame(filter_frame)
        source_frame.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        tk.Label(source_frame, text="Source:").grid(row=0, column=0, sticky='w')
        source_var = tk.StringVar(value="all")
        source_combo = ttk.Combobox(source_frame, textvariable=source_var, state="readonly", width=15)
        source_combo.grid(row=0, column=1, padx=5)
        
        # Run Index selection (initially hidden)
        run_index_frame = tk.Frame(filter_frame)
        
        run_index_label = tk.Label(run_index_frame, text="Run:")
        run_index_label.grid(row=0, column=0, sticky='w')
        run_index_var = tk.StringVar(value="all")
        run_index_combo = ttk.Combobox(run_index_frame, textvariable=run_index_var, state="readonly", width=10)
        run_index_combo.grid(row=0, column=1, padx=5)
        
         # Direction selection
        direction_frame = tk.Frame(filter_frame)
        direction_frame.grid(row=0, column=2, padx=10, pady=5, sticky='w')
        
        tk.Label(direction_frame, text="Direction:").grid(row=0, column=0, sticky='w')
        direction_var = tk.StringVar(value="both")
        direction_combo = ttk.Combobox(direction_frame, textvariable=direction_var, 
                                     values=["both", "enter", "exit"], state="readonly", width=10)
        direction_combo.grid(row=0, column=1, padx=5)
        
        # Date and time selection
        date_time_frame = tk.Frame(control_frame)
        date_time_frame.pack(fill=tk.X)
        
        # Start date/time
        start_frame = tk.LabelFrame(date_time_frame, text="Start", padx=5, pady=5)
        start_frame.grid(row=0, column=0, padx=10, pady=5, sticky='w')
        
        start_date_entry = DateEntry(start_frame, date_pattern='yyyy-MM-dd')
        start_date_entry.grid(row=0, column=0, padx=5, pady=5)

        time_frame = tk.Frame(start_frame)
        time_frame.grid(row=0, column=1, padx=5, pady=5)
        
        start_hour = tk.Spinbox(time_frame, from_=0, to=23, width=2, format="%02.0f")
        start_hour.grid(row=0, column=0)
        tk.Label(time_frame, text=":").grid(row=0, column=1)
        start_min = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        start_min.grid(row=0, column=2)
        tk.Label(time_frame, text=":").grid(row=0, column=3)
        start_sec = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        start_sec.grid(row=0, column=4)
        
        # End date/time
        end_frame = tk.LabelFrame(date_time_frame, text="End", padx=5, pady=5)
        end_frame.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        
        end_date_entry = DateEntry(end_frame, date_pattern='yyyy-MM-dd')
        end_date_entry.grid(row=0, column=0, padx=5, pady=5)

        time_frame = tk.Frame(end_frame)
        time_frame.grid(row=0, column=1, padx=5, pady=5)
        
        end_hour = tk.Spinbox(time_frame, from_=0, to=23, width=2, format="%02.0f")
        end_hour.grid(row=0, column=0)
        tk.Label(time_frame, text=":").grid(row=0, column=1)
        end_min = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        end_min.grid(row=0, column=2)
        tk.Label(time_frame, text=":").grid(row=0, column=3)
        end_sec = tk.Spinbox(time_frame, from_=0, to=59, width=2, format="%02.0f")
        end_sec.grid(row=0, column=4)
        
        # Function to populate run index dropdown based on selected video
        def populate_run_indices():
            try:
                current_source = source_var.get()
                if current_source != "all":
                    run_indices = get_video_run_indices(current_source)
                    run_values = ["all"] + [str(idx) for idx in run_indices]
                    run_index_combo['values'] = run_values
                    run_index_combo.set("all")
                else:
                    run_index_combo['values'] = ["all"]
                    run_index_combo.set("all")
            except Exception as e:
                print(f"Error populating run indices: {e}")
                run_index_combo['values'] = ["all"]
                run_index_combo.set("all")
        
        # Function to show/hide run index combobox
        def toggle_run_index_visibility():
            current_mode = mode_var.get()
            current_source = source_var.get()
            print("mode " + current_mode)
            print("source " + current_source)
            
            if current_mode == "video" and current_source != "all":
                print("Should be display grid")
                run_index_frame.grid(row=0, column=2, padx=10, pady=5, sticky='w')
                populate_run_indices()
            else:
                run_index_frame.grid_remove()
                run_index_var.set("all")
        
        # Function to populate source dropdown based on mode
        def populate_sources():
            try:
                current_mode = mode_var.get()
                if current_mode == "video":
                    sources = get_video_names()
                    source_values = ["all"] + sources
                else:
                    sources = get_distinct_sources()
                    source_values = ["all"] + sources
                source_combo['values'] = source_values
                source_combo.set("all")
            except Exception as e:
                print(f"Error populating sources: {e}")
                source_combo['values'] = ["all"]
                source_combo.set("all")
                
        # Function to handle mode change
        def on_mode_change(*args):
            populate_sources()
            toggle_run_index_visibility()
            # Reset to default datetime when changing mode
            if mode_var.get() != "video":
                reset_to_default_datetime()
                
        # Function to handle source change when in video mode
        def on_source_change(*args):
            current_mode = mode_var.get()
            current_source = source_var.get()
            
            if current_mode == "video" and current_source != "all":
                set_video_datetime(current_source)
                
            # Show/hide run index combobox based on selection
            toggle_run_index_visibility()
        
        # Function to reset datetime to default (today)
        def reset_to_default_datetime():
            today = date.today()
            start_date_entry.set_date(today)
            end_date_entry.set_date(today)
            start_hour.delete(0, tk.END)
            start_hour.insert(0, "00")
            start_min.delete(0, tk.END)
            start_min.insert(0, "00")
            start_sec.delete(0, tk.END)
            start_sec.insert(0, "00")
            end_hour.delete(0, tk.END)
            end_hour.insert(0, "23")
            end_min.delete(0, tk.END)
            end_min.insert(0, "59")
            end_sec.delete(0, tk.END)
            end_sec.insert(0, "59")
            
         # Function to set datetime based on video timestamps
        def set_video_datetime(video_name):
            try:
                start_dt, end_dt = get_video_timestamps(video_name)
                if start_dt and end_dt:
                    # Set start datetime
                    start_date_entry.set_date(start_dt.date())
                    start_hour.delete(0, tk.END)
                    start_hour.insert(0, f"{start_dt.hour:02d}")
                    start_min.delete(0, tk.END)
                    start_min.insert(0, f"{start_dt.minute:02d}")
                    start_sec.delete(0, tk.END)
                    start_sec.insert(0, f"{start_dt.second:02d}")
                    
                    # Set end datetime
                    end_date_entry.set_date(end_dt.date())
                    end_hour.delete(0, tk.END)
                    end_hour.insert(0, f"{end_dt.hour:02d}")
                    end_min.delete(0, tk.END)
                    end_min.insert(0, f"{end_dt.minute:02d}")
                    end_sec.delete(0, tk.END)
                    end_sec.insert(0, f"{end_dt.second:02d}")
            except Exception as e:
                print(f"Error setting video datetime: {e}")
                
        # Bind events to dropdowns
        mode_var.trace_add('write', on_mode_change)
        source_var.trace_add('write', on_source_change)
        # Populate sources on window open
        populate_sources()
        
        # Resolution selection
        resolution_frame = tk.LabelFrame(control_frame, text="Time Resolution", padx=5, pady=5)
        resolution_frame.pack(fill=tk.X, pady=10)
        
        resolution_var = tk.StringVar(value="hour")
        
        tk.Radiobutton(resolution_frame, text="Second", variable=resolution_var, value="second").grid(row=0, column=0, padx=10)
        tk.Radiobutton(resolution_frame, text="Minute", variable=resolution_var, value="minute").grid(row=0, column=1, padx=10)
        tk.Radiobutton(resolution_frame, text="Hour", variable=resolution_var, value="hour").grid(row=0, column=2, padx=10)
        tk.Radiobutton(resolution_frame, text="Day", variable=resolution_var, value="day").grid(row=0, column=3, padx=10)
        
        # Visualization options
        visual_frame = tk.LabelFrame(control_frame, text="Visualization", padx=5, pady=5)
        visual_frame.pack(fill=tk.X, pady=10)
        
        visual_var = tk.StringVar(value="bar")
        
        tk.Radiobutton(visual_frame, text="Bar Chart", variable=visual_var, value="bar").grid(row=0, column=0, padx=10)
        tk.Radiobutton(visual_frame, text="Line Chart", variable=visual_var, value="line").grid(row=0, column=1, padx=10)
        tk.Radiobutton(visual_frame, text="Area Chart", variable=visual_var, value="area").grid(row=0, column=2, padx=10)
        
        # Fetch button
        fetch_button = tk.Button(
            control_frame, 
            text="Fetch Data", 
            command=lambda: fetch_data(
                start_date_entry, start_hour, start_min, start_sec,
                end_date_entry, end_hour, end_min, end_sec,
                resolution_var.get(), visual_var.get(),
                mode_var.get(), source_var.get(), direction_var.get(),
                result_label, graph_frame, run_index_var.get()
            ),
            bg="#4CAF50", fg="white", padx=10, pady=5
        )
        fetch_button.pack(pady=10)
        
        # Results display
        result_label = tk.Label(result_frame, text="Total Entries: 0", font=("Helvetica", 12))
        result_label.pack(pady=5)
        
        # Default values - today
        today = date.today()
        start_date_entry.set_date(today)
        end_date_entry.set_date(today)
        start_hour.delete(0, tk.END)
        start_hour.insert(0, "00")
        start_min.delete(0, tk.END)
        start_min.insert(0, "00")
        start_sec.delete(0, tk.END)
        start_sec.insert(0, "00")
        end_hour.delete(0, tk.END)
        end_hour.insert(0, "23")
        end_min.delete(0, tk.END)
        end_min.insert(0, "59")
        end_sec.delete(0, tk.END)
        end_sec.insert(0, "59")
        
        def on_close():
            query_win.grab_release()
            query_win.destroy()
    
        query_win.protocol("WM_DELETE_WINDOW", on_close)

    # Fetch and display data for the query window
    def fetch_data(start_date_entry, start_hour, start_min, start_sec, 
                   end_date_entry, end_hour, end_min, end_sec,
                   resolution, visualization, mode_type, source, direction,
                   result_label, graph_frame, run_index="all"):
        # Get date and time values
        start_date = start_date_entry.get_date()
        start_time = f"{start_hour.get().zfill(2)}:{start_min.get().zfill(2)}:{start_sec.get().zfill(2)}"

        end_date = end_date_entry.get_date()
        end_time = f"{end_hour.get().zfill(2)}:{end_min.get().zfill(2)}:{end_sec.get().zfill(2)}"

        start_timestamp = f"{start_date}T{start_time}"
        end_timestamp = f"{end_date}T{end_time}"
        
        # Apply filters
        filters = {
            'mode_type': mode_type if mode_type != 'all' else None,
            'source': source if source != 'all' else None,
            'direction': direction if direction != 'both' else None,
            'run_index': run_index if run_index != 'all' else None
        }
        
        count = get_total_counts_filtered(start_timestamp, end_timestamp, filters)

        result_label.config(text=f"Total Entries: {count}")

        # Clear previous graph
        for widget in graph_frame.winfo_children():
            widget.destroy()
            
        # Exit if no data
        if count == 0:
            tk.Label(graph_frame, text="No data for the selected period", font=("Helvetica", 12)).pack()
            return

        # Query based on resolution
        title = ""
        groupby = ""
        
        if resolution == "second":
            title = "Entries by Second"
            groupby = "timestamp"
        elif resolution == "minute":
            title = "Entries by Minute"
            groupby = "strftime('%Y-%m-%d %H:%M', timestamp)"
        elif resolution == "hour":
            title = "Hourly Entries"
            groupby = "strftime('%Y-%m-%d %H:00:00', timestamp)"
        else:  # day
            title = "Daily Entries"
            groupby = "DATE(timestamp)"
            
         # Add filter info to title
        filter_info = []
        if filters['mode_type']:
            filter_info.append(f"Mode: {filters['mode_type']}")
        if filters['source']:
            filter_info.append(f"Source: {filters['source']}")
        if filters['direction']:
            filter_info.append(f"Direction: {filters['direction']}")
        if filters['run_index']:
            filter_info.append(f"Run: {filters['run_index']}")
        
        if filter_info:
            title += f" ({', '.join(filter_info)})"
            
        if resolution == "second":
            # For second resolution, get individual timestamps and process them
            data = get_individual_timestamps_filtered(start_timestamp, end_timestamp, filters)
        else:
            data = get_grouped_counts_filtered(start_timestamp, end_timestamp, groupby, filters)
        
        # Convert timestamps to datetime objects for better plotting
        time_periods = []
        counts = []
        
        if resolution == "second":
            # For second resolution, create a complete time series with gaps
            from collections import defaultdict
            
            # Parse start and end times
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            
            # Count occurrences per second
            second_counts = defaultdict(int)
            
            # Process individual timestamps
            for row in data:
                try:
                    if 'T' in row[0]:
                        dt = datetime.fromisoformat(row[0])
                    else:
                        dt = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                    
                    # Truncate to second precision
                    dt_second = dt.replace(microsecond=0)
                    second_counts[dt_second] += 1
                except ValueError as e:
                    print(f"Error parsing timestamp '{row[0]}': {e}")
                    continue
            
            # Create complete time series from start to end (every second)
            current_time = start_dt.replace(microsecond=0)
            end_time_truncated = end_dt.replace(microsecond=0)
            
            while current_time <= end_time_truncated:
                time_periods.append(current_time)
                counts.append(second_counts.get(current_time, 0))  # 0 for gaps
                current_time += timedelta(seconds=1)
                
        elif resolution == "minute":
            # For minute resolution, create complete time series
            
            # Parse start and end times
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            
            # Create dictionary from existing data
            minute_counts = {}
            for row in data:
                try:
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M")
                    minute_counts[dt] = row[1]
                except ValueError as e:
                    print(f"Error parsing datetime '{row[0]}': {e}")
                    continue
            
            # Create complete time series (every minute)
            current_time = start_dt.replace(second=0, microsecond=0)
            end_time_truncated = end_dt.replace(second=0, microsecond=0)
            
            while current_time <= end_time_truncated:
                time_periods.append(current_time)
                counts.append(minute_counts.get(current_time, 0))  # 0 for gaps
                current_time += timedelta(minutes=1)
                
        elif resolution == "hour":
            # For hour resolution, create complete time series
            
            # Parse start and end times
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            
            # Create dictionary from existing data
            hour_counts = {}
            for row in data:
                try:
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:00:00")
                    hour_counts[dt] = row[1]
                except ValueError as e:
                    print(f"Error parsing datetime '{row[0]}': {e}")
                    continue
            
            # Create complete time series (every hour)
            current_time = start_dt.replace(minute=0, second=0, microsecond=0)
            end_time_truncated = end_dt.replace(minute=0, second=0, microsecond=0)
            
            while current_time <= end_time_truncated:
                time_periods.append(current_time)
                counts.append(hour_counts.get(current_time, 0))  # 0 for gaps
                current_time += timedelta(hours=1)
                
        else:  # day resolution
            # For day resolution, create complete time series
            
            # Parse start and end times
            start_dt = datetime.fromisoformat(start_timestamp)
            end_dt = datetime.fromisoformat(end_timestamp)
            
            # Create dictionary from existing data
            day_counts = {}
            for row in data:
                try:
                    dt = datetime.strptime(row[0], "%Y-%m-%d")
                    day_counts[dt] = row[1]
                except ValueError as e:
                    print(f"Error parsing datetime '{row[0]}': {e}")
                    continue
            
            # Create complete time series (every day)
            current_date = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date_truncated = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            
            while current_date <= end_date_truncated:
                time_periods.append(current_date)
                counts.append(day_counts.get(current_date, 0))  # 0 for gaps
                current_date += timedelta(days=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if visualization == "bar":
            bars = ax.bar(time_periods, counts, width=0.8)
            
            # Color bars differently for zero values (gaps)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                if count == 0:
                    bar.set_color('lightgray')
                    bar.set_alpha(0.3)
                else:
                    bar.set_color('steelblue')
                    
        elif visualization == "line":
            # ax.plot(time_periods, counts, linestyle='-', linewidth=1)
            # For line charts, use None for gaps to create actual breaks in the line
            display_counts = [count if count > 0 else None for count in counts]
            ax.plot(time_periods, display_counts, linestyle='-', linewidth=2, marker='o', markersize=3)
        
        elif visualization == "area":
            # ax.fill_between(time_periods, counts, alpha=0.4)
            # ax.plot(time_periods, counts, linestyle='-', linewidth=1)
            # For area charts, use 0 for gaps but make them visible
            ax.fill_between(time_periods, counts, alpha=0.4, step='mid')
            ax.plot(time_periods, counts, linestyle='-', linewidth=1, marker='o', markersize=2)
        
        # FIXED: Smart tick handling to prevent overflow
        num_points = len(time_periods)
        max_ticks = 50  # Conservative limit to prevent overflow
        
        # Calculate appropriate tick intervals
        if resolution == "second":
            if num_points <= 60:  # Less than 1 minute
                interval = max(1, num_points // 20)
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            elif num_points <= 3600:  # Less than 1 hour
                interval = max(30, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            else:  # More than 1 hour
                interval = max(300, num_points // max_ticks)  # At least 5 minutes
                ax.xaxis.set_major_locator(mdates.SecondLocator(interval=interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
        elif resolution == "minute":
            if num_points <= 60:  # Less than 1 hour
                interval = max(1, num_points // 20)
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
            elif num_points <= 1440:  # Less than 1 day
                interval = max(30, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
            else:  # More than 1 day
                interval = max(60, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
        elif resolution == "hour":
            if num_points <= 24:  # Less than 1 day
                interval = max(1, num_points // 12)
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif num_points <= 168:  # Less than 1 week
                interval = max(6, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            else:  # More than 1 week
                interval = max(24, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            
        else:  # day resolution
            if num_points <= 31:  # Less than 1 month
                interval = max(1, num_points // 15)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            elif num_points <= 365:  # Less than 1 year
                interval = max(7, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            else:  # More than 1 year
                interval = max(30, num_points // max_ticks)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        # Always rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Entries')
        ax.set_title(title)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start from 0 for better gap visualization
        ax.set_ylim(bottom=0)
        
        # Add statistics text
        if len(counts) > 0:
            total_events = sum(counts)
            max_count = max(counts) if counts else 0
            gap_periods = counts.count(0)
            active_periods = len(counts) - gap_periods
            
            stats_text = f"Total Events: {total_events} | Max/Period: {max_count} | Active Periods: {active_periods}/{len(counts)}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        # Adjust layout
        plt.tight_layout()
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Close the matplotlib figure to prevent memory leaks
        plt.close(fig)

    # Handle start button click
    def on_start():
        global enter_count, exit_count, total_enter_count, total_exit_count, crowd_count, total_crowd_count
        m = mode_var.get()
        if m == 0:
            # Start fresh
            VideoProcessor.enter_count = [0 for _ in CAMERA_SOURCES]
            VideoProcessor.exit_count = [0 for _ in CAMERA_SOURCES]
            VideoProcessor.crowd_count = [0 for _ in CAMERA_SOURCES]
            VideoProcessor.total_enter_count = 0
            VideoProcessor.total_exit_count  = 0
            VideoProcessor.total_crowd_count = 0

        elif m == 2:
            # Custom date range
            try:
                init_custom_counts()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading custom date range: {e}")
                return
        
        # Hide selection window and start threads
        sel.withdraw()  # Hide instead of destroy
        
        
        # Start the threads
        # thread_controller.reset()
        start_threads()
        
        # Define callback for when counter window closes
        def on_counting_close():
            # Stop all threads
            thread_controller.stop_event.set()
            # join threads
            for t in thread_controller.threads:
                if t.is_alive():
                    t.join(timeout=1.0)
            # Re‐show the selection window
            try: sel.deiconify()
            except tk.TclError: show_selection_window()
        
        # Create a new window for the detector UI
        crowd_win = tk.Toplevel(sel)
        crowd_win.title("Crowd Detection")
        # Pass either source_index=0 or loop for multiple sources
        app = EmbeddedFrame(crowd_win, source_index=0, on_close=on_counting_close)
        

    # Add buttons to selection window
    button_frame = tk.Frame(content_frame)
    button_frame.pack(fill=tk.X, pady=10)

    query_button = tk.Button(
        button_frame, 
        text="Statistic Dashboard", 
        command=open_query_window,
        width=15
    )
    query_button.pack(side=tk.LEFT, padx=5)

    start_button = tk.Button(
        button_frame, 
        text="Start", 
        command=on_start,
        width=15
    )
    start_button.pack(side=tk.RIGHT, padx=5)
    
    # Video Analysis button
    model_button = tk.Button(
        button_frame,
        text="Model Setting",
        command=lambda: open_model_setting(sel),
        width=15
    )
    model_button.pack(side=tk.LEFT, padx=5)
    
    # Video Analysis button
    video_button = tk.Button(
        button_frame,
        text="Video Analysis",
        command=lambda: open_video_analysis(sel),
        width=15
    )
    video_button.pack(side=tk.LEFT, padx=5)

    # Handle window close properly
    def on_close():
        # Clean up database connection
        db = Database()
        db.close()
        # Destroy the window
        sel.destroy()

    sel.protocol("WM_DELETE_WINDOW", on_close)
    
    # Start the main loop for selection window
    sel.mainloop()

def open_model_setting(sel):
    import tracker_config
    # Hide main menu
    sel.withdraw()

    def on_close():
        # When this window closes, re-show menu
        try:
            sel.deiconify()
        except tk.TclError:
            # if menu was destroyed, recreate
            show_selection_window()
        va_win.destroy()

    va_win = tk.Toplevel(sel)
    va_win.title("Model & Tracker Setting")
    va_win.geometry("400x600")
    va_win.protocol("WM_DELETE_WINDOW", on_close)
    
    # --- Model Defaults Section ---
    section = tk.LabelFrame(va_win, text="Model Settings", padx=10, pady=10)
    section.pack(fill="x", padx=10, pady=(10,5))

    # --- Class Selection ---
    tk.Label(section, text="Select Class:").grid(row=0, column=0, sticky="w")
    class_var = tk.StringVar(value="head")
    ttk.OptionMenu(section, class_var, "head", "head", "person").grid(row=0, column=1, sticky="ew")

    # --- Model Selection ---
    tk.Label(section, text="Select Model:").grid(row=1, column=0, sticky="w", pady=(5,0))
    model_var = tk.StringVar(value=config.get_model_name())
    head_models   = ["headv1.pt", "headv2.pt", "headv3.pt"]
    person_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"]
    model_menu = ttk.OptionMenu(section, model_var, model_var.get(), *head_models)
    model_menu.grid(row=1, column=1, sticky="ew", pady=(5,0))

    def refresh_models(*_):
        opts = head_models if class_var.get()=="head" else person_models
        model_var.set(opts[0])
        menu = model_menu["menu"]
        menu.delete(0, "end")
        for m in opts:
            menu.add_command(label=m, command=tk._setit(model_var, m))
    class_var.trace_add("write", refresh_models)

     # Confidence & IoU
    tk.Label(section, text="Confidence:").grid(row=2, column=0, sticky="w", pady=(10,0))
    conf_entry = tk.Entry(section); conf_entry.grid(row=2, column=1, sticky="ew", pady=(10,0))
    conf_entry.insert(0, str(config.get_model_conf()))

    tk.Label(section, text="IoU:").grid(row=3, column=0, sticky="w", pady=(5,0))
    iou_entry = tk.Entry(section); iou_entry.grid(row=3, column=1, sticky="ew", pady=(5,0))
    iou_entry.insert(0, str(config.get_model_iou()))

    for col in (0,1):
        section.grid_columnconfigure(col, weight=1)


    # --- Tracker Settings Section ---
    tsec = tk.LabelFrame(va_win, text="Tracker Settings", padx=10, pady=10)
    tsec.pack(fill="x", padx=10, pady=(5,10))

    # Load current tracker yaml to pre-populate
    current = tracker_config.get_tracker_settings()

    # Tracker Type
    tk.Label(tsec, text="Type:").grid(row=0, column=0, sticky="w")
    tracker_type_var = tk.StringVar(value=current.get("tracker_type"))
    ttk.OptionMenu(tsec, tracker_type_var, tracker_type_var.get(), "botsort", "bytetrack").grid(row=0, column=1, sticky="ew")

    # High / Low / New / Buffer / Match / Fuse
    labels = [
        ("High Thresh", "track_high_thresh"),
        ("Low Thresh",  "track_low_thresh"),
        ("New Thresh",  "new_track_thresh"),
        ("Buffer",      "track_buffer"),
        ("Match Thresh","match_thresh"),
    ]
    entries = {}
    for i, (lbl, key) in enumerate(labels, start=1):
        tk.Label(tsec, text=lbl+":").grid(row=i, column=0, sticky="w", pady=(5,0))
        e = tk.Entry(tsec); e.grid(row=i, column=1, sticky="ew", pady=(5,0))
        e.insert(0, str(current.get(key,"")))
        entries[key] = e

    # Fuse score checkbox
    fuse_var = tk.BooleanVar(value=bool(current.get("fuse_score", True)))
    tk.Checkbutton(tsec, text="Fuse Score", variable=fuse_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=(5,0))

    # GMC / Proximity / Appearance / with_reid / model
    extra = [
        ("GMC Method",       "gmc_method"),
        ("Proximity Thresh", "proximity_thresh"),
        ("Appearance Thresh","appearance_thresh"),
    ]
    extra_entries = {}
    for i, (lbl, key) in enumerate(extra, start=7):
        tk.Label(tsec, text=lbl+":").grid(row=i, column=0, sticky="w", pady=(5,0))
        e = tk.Entry(tsec); e.grid(row=i, column=1, sticky="ew", pady=(5,0))
        e.insert(0, str(current.get(key,"")))
        extra_entries[key] = e

    reid_var = tk.BooleanVar(value=bool(current.get("with_reid", True)))
    tk.Checkbutton(tsec, text="Use ReID", variable=reid_var).grid(row=10, column=0, columnspan=2, sticky="w", pady=(5,0))

    model_tracker_var = tk.StringVar(value=current.get("model","auto"))
    tk.Label(tsec, text="ReID Model:").grid(row=11, column=0, sticky="w", pady=(5,0))
    ttk.OptionMenu(tsec, model_tracker_var, model_tracker_var.get(), "auto").grid(row=11, column=1, sticky="ew", pady=(5,0))

    for col in (0,1):
        tsec.grid_columnconfigure(col, weight=1)


    # --- Buttons ---
    btns = tk.Frame(va_win)
    btns.pack(fill="x", pady=10, padx=10)
    tk.Button(btns, text="Reset", width=10, command=lambda: do_reset()).pack(side=tk.LEFT)
    tk.Button(btns, text="Update Changes", width=15, command=lambda: do_update()).pack(side=tk.RIGHT)

    def do_reset():
        # model defaults
        model_var.set(config.get_model_name())
        conf_entry.delete(0, tk.END); conf_entry.insert(0, str(config.get_model_conf()))
        iou_entry.delete(0, tk.END); iou_entry.insert(0, str(config.get_model_iou()))
        # tracker defaults
        defaults = tracker_config.reset_tracker_to_defaults()
        tracker_type_var.set(defaults["tracker_type"])
        for key, e in entries.items():
            e.delete(0, tk.END); e.insert(0, str(defaults.get(key, "")))
        fuse_var.set(bool(defaults.get("fuse_score", True)))
        for key, e in extra_entries.items():
            e.delete(0, tk.END); e.insert(0, str(defaults.get(key, "")))
        reid_var.set(bool(defaults.get("with_reid", True)))
        model_tracker_var.set(defaults.get("model","auto"))

    def do_update():
        # collect model settings
        new_model = model_var.get()
        new_conf  = float(conf_entry.get())
        new_iou   = float(iou_entry.get())

        # collect tracker settings
        new_tracker = {
            "tracker_type":       tracker_type_var.get(),
            "track_high_thresh":  float(entries["track_high_thresh"].get()),
            "track_low_thresh":   float(entries["track_low_thresh"].get()),
            "new_track_thresh":   float(entries["new_track_thresh"].get()),
            "track_buffer":       int(entries["track_buffer"].get()),
            "match_thresh":       float(entries["match_thresh"].get()),
            "fuse_score":         fuse_var.get(),
            "gmc_method":         extra_entries["gmc_method"].get(),
            "proximity_thresh":   float(extra_entries["proximity_thresh"].get()),
            "appearance_thresh":  float(extra_entries["appearance_thresh"].get()),
            "with_reid":          reid_var.get(),
            "model":              model_tracker_var.get()
        }

        # show confirmation
        summary = (
            f"Model → {new_model}\n"
            f"  conf: {new_conf}, iou: {new_iou}\n\n"
            f"Tracker → {new_tracker['tracker_type']}\n"
            f"  high:{new_tracker['track_high_thresh']} low:{new_tracker['track_low_thresh']}\n"
            f"  new:{new_tracker['new_track_thresh']} buf:{new_tracker['track_buffer']}\n"
            f"  match:{new_tracker['match_thresh']} fuse:{new_tracker['fuse_score']}\n"
            f"  GMC:{new_tracker['gmc_method']}\n"
            f"  prox:{new_tracker['proximity_thresh']} app:{new_tracker['appearance_thresh']}\n"
            f"  with_reid:{new_tracker['with_reid']} model:{new_tracker['model']}\n"
        )
        if not messagebox.askokcancel("Confirm Changes", summary):
            return

        # write changes
        config.set_model_name(new_model)
        config.set_model_conf(new_conf)
        config.set_model_iou(new_iou)
        tracker_config.set_tracker_settings(new_tracker)

        # close and back to menu
        on_close()
        
def open_video_analysis(sel):
    from VideoAnalysisFrame import VideoAnalysisFrame
    # Hide main menu
    sel.withdraw()

    def on_close():
        # When this window closes, re-show menu
        try:
            sel.deiconify()
        except tk.TclError:
            # if menu was destroyed, recreate
            show_selection_window()
        va_win.destroy()

    # Create Video Analysis window
    va_win = tk.Toplevel(sel)
    va_win.title("Video Analysis")
    va_win.geometry("350x280")
    va_win.protocol("WM_DELETE_WINDOW", on_close)
    
    # --- Video Selection ---
    tk.Label(va_win, text="Select Video:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15,5), padx=15)
    video_files = [f for f in os.listdir("video") if f.lower().endswith((".mp4", ".avi"))]
    video_var = tk.StringVar(value=video_files[0] if video_files else "")
    video_menu = ttk.OptionMenu(va_win, video_var, video_var.get(), *video_files)
    video_menu.pack(fill="x", padx=15, pady=(0,10))
    
    # --- Ground Truth Count Input ---
    ground_truth_frame = tk.Frame(va_win)
    ground_truth_frame.pack(fill="x", padx=15, pady=(0,10))
    
    tk.Label(ground_truth_frame, text="Ground Truth Count:", font=("Arial", 10, "bold")).pack(anchor="w")
    
    # Create frame for input and info
    input_frame = tk.Frame(ground_truth_frame)
    input_frame.pack(fill="x", pady=(5,0))
    
    ground_truth_var = tk.StringVar(value="")
    ground_truth_entry = tk.Entry(input_frame, textvariable=ground_truth_var, width=10)
    ground_truth_entry.pack(side="left")
    
    # Info label
    info_label = tk.Label(input_frame, text="(Expected number of people/objects)", 
                         font=("Arial", 8), fg="gray")
    info_label.pack(side="left", padx=(10,0))
    
    # Optional checkbox
    optional_frame = tk.Frame(ground_truth_frame)
    optional_frame.pack(fill="x", pady=(5,0))
    
    use_ground_truth_var = tk.BooleanVar(value=False)
    optional_check = tk.Checkbutton(optional_frame, 
                                text="Enable performance metrics calculation", 
                                variable=use_ground_truth_var,
                                command=lambda: toggle_ground_truth_input()
                                )
    optional_check.pack(anchor="w")
    
    def toggle_ground_truth_input():
        if use_ground_truth_var.get():
            ground_truth_entry.config(state="normal")
            info_label.config(fg="black")
        else:
            ground_truth_entry.config(state="disabled")
            info_label.config(fg="gray")
            ground_truth_var.set("")
    
    # Initially disable ground truth input
    toggle_ground_truth_input()
    
    # --- Run Index Information ---
    run_info_frame = tk.Frame(va_win)
    run_info_frame.pack(fill="x", padx=15, pady=(0,15))
    
    tk.Label(run_info_frame, text="Run Information:", font=("Arial", 10, "bold")).pack(anchor="w")
    
    run_index_label = tk.Label(run_info_frame, text="", font=("Arial", 9), fg="blue")
    run_index_label.pack(anchor="w", pady=(2,0))
    
    def update_run_info(*args):
        selected_video = video_var.get()
        if selected_video:
            try:
                next_run = get_next_run_index(selected_video)
                existing_runs = get_analysis_comparison(selected_video)
                
                if existing_runs:
                    run_info_text = f"This will be Run #{next_run} (Previous runs: {len(existing_runs)})"
                else:
                    run_info_text = f"This will be Run #{next_run} (First analysis of this video)"
                
                run_index_label.config(text=run_info_text)
            except Exception as e:
                run_index_label.config(text=f"Run index: {1} (Could not check previous runs)")
        else:
            run_index_label.config(text="")
    
    # Update run info when video selection changes
    video_var.trace_add("write", update_run_info)
    update_run_info()  # Initial update
    
    # Buttons: Start analysis & Cancel
    btn_frame = tk.Frame(va_win)
    btn_frame.pack(fill="x", pady=15, padx=15)
    
    tk.Button(btn_frame, text="Cancel", width=12, command=on_close).pack(side=tk.LEFT)
    tk.Button(btn_frame, text="Start Analysis", width=15, command=lambda: on_submit()).pack(side=tk.RIGHT)
    
    def validate_input():
        """Validate user input before starting analysis"""
        if not video_var.get():
            tk.messagebox.showerror("Error", "Please select a video file.")
            return False
        
        if use_ground_truth_var.get():
            try:
                ground_truth_value = ground_truth_var.get().strip()
                if ground_truth_value:
                    ground_truth_count = int(ground_truth_value)
                    if ground_truth_count < 0:
                        tk.messagebox.showerror("Error", "Ground truth count must be a positive number.")
                        return False
                else:
                    tk.messagebox.showerror("Error", "Please enter a ground truth count or disable performance metrics.")
                    return False
            except ValueError:
                tk.messagebox.showerror("Error", "Ground truth count must be a valid number.")
                return False
        
        return True
    
    def on_submit():
        if not validate_input():
            return
        
        # Get ground truth count
        ground_truth_count = None
        if use_ground_truth_var.get() and ground_truth_var.get().strip():
            try:
                ground_truth_count = int(ground_truth_var.get().strip())
            except ValueError:
                ground_truth_count = None
        
        # Get run index
        try:
            run_index = get_next_run_index(video_var.get())
        except:
            run_index = 1
        
        # Hide selection window and start threads
        va_win.destroy()
 
        # Start the threads
        # thread_controller.reset()
        start_threads()
        
        # Define callback for when counter window closes
        def on_va_close():
            # Stop all threads
            thread_controller.stop_event.set()
            # join threads
            for t in thread_controller.threads:
                if t.is_alive():
                    t.join(timeout=1.0)
            # Re‐show the selection window
            try: 
                sel.deiconify()
            except tk.TclError: 
                show_selection_window()
        
        # Create a new window for the detector UI
        video_analysis = tk.Toplevel(sel)
        video_analysis.title(f"Video Analysis - {video_var.get()} (Run #{run_index})")
        
        # Pass video path, ground truth count, and run index
        app = VideoAnalysisFrame(
            video_analysis, 
            video_path=video_var.get(), 
            on_close=on_va_close,
            ground_truth_count=ground_truth_count,
            run_index=run_index
        )
        
