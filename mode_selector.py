import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import VideoProcessor

from datetime import datetime, date
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
from tkinter import messagebox
from database_utils import *
from EmbeddedFrame import EmbeddedFrame

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

    from thread_manager import start_threads, thread_controller

    # Ensure any previous stop event is set
    thread_controller.stop_event.set()
    
    # Create the main window
    sel = tk.Tk()
    sel.title("People Counter - Select Counting Mode")
    sel.geometry("500x500")
    
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
        query_win.title("Query People Entering")
        query_win.geometry("900x700")
        query_win.grab_set()  # Make window modal
        
        # Create frames for better organization
        control_frame = tk.Frame(query_win, padx=10, pady=10)
        control_frame.pack(fill=tk.X)
        
        result_frame = tk.Frame(query_win, padx=10)
        result_frame.pack(fill=tk.X)
        
        graph_frame = tk.Frame(query_win, padx=10, pady=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
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
                result_label, graph_frame
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
                   resolution, visualization,
                   result_label, graph_frame):
        # Get date and time values
        start_date = start_date_entry.get_date()
        start_time = f"{start_hour.get().zfill(2)}:{start_min.get().zfill(2)}:{start_sec.get().zfill(2)}"

        end_date = end_date_entry.get_date()
        end_time = f"{end_hour.get().zfill(2)}:{end_min.get().zfill(2)}:{end_sec.get().zfill(2)}"

        start_timestamp = f"{start_date}T{start_time}"
        end_timestamp = f"{end_date}T{end_time}"

        count = get_total_counts(start_timestamp, end_timestamp)

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
            groupby = "strftime('%Y-%m-%d %H:%M:%S', timestamp)"
        elif resolution == "minute":
            title = "Entries by Minute"
            groupby = "strftime('%Y-%m-%d %H:%M', timestamp)"
        elif resolution == "hour":
            title = "Hourly Entries"
            groupby = "strftime('%Y-%m-%d %H:00:00', timestamp)"
        else:  # day
            title = "Daily Entries"
            groupby = "DATE(timestamp)"

        data = get_grouped_counts(start_timestamp, end_timestamp, groupby)
        
        # Convert timestamps to datetime objects for better plotting
        time_periods = []
        counts = []
        
        for row in data:
            try:
                if resolution == "day":
                    dt = datetime.strptime(row[0], "%Y-%m-%d")
                elif resolution == "hour":
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:00:00")
                elif resolution == "minute":
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M")
                else:  # second
                    dt = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            except ValueError as e:
                print(f"Error parsing datetime '{row[0]}': {e}")
                continue
            time_periods.append(dt)
            counts.append(row[1])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if visualization == "bar":
            ax.bar(time_periods, counts, width=0.8)
        elif visualization == "line":
            ax.plot(time_periods, counts, linestyle='-', linewidth=1)
        elif visualization == "area":
            ax.fill_between(time_periods, counts, alpha=0.4)
            ax.plot(time_periods, counts, linestyle='-', linewidth=1)
        
        # Format x-axis based on resolution
        if resolution == "second" or resolution == "minute":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
        elif resolution == "hour":
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
        else:  # day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Entries')
        ax.set_title(title)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
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
        text="Query Past Data", 
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
