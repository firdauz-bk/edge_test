import tkinter as tk
from tkinter import ttk
import requests
import json
import threading
import time
from datetime import datetime
import webbrowser

class SmartDoorDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Door Lock Dashboard")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Server URL
        self.server_url = "http://localhost:8080"
        
        # Variables
        self.system_mode = tk.StringVar(value="Unknown")
        self.door_status = tk.StringVar(value="Unknown")
        self.presence_status = tk.StringVar(value="No")
        self.wake_word_status = tk.StringVar(value="No")
        self.face_status = tk.StringVar(value="No")
        self.last_event = tk.StringVar(value="None")
        self.last_event_time = tk.StringVar(value="None")
        
        # Create the dashboard UI
        self.create_ui()
        
        # Start polling for status updates
        self.status_polling_active = True
        self.poll_thread = threading.Thread(target=self.poll_status, daemon=True)
        self.poll_thread.start()
    
    def create_ui(self):
        # Create a tabbed interface
        self.tab_control = ttk.Notebook(self.root)
        
        # Main dashboard tab
        self.main_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.main_tab, text="Dashboard")
        
        # Event log tab
        self.log_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.log_tab, text="Event Log")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Configure rows and columns in main tab
        for i in range(6):
            self.main_tab.grid_rowconfigure(i, weight=1)
        self.main_tab.grid_columnconfigure(0, weight=1)
        self.main_tab.grid_columnconfigure(1, weight=1)
        
        # Status section
        status_frame = ttk.LabelFrame(self.main_tab, text="System Status")
        status_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", rowspan=3)
        
        # System mode
        ttk.Label(status_frame, text="System Mode:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.mode_label = ttk.Label(status_frame, textvariable=self.system_mode, font=("Arial", 12, "bold"))
        self.mode_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Door status
        ttk.Label(status_frame, text="Door Status:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.door_label = ttk.Label(status_frame, textvariable=self.door_status, font=("Arial", 12, "bold"))
        self.door_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Presence detection
        ttk.Label(status_frame, text="Presence Detected:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.presence_label = ttk.Label(status_frame, textvariable=self.presence_status)
        self.presence_label.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Wake word detection
        ttk.Label(status_frame, text="Wake Word Detected:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.wake_word_label = ttk.Label(status_frame, textvariable=self.wake_word_status)
        self.wake_word_label.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Face recognition
        ttk.Label(status_frame, text="Face Recognized:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.face_label = ttk.Label(status_frame, textvariable=self.face_status)
        self.face_label.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Last event
        ttk.Label(status_frame, text="Last Event:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.event_label = ttk.Label(status_frame, textvariable=self.last_event)
        self.event_label.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        
        # Last event time
        ttk.Label(status_frame, text="Time:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.event_time_label = ttk.Label(status_frame, textvariable=self.last_event_time)
        self.event_time_label.grid(row=6, column=1, padx=5, pady=5, sticky="w")
        
        # Controls section
        controls_frame = ttk.LabelFrame(self.main_tab, text="System Controls")
        controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew", rowspan=2)
        
        # Reset system button
        self.reset_button = ttk.Button(controls_frame, text="Reset System", command=self.reset_system)
        self.reset_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Register face button
        self.register_button = ttk.Button(controls_frame, text="Register Face", command=self.register_face)
        self.register_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Unlock door button
        self.unlock_button = ttk.Button(controls_frame, text="Unlock Door", command=self.unlock_door)
        self.unlock_button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        
        # Connection status
        self.connection_status = ttk.Label(self.main_tab, text="Not connected", foreground="red")
        self.connection_status.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="s")
        
        # Server URL entry
        server_frame = ttk.Frame(self.main_tab)
        server_frame.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(server_frame, text="Server URL:").pack(side=tk.LEFT, padx=5)
        self.server_entry = ttk.Entry(server_frame, width=30)
        self.server_entry.insert(0, self.server_url)
        self.server_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(server_frame, text="Connect", command=self.update_server_url).pack(side=tk.LEFT, padx=5)
        
        # Event log
        self.log_frame = ttk.Frame(self.log_tab)
        self.log_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create a treeview
        self.event_tree = ttk.Treeview(self.log_frame, columns=("Time", "Event", "Details"), show="headings")
        self.event_tree.heading("Time", text="Time")
        self.event_tree.heading("Event", text="Event")
        self.event_tree.heading("Details", text="Details")
        
        self.event_tree.column("Time", width=150)
        self.event_tree.column("Event", width=150)
        self.event_tree.column("Details", width=300)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.log_frame, orient="vertical", command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.event_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Clear log button
        ttk.Button(self.log_tab, text="Clear Log", command=self.clear_log).pack(pady=10)
    
    def update_ui(self, status_data):
        try:
            # Update connection status
            self.connection_status.config(text="Connected", foreground="green")
            
            # Update variables
            self.system_mode.set(status_data.get("system_mode", "Unknown"))
            self.door_status.set("Locked" if status_data.get("door_locked", True) else "Unlocked")
            self.presence_status.set("Yes" if status_data.get("presence_detected", False) else "No")
            self.wake_word_status.set("Yes" if status_data.get("wake_word_detected", False) else "No")
            self.face_status.set("Yes" if status_data.get("face_recognized", False) else "No")
            
            # Update last event
            last_event = status_data.get("last_event", "")
            last_event_time = status_data.get("last_event_time", "")
            
            if last_event and last_event != self.last_event.get():
                self.last_event.set(last_event)
                self.last_event_time.set(last_event_time)
                
                # Add to event log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.event_tree.insert("", 0, values=(timestamp, last_event, f"Mode: {self.system_mode.get()}"))
            
            # Update status indicator colors
            self.door_label.config(foreground="red" if self.door_status.get() == "Locked" else "green")
            
            if self.system_mode.get() == "Idle":
                self.mode_label.config(foreground="blue")
            elif "Listening" in self.system_mode.get():
                self.mode_label.config(foreground="orange")
            elif "Scanning" in self.system_mode.get():
                self.mode_label.config(foreground="purple")
            elif "unlocked" in self.system_mode.get().lower():
                self.mode_label.config(foreground="green")
            else:
                self.mode_label.config(foreground="black")
            
        except Exception as e:
            print(f"Error updating UI: {e}")
    
    def poll_status(self):
        while self.status_polling_active:
            try:
                response = requests.get(f"{self.server_url}/api/status", timeout=2)
                if response.status_code == 200:
                    status_data = response.json()
                    # Schedule UI update on the main thread
                    self.root.after(0, lambda: self.update_ui(status_data))
                else:
                    self.root.after(0, lambda: self.connection_status.config(
                        text=f"Error: Status code {response.status_code}", foreground="red"))
            except requests.RequestException as e:
                self.root.after(0, lambda: self.connection_status.config(
                    text=f"Connection error: {str(e)}", foreground="red"))
            
            # Poll every second
            time.sleep(1)
    
    def send_command(self, action):
        try:
            response = requests.get(f"{self.server_url}/api/command?action={action}", timeout=2)
            if response.status_code == 200:
                result = response.json()
                
                # Add to event log
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.event_tree.insert("", 0, values=(timestamp, f"Command: {action}", result.get("message", "")))
                
                return True
            else:
                print(f"Error sending command: Status code {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"Error sending command: {e}")
            return False
    
    def reset_system(self):
        self.send_command("reset")
    
    def register_face(self):
        self.send_command("register_face")
    
    def unlock_door(self):
        self.send_command("unlock")
    
    def update_server_url(self):
        new_url = self.server_entry.get()
        if new_url:
            self.server_url = new_url
            self.connection_status.config(text="Connecting...", foreground="orange")
    
    def clear_log(self):
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
    
    def on_closing(self):
        self.status_polling_active = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartDoorDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()