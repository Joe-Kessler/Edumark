import os
import cv2
import sqlite3
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import csv
from datetime import datetime
import numpy as np
import shutil
import face_recognition  # Using face_recognition for robust detection

# ---------- Database Setup ----------
DB_NAME = "face_recognition.db"
if not os.path.exists(DB_NAME):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE admin (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                   )''')
    cur.execute('''CREATE TABLE time_slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER,
                    start_time TEXT,
                    end_time TEXT,
                    FOREIGN KEY(admin_id) REFERENCES admin(id)
                   )''')
    conn.commit()
    conn.close()

# Haar Cascade for face detection (used only for drawing boxes during capture)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Global variable for camera control
camera_running = False

# ---------- Utility Functions ----------
def get_admin_folder(username):
    folder = os.path.join("data", f"admin_{username}")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "users"), exist_ok=True)
    os.makedirs(os.path.join(folder, "attendance"), exist_ok=True)
    return folder

def load_known_faces(admin_folder):
    """
    Load all known face encodings from the admin's user folder.
    Each user's folder should contain an 'encoding.npy' file.
    Returns a dict: {user_id (int): {"name": str, "encoding": np.array}}
    """
    known = {}
    user_folder = os.path.join(admin_folder, "users")
    if not os.path.exists(user_folder):
        return known
    for user_id in os.listdir(user_folder):
        user_path = os.path.join(user_folder, user_id)
        info_file = os.path.join(user_path, "info.txt")
        encoding_file = os.path.join(user_path, "encoding.npy")
        if os.path.exists(info_file) and os.path.exists(encoding_file):
            with open(info_file, "r") as f:
                name = f.read().strip()
            try:
                uid = int(user_id)
            except ValueError:
                continue
            encoding = np.load(encoding_file)
            known[uid] = {"name": name, "encoding": encoding}
    return known

def save_face_encoding(user_folder, encoding):
    """
    Save the given face encoding as a NumPy file.
    """
    np.save(os.path.join(user_folder, "encoding.npy"), encoding)

# ---------- Main Application Class ----------
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("900x700")
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.admin_info = None   # (id, name, username)
        self.admin_folder = None
        self.known_faces = {}    # {user_id: {"name": str, "encoding": np.array}}
        self.attendance = {}     # {user_id: "Present" or "Absent"}
        self.selected_time_slot = None
        self.camera_image = None
        self.cap = None          # VideoCapture object
        self.frame_counter = 0   # For frame skipping in update_camera
        self.last_detections = []  # Cache last detections (locations and encodings)
        self.create_login_ui()

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ------------- Login & Registration -------------
    def create_login_ui(self):
        self.clear_frame()
        frm = ttk.Frame(self.root, padding=20)
        frm.pack(expand=True)
        ttk.Label(frm, text="Login", font=("Helvetica", 20)).grid(row=0, column=0, columnspan=2, pady=10)
        ttk.Label(frm, text="Username:").grid(row=1, column=0, sticky="e", pady=5)
        self.username_entry = ttk.Entry(frm, width=30)
        self.username_entry.grid(row=1, column=1, pady=5)
        ttk.Label(frm, text="Password:").grid(row=2, column=0, sticky="e", pady=5)
        self.password_entry = ttk.Entry(frm, width=30, show="*")
        self.password_entry.grid(row=2, column=1, pady=5)
        ttk.Button(frm, text="Login", command=self.login).grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(frm, text="Register", command=self.register).grid(row=4, column=0, columnspan=2, pady=5)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT id, name, username FROM admin WHERE username=? AND password=?", (username, password))
        admin = cur.fetchone()
        conn.close()
        if admin:
            self.admin_info = admin
            messagebox.showinfo("Login Success", f"Welcome, {admin[1]}!")
            self.admin_folder = get_admin_folder(admin[2])
            self.known_faces = load_known_faces(self.admin_folder)
            self.create_dashboard_ui()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    def register(self):
        reg_win = tk.Toplevel(self.root)
        reg_win.title("Admin Registration")
        reg_win.geometry("300x200")
        frm = ttk.Frame(reg_win, padding=10)
        frm.pack(expand=True, fill="both")
        ttk.Label(frm, text="Name:").grid(row=0, column=0, sticky="e", pady=5)
        name_entry = ttk.Entry(frm, width=25)
        name_entry.grid(row=0, column=1, pady=5)
        ttk.Label(frm, text="Username:").grid(row=1, column=0, sticky="e", pady=5)
        username_entry = ttk.Entry(frm, width=25)
        username_entry.grid(row=1, column=1, pady=5)
        ttk.Label(frm, text="Password:").grid(row=2, column=0, sticky="e", pady=5)
        password_entry = ttk.Entry(frm, width=25, show="*")
        password_entry.grid(row=2, column=1, pady=5)
        def do_register():
            name = name_entry.get()
            username = username_entry.get()
            password = password_entry.get()
            if name and username and password:
                conn = sqlite3.connect(DB_NAME)
                cur = conn.cursor()
                try:
                    cur.execute("INSERT INTO admin (name, username, password) VALUES (?, ?, ?)", (name, username, password))
                    conn.commit()
                    messagebox.showinfo("Success", "Registration successful!")
                    reg_win.destroy()
                except sqlite3.IntegrityError:
                    messagebox.showerror("Error", "Username already exists")
                conn.close()
            else:
                messagebox.showerror("Error", "All fields required")
        ttk.Button(frm, text="Register", command=do_register).grid(row=3, column=0, columnspan=2, pady=10)

    # ------------- Dashboard & Attendance UI -------------
    def create_dashboard_ui(self):
        self.clear_frame()
        header_frame = ttk.Frame(self.root, padding=10)
        header_frame.pack(fill="x")
        header = ttk.Label(header_frame, text=f"Welcome Admin: {self.admin_info[1]}", font=("Helvetica", 18))
        header.pack(side="left", padx=10)
        ttk.Button(header_frame, text="Logout", command=self.logout).pack(side="right", padx=10)
        
        # Top frame for control buttons
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x")
        ttk.Button(control_frame, text="Manage Time Slots", command=self.manage_time_slots).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Start Attendance", command=self.show_time_slot_dropdown).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Stop Attendance", command=self.stop_attendance).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Register User", command=self.register_user).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(control_frame, text="View Users", command=self.view_users).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(control_frame, text="Delete User", command=self.delete_user).grid(row=0, column=5, padx=5, pady=5)

        # Main frame for camera feed and attendance table
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(expand=True, fill="both")
        # Left: Camera feed
        cam_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding=10)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.camera_panel = ttk.Label(cam_frame)
        self.camera_panel.pack(expand=True, fill="both")
        # Right: Attendance table
        table_frame = ttk.LabelFrame(main_frame, text="Attendance", padding=10)
        table_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.attendance_table = ttk.Treeview(table_frame, columns=("ID", "Name", "Status"), show="headings", height=15)
        self.attendance_table.heading("ID", text="ID")
        self.attendance_table.heading("Name", text="Name")
        self.attendance_table.heading("Status", text="Attendance")
        self.attendance_table.column("ID", width=50, anchor="center")
        self.attendance_table.column("Name", width=150, anchor="center")
        self.attendance_table.column("Status", width=100, anchor="center")
        self.attendance_table.pack(expand=True, fill="both", side="left")
        self.attendance_table.bind("<Double-1>", self.edit_attendance)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.attendance_table.yview)
        self.attendance_table.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom frame for export button
        export_frame = ttk.Frame(self.root, padding=10)
        export_frame.pack(fill="x")
        ttk.Button(export_frame, text="Export Attendance as CSV", command=self.export_attendance_csv).pack()

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def logout(self):
        self.stop_attendance()
        self.admin_info = None
        self.admin_folder = None
        self.known_faces = {}
        self.attendance = {}
        self.create_login_ui()

    # ------------- Time Slot Management -------------
    def manage_time_slots(self):
        ts_win = tk.Toplevel(self.root)
        ts_win.title("Time Slot Management")
        ts_win.geometry("400x300")
        frm = ttk.Frame(ts_win, padding=10)
        frm.pack(expand=True, fill="both")
        ttk.Label(frm, text="Time Slots", font=("Helvetica", 14)).grid(row=0, column=0, columnspan=3, pady=5)
        listbox = tk.Listbox(frm, width=40, height=10)
        listbox.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT id, start_time, end_time FROM time_slots WHERE admin_id=?", (self.admin_info[0],))
        slots = cur.fetchall()
        conn.close()
        for s in slots:
            listbox.insert(tk.END, f"{s[0]}: {s[1]} - {s[2]}")
        def add_slot():
            start = simpledialog.askstring("Input", "Enter start time (HH:MM)", parent=ts_win)
            end = simpledialog.askstring("Input", "Enter end time (HH:MM)", parent=ts_win)
            if start and end:
                conn = sqlite3.connect(DB_NAME)
                cur = conn.cursor()
                cur.execute("INSERT INTO time_slots (admin_id, start_time, end_time) VALUES (?, ?, ?)",
                            (self.admin_info[0], start, end))
                conn.commit()
                conn.close()
                listbox.insert(tk.END, f"(new): {start} - {end}")
        def delete_slot():
            sel = listbox.curselection()
            if sel:
                slot_text = listbox.get(sel[0])
                slot_id = slot_text.split(":")[0]
                if messagebox.askyesno("Confirm", "Are you sure you want to delete this time slot?"):
                    conn = sqlite3.connect(DB_NAME)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM time_slots WHERE id=?", (slot_id,))
                    conn.commit()
                    conn.close()
                    listbox.delete(sel[0])
        ttk.Button(frm, text="Add Time Slot", command=add_slot).grid(row=2, column=0, pady=5)
        ttk.Button(frm, text="Delete Time Slot", command=delete_slot).grid(row=2, column=1, pady=5)

    # ------------- Attendance Functions -------------
    def show_time_slot_dropdown(self):
        # Create a new window to select a time slot using a dropdown
        ts_win = tk.Toplevel(self.root)
        ts_win.title("Select Time Slot")
        ts_win.geometry("300x150")
        frm = ttk.Frame(ts_win, padding=10)
        frm.pack(expand=True, fill="both")
        ttk.Label(frm, text="Select Time Slot:", font=("Helvetica", 12)).pack(pady=10)
        # Fetch time slots for current admin
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT id, start_time, end_time FROM time_slots WHERE admin_id=?", (self.admin_info[0],))
        slots = cur.fetchall()
        conn.close()
        if not slots:
            messagebox.showerror("Error", "No time slots defined. Please add one first.")
            ts_win.destroy()
            return
        # Create a dict for display and mapping to slot id
        self.slot_options = {f"{s[1]} - {s[2]}": s[0] for s in slots}
        slot_names = list(self.slot_options.keys())
        self.slot_var = tk.StringVar(value=slot_names[0])
        slot_dropdown = ttk.Combobox(frm, textvariable=self.slot_var, values=slot_names, state="readonly")
        slot_dropdown.pack(pady=5)
        def select_slot():
            chosen = self.slot_var.get()
            self.selected_time_slot = self.slot_options.get(chosen)
            ts_win.destroy()
            self.start_attendance_session()
        ttk.Button(frm, text="Start", command=select_slot).pack(pady=10)

    def start_attendance_session(self):
        # Initialize attendance: mark all registered users as "Absent"
        self.attendance = {uid: "Absent" for uid in self.known_faces.keys()}
        self.refresh_attendance_table()
        global camera_running
        camera_running = True
        self.cap = cv2.VideoCapture(0)
        # Reset frame counter and detections cache
        self.frame_counter = 0
        self.last_detections = []
        self.update_camera()

    def update_camera(self):
        if not camera_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            # Mirror and reduce resolution to lessen processing load
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            self.frame_counter += 1
            # Process every 3rd frame for face detection/recognition
            if self.frame_counter % 3 == 0:
                rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB for face_recognition
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                # Cache detections to use for drawing on every frame
                self.last_detections = list(zip(face_locations, face_encodings))
                # Update attendance based on detections
                for (top, right, bottom, left), face_encoding in self.last_detections:
                    for uid, data in self.known_faces.items():
                        matches = face_recognition.compare_faces([data["encoding"]], face_encoding, tolerance=0.5)
                        if matches[0]:
                            self.attendance[uid] = "Present"
                            break
            # Draw the cached detections
            for (top, right, bottom, left), face_encoding in self.last_detections:
                name = "Unknown"
                for uid, data in self.known_faces.items():
                    matches = face_recognition.compare_faces([data["encoding"]], face_encoding, tolerance=0.5)
                    if matches[0]:
                        name = data["name"]
                        break
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # Convert the frame to ImageTk format and update panel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self.camera_image = ImageTk.PhotoImage(image=img)
            self.camera_panel.config(image=self.camera_image)
            self.refresh_attendance_table()
        self.root.after(30, self.update_camera)

    def stop_attendance(self):
        global camera_running
        camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_panel.config(image='')

    def refresh_attendance_table(self):
        for row in self.attendance_table.get_children():
            self.attendance_table.delete(row)
        for uid, status in self.attendance.items():
            name = self.known_faces.get(uid, {}).get("name", "Unknown")
            self.attendance_table.insert("", tk.END, values=(uid, name, status))

    def export_attendance_csv(self):
        if not self.selected_time_slot:
            messagebox.showerror("Error", "No attendance session in progress.")
            return
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 initialfile=f"attendance_{now}.csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time Slot", "User ID", "Name", "Attendance"])
                for uid, status in self.attendance.items():
                    writer.writerow([now, self.selected_time_slot, uid,
                                     self.known_faces.get(uid, {}).get("name", "Unknown"), status])
            messagebox.showinfo("Export Success", f"Attendance exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export CSV: {e}")

    # ------------- User Registration -------------
    def register_user(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        already_registered = False
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                # Compute encoding for the detected face
                rgb_face = face_img[:, :, ::-1]
                encodings = face_recognition.face_encodings(rgb_face)
                if encodings:
                    face_encoding = encodings[0]
                    # Check against known faces
                    for uid, data in self.known_faces.items():
                        matches = face_recognition.compare_faces([data["encoding"]], face_encoding, tolerance=0.5)
                        if matches[0]:
                            messagebox.showinfo("Info", "Face is already registered!")
                            already_registered = True
                            break
            cv2.imshow("Register - Press 'q' to capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or already_registered:
                break
        cap.release()
        cv2.destroyAllWindows()
        if already_registered:
            return
        user_name = simpledialog.askstring("Input", "Enter user name:")
        user_id = simpledialog.askstring("Input", "Enter user ID (numeric):")
        if not user_name or not user_id:
            messagebox.showerror("Error", "Name and ID are required")
            return
        try:
            user_id_int = int(user_id)
        except ValueError:
            messagebox.showerror("Error", "User ID must be numeric")
            return
        user_folder = os.path.join(self.admin_folder, "users", str(user_id_int))
        if os.path.exists(user_folder):
            messagebox.showerror("Error", "User ID already exists")
            return
        os.makedirs(user_folder)
        with open(os.path.join(user_folder, "info.txt"), "w") as f:
            f.write(user_name)
        # Capture images for registration and compute face encoding from a clear sample
        cap = cv2.VideoCapture(0)
        captured_encoding = None
        count = 0
        messagebox.showinfo("Info", "Capturing images. Press 'q' when a clear face is visible.")
        while count < 100:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    captured_encoding = face_encodings[0]
                    count += 1
                    cv2.putText(frame, f"Captured {count}/100", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Capturing images", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if captured_encoding is None:
            messagebox.showerror("Error", "Could not capture face encoding")
            return
        # Save the captured face encoding for later use during recognition
        save_face_encoding(user_folder, captured_encoding)
        messagebox.showinfo("Info", "User registered. Training model...")
        # Reload known faces
        self.known_faces = load_known_faces(self.admin_folder)
        self.attendance[user_id_int] = "Absent"
        self.refresh_attendance_table()

    # ------------- View & Delete Users -------------
    def view_users(self):
        view_win = tk.Toplevel(self.root)
        view_win.title("Registered Users")
        view_win.geometry("350x300")
        frm = ttk.Frame(view_win, padding=10)
        frm.pack(expand=True, fill="both")
        listbox = tk.Listbox(frm, width=50)
        listbox.pack(padx=10, pady=10, fill="both", expand=True)
        user_folder = os.path.join(self.admin_folder, "users")
        if os.path.exists(user_folder):
            for user_id in os.listdir(user_folder):
                user_path = os.path.join(user_folder, user_id)
                info_file = os.path.join(user_path, "info.txt")
                if os.path.exists(info_file):
                    with open(info_file, "r") as f:
                        name = f.read().strip()
                    listbox.insert(tk.END, f"ID: {user_id}, Name: {name}")

    def delete_user(self):
        user_id = simpledialog.askstring("Delete User", "Enter User ID to delete:")
        if not user_id:
            return
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete user {user_id}?"):
            user_folder = os.path.join(self.admin_folder, "users", user_id)
            if os.path.exists(user_folder):
                shutil.rmtree(user_folder)
                messagebox.showinfo("Info", f"User {user_id} deleted.")
                self.known_faces = load_known_faces(self.admin_folder)
                try:
                    uid_int = int(user_id)
                    if uid_int in self.attendance:
                        del self.attendance[uid_int]
                except ValueError:
                    pass
                self.refresh_attendance_table()
            else:
                messagebox.showerror("Error", "User not found.")

    # ------------- Manual Edit of Attendance (with dropdown) -------------
    def edit_attendance(self, event):
        selected_item = self.attendance_table.focus()
        if not selected_item:
            return
        values = self.attendance_table.item(selected_item, "values")
        try:
            user_id = int(values[0])
        except ValueError:
            return
        current_status = values[2]
        
        # Popup window with a dropdown for attendance status
        edit_win = tk.Toplevel(self.root)
        edit_win.title("Edit Attendance")
        edit_win.geometry("300x150")
        frm = ttk.Frame(edit_win, padding=10)
        frm.pack(expand=True, fill="both")
        ttk.Label(frm, text=f"User {values[1]}:", font=("Helvetica", 12)).pack(pady=10)
        status_var = tk.StringVar(value=current_status)
        status_dropdown = ttk.Combobox(frm, textvariable=status_var, values=["Present", "Absent"], state="readonly")
        status_dropdown.pack(pady=5)
        def update_status():
            new_status = status_var.get()
            self.attendance[user_id] = new_status
            self.refresh_attendance_table()
            edit_win.destroy()
        ttk.Button(frm, text="Update", command=update_status).pack(pady=10)

# ------------- Main -------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
