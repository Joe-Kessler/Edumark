#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import cv2
import sqlite3
import shutil
import csv
import numpy as np
import face_recognition
from datetime import datetime

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QTabWidget, QLabel, QPushButton, QLineEdit,
    QMessageBox, QFileDialog, QComboBox, QListWidget, QInputDialog,
    QDialog, QFormLayout, QTableWidget, QTableWidgetItem, QGroupBox,
    QHeaderView, QListWidgetItem
)

DB_NAME = "face_recognition.db"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

GLOBAL_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #2E3440;
    color: #ECEFF4;
    font-family: Helvetica;
    font-size: 13px;
}
QTabWidget::pane { border: none; }
QTabBar::tab {
    background: #3B4252;
    color: #ECEFF4;
    padding: 10px 20px;
    font-size: 14px;
    min-width: 120px;
    min-height: 40px;
}
QTabBar::tab:selected {
    background: #88C0D0;
    color: #2E3440;
}
QPushButton {
    background-color: #88C0D0;
    color: #2E3440;
    border: none;
    border-radius: 5px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 14px;
    min-height: 36px;
}
QPushButton:hover {
    background-color: #81A1C1;
}
QLineEdit, QComboBox, QListWidget, QTableWidget {
    background-color: #3B4252;
    border: 1px solid #4C566A;
    border-radius: 4px;
    selection-background-color: #88C0D0;
    selection-color: #2E3440;
    font-size: 13px;
}
QHeaderView::section {
    background-color: #3B4252;
    padding: 6px;
    border: 1px solid #4C566A;
    font-size: 13px;
}
"""

def update_schema():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(time_slots)")
    cols = [c[1] for c in cur.fetchall()]
    if 'batch_id' not in cols:
        cur.execute("ALTER TABLE time_slots ADD COLUMN batch_id INTEGER")
        conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS admin (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL
                   )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS batch (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER,
                    batch_name TEXT,
                    UNIQUE(admin_id,batch_name),
                    FOREIGN KEY(admin_id) REFERENCES admin(id)
                   )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS time_slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id INTEGER,
                    batch_id INTEGER,
                    start_time TEXT,
                    end_time TEXT,
                    FOREIGN KEY(admin_id) REFERENCES admin(id)
                   )''')
    conn.commit()
    conn.close()
    update_schema()

def get_admin_folder(username):
    path = os.path.join('data', f'admin_{username}')
    os.makedirs(path, exist_ok=True)
    return path

def get_batch_folder(admin_folder, batch_id, batch_name):
    path = os.path.join(admin_folder, f'batch_{batch_id}_{batch_name}')
    os.makedirs(os.path.join(path, 'users'), exist_ok=True)
    return path

def load_known_faces(batch_folder):
    known = {}
    user_dir = os.path.join(batch_folder, 'users')
    if not os.path.isdir(user_dir):
        return known
    for uid in os.listdir(user_dir):
        try:
            u = int(uid)
        except:
            continue
        info_file = os.path.join(user_dir, uid, 'info.txt')
        enc_file = os.path.join(user_dir, uid, 'encoding.npy')
        if os.path.exists(info_file) and os.path.exists(enc_file):
            name = open(info_file).read().strip()
            encoding = np.load(enc_file)
            known[u] = {'name': name, 'encoding': encoding}
    return known

def save_face_encoding(user_folder, encoding):
    np.save(os.path.join(user_folder, 'encoding.npy'), encoding)

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        init_db()
        self.admin_info = None
        self.batches = []
        self.known_faces = {}
        self.attendance = {}
        self.selected_batch = None
        self.selected_slot = None
        self.admin_folder = None
        self.batch_folder = None
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.setWindowTitle('Edumark: Face Recognition Attendance')
        self.setGeometry(200, 100, 1000, 700)
        self._setup_ui()

    def _setup_ui(self):
        app = QApplication.instance()
        app.setStyleSheet(GLOBAL_STYLESHEET)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Login
        self.login_widget = QWidget()
        self._build_login_ui()
        self.stack.addWidget(self.login_widget)

        # Dashboard + Tabs
        self.dashboard_widget = QWidget()
        self._build_dashboard_ui()
        self.stack.addWidget(self.dashboard_widget)

    def _build_login_ui(self):
        layout = QVBoxLayout(self.login_widget)
        layout.setAlignment(Qt.AlignCenter)
        box = QGroupBox('Admin Login / Register')
        form = QFormLayout(box)
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form.addRow('Username:', self.username_input)
        form.addRow('Password:', self.password_input)
        btns = QHBoxLayout()
        login_btn = QPushButton('Login')
        reg_btn = QPushButton('Register')
        login_btn.clicked.connect(self.login)
        reg_btn.clicked.connect(self.register)
        btns.addWidget(login_btn); btns.addWidget(reg_btn)
        form.addRow(btns)
        layout.addWidget(box)

    def login(self):
        uname = self.username_input.text().strip()
        pwd = self.password_input.text().strip()
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "SELECT id,name,username FROM admin WHERE username=? AND password=?",
            (uname, pwd)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            self.admin_info = row
            QMessageBox.information(self, 'Success', f'Welcome {row[1]}')
            self.admin_folder = get_admin_folder(row[2])
            self._refresh_batches()
            self.stack.setCurrentWidget(self.dashboard_widget)
        else:
            QMessageBox.warning(self, 'Error', 'Invalid credentials')

    def register(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Register Admin')
        form = QFormLayout(dlg)
        name_in = QLineEdit()
        user_in = QLineEdit()
        pwd_in = QLineEdit()
        pwd_in.setEchoMode(QLineEdit.Password)
        form.addRow('Name:', name_in)
        form.addRow('Username:', user_in)
        form.addRow('Password:', pwd_in)
        btns = QHBoxLayout()
        ok = QPushButton('OK'); cancel = QPushButton('Cancel')
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        form.addRow(btns)
        if dlg.exec_() == QDialog.Accepted:
            name, uname, pwd = name_in.text(), user_in.text(), pwd_in.text()
            if name and uname and pwd:
                try:
                    conn = sqlite3.connect(DB_NAME)
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO admin(name,username,password) VALUES(?,?,?)",
                        (name, uname, pwd)
                    )
                    conn.commit()
                    conn.close()
                    QMessageBox.information(self, 'Success', 'Registered Successfully')
                except sqlite3.IntegrityError:
                    QMessageBox.warning(self, 'Error', 'Username already exists')
            else:
                QMessageBox.warning(self, 'Error', 'All fields required')

    def _build_dashboard_ui(self):
        layout = QVBoxLayout(self.dashboard_widget)

        # Top row: batch selector + add/delete + logout
        top = QGroupBox()
        hl = QHBoxLayout(top)
        hl.addWidget(QLabel('Batch:'))
        self.batch_combo = QComboBox()
        hl.addWidget(self.batch_combo)
        add_b = QPushButton('Add Batch'); del_b = QPushButton('Delete Batch')
        for btn in (add_b, del_b):
            btn.setStyleSheet("background-color:#88C0D0; color:#2E3440;")
        hl.addWidget(add_b); hl.addWidget(del_b)
        hl.addStretch()
        logout_btn = QPushButton('Logout')
        logout_btn.setStyleSheet("background-color:#88C0D0; color:#2E3440;")
        hl.addWidget(logout_btn)

        add_b.clicked.connect(self.add_batch)
        del_b.clicked.connect(self.delete_batch)
        logout_btn.clicked.connect(self.logout)
        self.batch_combo.currentIndexChanged.connect(self.change_batch)

        layout.addWidget(top)

        # Tabs
        self.inner_tabs = QTabWidget()
        layout.addWidget(self.inner_tabs)

        # Attendance tab
        att_tab = QWidget()
        att_tab.setStyleSheet("background-color: #434C5E;")
        self._build_attendance_ui(att_tab)
        self.inner_tabs.addTab(att_tab, 'Attendance')

        # User Management
        user_tab = QWidget()
        user_tab.setStyleSheet("background-color: #3B4252;")
        self._build_user_ui(user_tab)
        self.inner_tabs.addTab(user_tab, 'User Management')

        # Time Slots
        slot_tab = QWidget()
        slot_tab.setStyleSheet("background-color: #4C566A;")
        self._build_timeslot_ui(slot_tab)
        self.inner_tabs.addTab(slot_tab, 'Time Slots')

    def _refresh_batches(self):
        self.batch_combo.clear()
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT id,batch_name FROM batch WHERE admin_id=?", (self.admin_info[0],))
        self.batches = cur.fetchall()
        conn.close()
        for bid, name in self.batches:
            self.batch_combo.addItem(name, bid)
        if self.batches:
            self.change_batch(0)

    def add_batch(self):
        text, ok = QInputDialog.getText(self, 'Add Batch', 'Batch Name:')
        if ok and text:
            try:
                conn = sqlite3.connect(DB_NAME)
                cur = conn.cursor()
                cur.execute("INSERT INTO batch(admin_id,batch_name) VALUES(?,?)",
                            (self.admin_info[0], text))
                conn.commit()
                conn.close()
                self._refresh_batches()
            except sqlite3.IntegrityError:
                QMessageBox.warning(self, 'Error', 'Batch exists')

    def delete_batch(self):
        idx = self.batch_combo.currentIndex()
        if idx < 0: return
        bid = self.batch_combo.itemData(idx)
        name = self.batch_combo.currentText()
        if QMessageBox.question(self, 'Confirm', f'Delete batch {name}?') == QMessageBox.Yes:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("DELETE FROM batch WHERE id=?", (bid,))
            conn.commit(); conn.close()
            shutil.rmtree(get_batch_folder(self.admin_folder, bid, name), ignore_errors=True)
            self._refresh_batches()

    def change_batch(self, idx):
        if idx < 0: return
        bid, name = self.batches[idx]
        self.selected_batch = (bid, name)
        self.batch_folder = get_batch_folder(self.admin_folder, bid, name)
        self.known_faces = load_known_faces(self.batch_folder)
        self.attendance = {uid: 'Absent' for uid in self.known_faces}
        self._refresh_attendance_table()
        self._refresh_user_list()
        self._refresh_timeslot_list()

    # ---------- Attendance ----------
    def _build_attendance_ui(self, parent):
        layout = QVBoxLayout(parent)
        ctrl = QHBoxLayout()
        start_btn = QPushButton('Start Attendance')
        stop_btn = QPushButton('Stop Attendance')
        export_btn = QPushButton('Export CSV')
        for btn in (start_btn, stop_btn, export_btn):
            btn.setStyleSheet("background-color:#88C0D0; color:#2E3440;")
        ctrl.addWidget(start_btn); ctrl.addWidget(stop_btn); ctrl.addWidget(export_btn)
        layout.addLayout(ctrl)
        start_btn.clicked.connect(self.select_and_start)
        stop_btn.clicked.connect(self.stop_attendance)
        export_btn.clicked.connect(self.export_csv)

        content = QHBoxLayout()
        self.video_label = QLabel(); self.video_label.setFixedSize(640, 480)
        content.addWidget(self.video_label)
        self.att_table = QTableWidget(0, 3)
        self.att_table.setHorizontalHeaderLabels(['ID', 'Name', 'Status'])
        self.att_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.att_table.setEditTriggers(QTableWidget.NoEditTriggers)
        content.addWidget(self.att_table)
        layout.addLayout(content)

    def select_and_start(self):
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, start_time || ' - ' || end_time FROM time_slots "
            "WHERE admin_id=? AND batch_id=?",
            (self.admin_info[0], self.selected_batch[0])
        )
        slots = cur.fetchall()
        conn.close()
        if not slots:
            QMessageBox.warning(self, 'Error', 'No time slots defined'); return
        items = [s[1] for s in slots]
        sel, ok = QInputDialog.getItem(self, 'Select Time Slot', 'Time Slot:', items, 0, False)
        if ok:
            idx = items.index(sel)
            self.selected_slot = slots[idx][0]
            self.attendance = {uid: 'Absent' for uid in self.known_faces}
            self._refresh_attendance_table()
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        frame = cv2.flip(frame, 1)
        rgb = frame[:, :, ::-1]
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        for loc, enc in zip(locs, encs):
            name = 'Unknown'
            for uid, data in self.known_faces.items():
                if face_recognition.compare_faces([data['encoding']], enc, tolerance=0.5)[0]:
                    name = data['name']
                    self.attendance[uid] = 'Present'
                    break
            top, right, bottom, left = loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        h, w, _ = frame.shape
        img = QImage(frame.data, w, h, 3*w, QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(img))
        self._refresh_attendance_table()

    def stop_attendance(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.clear()

    def _refresh_attendance_table(self):
        self.att_table.setRowCount(0)
        for uid, status in self.attendance.items():
            r = self.att_table.rowCount()
            self.att_table.insertRow(r)
            self.att_table.setItem(r, 0, QTableWidgetItem(str(uid)))
            self.att_table.setItem(r, 1, QTableWidgetItem(self.known_faces[uid]['name']))
            cb = QComboBox()
            cb.addItems(['Present', 'Absent'])
            cb.setCurrentText(status)
            cb.currentTextChanged.connect(lambda s, uid=uid: self.attendance.__setitem__(uid, s))
            self.att_table.setCellWidget(r, 2, cb)

    def export_csv(self):
        if not self.selected_slot:
            QMessageBox.warning(self, 'Error', 'No session in progress'); return
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save CSV', f'attendance_{now}.csv', 'CSV Files (*.csv)'
        )
        if not path: return
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Time Slot', 'User ID', 'Name', 'Status'])
            for uid, status in self.attendance.items():
                writer.writerow([
                    now,
                    self.selected_slot,
                    uid,
                    self.known_faces[uid]['name'],
                    status
                ])
        QMessageBox.information(self, 'Export Success', f'Saved to {path}')

    # ---------- User Management ----------
    def _build_user_ui(self, parent):
        layout = QVBoxLayout(parent)
        btns = QHBoxLayout()
        add_btn = QPushButton('Register User')
        del_btn = QPushButton('Delete User')
        for btn in (add_btn, del_btn):
            btn.setStyleSheet("background-color:#88C0D0; color:#2E3440;")
        btns.addWidget(add_btn); btns.addWidget(del_btn)
        layout.addLayout(btns)
        self.user_list = QListWidget()
        layout.addWidget(self.user_list)
        add_btn.clicked.connect(self.register_user)
        del_btn.clicked.connect(self.delete_user)

    def _refresh_user_list(self):
        self.user_list.clear()
        for uid, data in self.known_faces.items():
            item = QListWidgetItem(f"ID: {uid}, Name: {data['name']}")
            self.user_list.addItem(item)

    def register_user(self):
        cap = cv2.VideoCapture(0)
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.warning(self, 'Error', 'Camera error'); return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if not len(faces):
            QMessageBox.warning(self, 'Error', 'No face detected'); return
        x, y, w, h = faces[0]
        rgb = frame[:, :, ::-1]
        encs = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])
        if not encs:
            QMessageBox.warning(self, 'Error', 'Encoding failed'); return
        enc = encs[0]
        name, ok1 = QInputDialog.getText(self, 'Name', 'Enter Name:')
        uid_str, ok2 = QInputDialog.getText(self, 'ID', 'Enter numeric ID:')
        if not(ok1 and ok2): return
        try:
            uid = int(uid_str)
        except:
            QMessageBox.warning(self, 'Error', 'ID must be numeric'); return
        user_dir = os.path.join(self.batch_folder, 'users', str(uid))
        if os.path.exists(user_dir):
            QMessageBox.warning(self, 'Error', 'User exists'); return
        os.makedirs(user_dir)
        with open(os.path.join(user_dir, 'info.txt'), 'w') as f:
            f.write(name)
        save_face_encoding(user_dir, enc)
        self.known_faces = load_known_faces(self.batch_folder)
        self.attendance[uid] = 'Absent'
        self._refresh_user_list()

    def delete_user(self):
        item = self.user_list.currentItem()
        if not item: return
        uid = int(item.text().split(',')[0].split(':')[1].strip())
        if QMessageBox.question(self, 'Confirm', f'Delete user {uid}?') == QMessageBox.Yes:
            path = os.path.join(self.batch_folder, 'users', str(uid))
            shutil.rmtree(path, ignore_errors=True)
            self.known_faces.pop(uid, None)
            self.attendance.pop(uid, None)
            self._refresh_user_list()

    # ---------- Time Slot Management ----------
    def _build_timeslot_ui(self, parent):
        layout = QVBoxLayout(parent)
        btns = QHBoxLayout()
        add_btn = QPushButton('Add Time Slot')
        del_btn = QPushButton('Delete Time Slot')
        for btn in (add_btn, del_btn):
            btn.setStyleSheet("background-color:#88C0D0; color:#2E3440;")
        btns.addWidget(add_btn); btns.addWidget(del_btn)
        layout.addLayout(btns)
        self.slot_list = QListWidget()
        layout.addWidget(self.slot_list)
        add_btn.clicked.connect(self.add_timeslot)
        del_btn.clicked.connect(self.delete_timeslot)

    def _refresh_timeslot_list(self):
        self.slot_list.clear()
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, start_time || ' - ' || end_time FROM time_slots "
            "WHERE admin_id=? AND batch_id=?",
            (self.admin_info[0], self.selected_batch[0])
        )
        for sid, label in cur.fetchall():
            self.slot_list.addItem(f"{sid}: {label}")
        conn.close()

    def add_timeslot(self):
        start, ok1 = QInputDialog.getText(self, 'Start Time', 'Enter HH:MM')
        end, ok2 = QInputDialog.getText(self, 'End Time', 'Enter HH:MM')
        if not(ok1 and ok2): return
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO time_slots(admin_id,batch_id,start_time,end_time) VALUES(?,?,?,?)",
            (self.admin_info[0], self.selected_batch[0], start, end)
        )
        conn.commit(); conn.close()
        self._refresh_timeslot_list()

    def delete_timeslot(self):
        item = self.slot_list.currentItem()
        if not item: return
        sid = int(item.text().split(':')[0])
        if QMessageBox.question(self, 'Confirm', f'Delete time slot {sid}?') == QMessageBox.Yes:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("DELETE FROM time_slots WHERE id=?", (sid,))
            conn.commit(); conn.close()
            self._refresh_timeslot_list()

    def logout(self):
        self.stop_attendance()
        self.stack.setCurrentWidget(self.login_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FaceRecognitionApp()
    win.show()
    sys.exit(app.exec_())


# In[ ]:
