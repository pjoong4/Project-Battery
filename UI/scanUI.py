import sys
import os
import random
import time
import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox
)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class CameraThread(QThread):
    frame_processed = Signal(np.ndarray, str, float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.use_webcam = False
        self.is_scanning = True
        self.cap = None
        self.model = None

    def set_model(self, model):
        self.model = model

    def toggle_webcam(self):
        if not self.use_webcam:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                self.use_webcam = False
                return False
            else:
                self.use_webcam = True
                return True
        else:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.use_webcam = False
            return False

    def run(self):
        while self.running:
            class_name = ""
            conf_val = 0.0

            if self.use_webcam and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    if self.is_scanning and self.model is not None:
                        results = self.model.predict(frame, conf=0.3, verbose=False)
                        frame = results[0].plot()

                        if len(results[0].boxes) > 0:
                            box = results[0].boxes[0]
                            cls_id = int(box.cls[0])
                            class_name = self.model.names[cls_id]
                            conf_val = float(box.conf[0]) * 100.0
                else:
                    frame = self.generate_synthetic_frame()
            else:
                frame = self.generate_synthetic_frame()

            self.frame_processed.emit(frame, class_name, conf_val)
            self.msleep(40)

    def generate_synthetic_frame(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (180, 80), (460, 280), (0, 247, 159), 2)
        cv2.putText(frame, "YOLO TARGET_LOCKED", (185, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 247, 159), 2)
        if self.is_scanning:
            scan_y = int(80 + (time.time() * 120) % 200)
            cv2.line(frame, (180, scan_y), (460, scan_y), (0, 247, 159), 2)
        return frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class VoltScanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VoltScan - Industrial Battery Identification System (PySide6)")
        self.setGeometry(100, 100, 1150, 700)

        # Color Palette
        self.COLOR_BG = "#f9f9f9"
        self.COLOR_SIDEBAR = "#000000"
        self.COLOR_ACCENT = "#a1fb00"
        self.COLOR_PRIMARY = "#416900"
        self.COLOR_TEXT = "#1a1c1c"
        self.COLOR_CARD_BG = "#ffffff"

        self.model = None
        self.model_path = "runs/detect/train-11/weights/best.pt"

        self.cam_thread = CameraThread()
        self.cam_thread.frame_processed.connect(self.on_frame_processed)

        self.load_yolo_model(self.model_path)
        self.setup_ui()
        self.cam_thread.start()

    def load_yolo_model(self, path):
        if not HAS_YOLO:
            print("[Warning] 'ultralytics' 라이브러리가 설치되어 있지 않습니다. (pip install ultralytics)")
            return

        if os.path.exists(path):
            try:
                print(f"[YOLO Engine] 학습 모델 파일 로드 중: {path}")
                self.model = YOLO(path)
                self.cam_thread.set_model(self.model)
                print("[YOLO Engine] ✅ YOLO 모델이 성공적으로 로드되었습니다!")
            except Exception as e:
                print(f"[YOLO Engine Error] 모델 로드 실패: {e}")
                self.model = None
        else:
            print(f"[YOLO Engine] '{path}' 파일이 없습니다. 기본 시뮬레이션 모드로 동작합니다.")

    def select_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "학습된 YOLO 모델 (.pt) 선택", "", "PyTorch YOLO Model (*.pt);;All Files (*)"
        )
        if file_path:
            self.model_path = file_path
            self.load_yolo_model(file_path)
            if self.model is not None:
                QMessageBox.information(self, "YOLO 모델 로드 성공", f"성공적으로 YOLO 모델을 로드했습니다:\n{os.path.basename(file_path)}")
                self.lbl_model_status.setText(f"YOLO Model: {os.path.basename(file_path)}")
                self.lbl_model_status.setStyleSheet(f"color: {self.COLOR_ACCENT}; font-size: 11px; font-weight: bold;")

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {self.COLOR_BG};")
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet(f"background-color: {self.COLOR_SIDEBAR};")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 25, 15, 25)

        lbl_brand = QLabel("VoltScan")
        lbl_brand.setFont(QFont("Helvetica", 18, QFont.Bold))
        lbl_brand.setStyleSheet(f"color: {self.COLOR_ACCENT}; background: transparent;")
        sidebar_layout.addWidget(lbl_brand)

        lbl_ver = QLabel("YOLO-v8 Engine v1.0")
        lbl_ver.setFont(QFont("Helvetica", 9))
        lbl_ver.setStyleSheet("color: #888888; background: transparent;")
        sidebar_layout.addWidget(lbl_ver)
        sidebar_layout.addSpacing(15)

        self.btn_status = QPushButton("● Scanning Active")
        self.btn_status.setFont(QFont("Helvetica", 10, QFont.Bold))
        self.btn_status.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.COLOR_ACCENT};
                color: #102000;
                border: none;
                padding: 10px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #8ddc00;
            }}
        """)
        self.btn_status.clicked.connect(self.toggle_scan)
        sidebar_layout.addWidget(self.btn_status)
        sidebar_layout.addSpacing(10)

        model_name = os.path.basename(self.model_path) if self.model else "Not Loaded (Simulation)"
        status_color = self.COLOR_ACCENT if self.model else "#888888"
        self.lbl_model_status = QLabel(f"YOLO: {model_name}")
        self.lbl_model_status.setWordWrap(True)
        self.lbl_model_status.setStyleSheet(f"color: {status_color}; font-size: 11px; font-weight: bold; background: transparent;")
        sidebar_layout.addWidget(self.lbl_model_status)
        sidebar_layout.addSpacing(10)

        btn_load_pt = QPushButton("📂 YOLO (.pt) 모델 선택")
        btn_load_pt.setFont(QFont("Helvetica", 9, QFont.Bold))
        btn_load_pt.setStyleSheet(f"""
            QPushButton {{
                background-color: #1c1b1b;
                color: {self.COLOR_ACCENT};
                border: 1px solid {self.COLOR_ACCENT};
                padding: 8px;
                border-radius: 4px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #2a2928;
                color: #ffffff;
            }}
        """)
        btn_load_pt.clicked.connect(self.select_model_file)
        sidebar_layout.addWidget(btn_load_pt)

        btn_cam = QPushButton("📷 카메라 (Webcam) 전환")
        btn_cam.setFont(QFont("Helvetica", 10))
        btn_cam.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.COLOR_SIDEBAR};
                color: #ffffff;
                border: none;
                padding: 10px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: #1c1b1b;
                color: {self.COLOR_ACCENT};
            }}
        """)
        btn_cam.clicked.connect(self.toggle_camera_source)
        sidebar_layout.addWidget(btn_cam)

        sidebar_layout.addStretch()
        root_layout.addWidget(sidebar)

        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"background-color: {self.COLOR_BG}; border-bottom: 1px solid #000000;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        lbl_header = QLabel("Battery Identification System (YOLO Detector)")
        lbl_header.setFont(QFont("Helvetica", 16, QFont.Bold))
        lbl_header.setStyleSheet(f"color: {self.COLOR_TEXT}; border: none;")
        header_layout.addWidget(lbl_header)

        content_layout.addWidget(header)

        body_frame = QFrame()
        body_layout = QVBoxLayout(body_frame)
        body_layout.setContentsMargins(20, 15, 20, 15)
        body_layout.setSpacing(15)

        top_split_layout = QHBoxLayout()
        top_split_layout.setSpacing(15)

        self.cam_canvas = QLabel()
        self.cam_canvas.setFixedSize(640, 360)
        self.cam_canvas.setStyleSheet(f"background-color: #000000; border: 2px solid {self.COLOR_PRIMARY};")
        self.cam_canvas.setAlignment(Qt.AlignCenter)
        top_split_layout.addWidget(self.cam_canvas)

        card_frame = QFrame()
        card_frame.setFixedHeight(360)
        card_frame.setStyleSheet(f"background-color: {self.COLOR_CARD_BG}; border: 2px solid {self.COLOR_PRIMARY}; border-radius: 4px;")
        card_layout = QVBoxLayout(card_frame)
        card_layout.setContentsMargins(15, 15, 15, 15)

        lbl_rt_title = QLabel("YOLO REAL-TIME DETECTION")
        lbl_rt_title.setFont(QFont("Helvetica", 9, QFont.Bold))
        lbl_rt_title.setStyleSheet(f"color: {self.COLOR_PRIMARY}; border: none;")
        card_layout.addWidget(lbl_rt_title)

        lbl_match_sub = QLabel("Primary Match (Class)")
        lbl_match_sub.setFont(QFont("Helvetica", 9))
        lbl_match_sub.setStyleSheet("color: #666666; border: none;")
        card_layout.addWidget(lbl_match_sub)

        self.lbl_match = QLabel("Lithium-Ion")
        self.lbl_match.setFont(QFont("Helvetica", 20, QFont.Bold))
        self.lbl_match.setStyleSheet(f"color: {self.COLOR_TEXT}; border: none;")
        card_layout.addWidget(self.lbl_match)

        self.lbl_form = QLabel("Bounding Box: [x1, y1, x2, y2]")
        self.lbl_form.setFont(QFont("Helvetica", 9))
        self.lbl_form.setStyleSheet("color: #444444; border: none;")
        card_layout.addWidget(self.lbl_form)

        card_layout.addSpacing(10)

        lbl_conf_sub = QLabel("CONFIDENCE SCORE")
        lbl_conf_sub.setFont(QFont("Helvetica", 9, QFont.Bold))
        lbl_conf_sub.setStyleSheet("color: #666666; border: none;")
        card_layout.addWidget(lbl_conf_sub)

        self.lbl_conf = QLabel("99.1%")
        self.lbl_conf.setFont(QFont("Helvetica", 22, QFont.Bold))
        self.lbl_conf.setStyleSheet(f"color: {self.COLOR_PRIMARY}; border: none;")
        card_layout.addWidget(self.lbl_conf)

        card_layout.addStretch()

        btn_trigger = QPushButton("⚡ YOLO 추론 감지 스캔")
        btn_trigger.setFont(QFont("Helvetica", 10, QFont.Bold))
        btn_trigger.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.COLOR_PRIMARY};
                color: #ffffff;
                border: none;
                padding: 10px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #304f00;
            }}
        """)
        btn_trigger.clicked.connect(self.run_detection)
        card_layout.addWidget(btn_trigger)

        top_split_layout.addWidget(card_frame)
        body_layout.addLayout(top_split_layout)

        # Lower Section: Treeview (QTableWidget)
        table_frame = QFrame()
        table_frame.setStyleSheet(f"background-color: {self.COLOR_CARD_BG}; border: 2px solid #000000; border-radius: 4px;")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(15, 10, 15, 15)

        lbl_table_head = QLabel("YOLO Detection Log")
        lbl_table_head.setFont(QFont("Helvetica", 11, QFont.Bold))
        lbl_table_head.setStyleSheet(f"color: {self.COLOR_TEXT}; border: none;")
        table_layout.addWidget(lbl_table_head)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID CODE", "CLASSIFICATION (CLASS)", "CONFIDENCE", "HAZARD LEVEL"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dddddd;
                background-color: #ffffff;
                color: #1a1c1c; /* 글자색 명시 */
            }
            QTableWidget::item {
                color: #1a1c1c; /* 셀 아이템 글자색 명시 */
            }
        """)
        table_layout.addWidget(self.table)

        self.add_table_row("#BAT-8821A", "Lithium-Ion (18650)", "99.1%", "High (Flammable)")
        self.add_table_row("#BAT-8820B", "Alkaline AA", "97.4%", "Low")

        body_layout.addWidget(table_frame)
        content_layout.addWidget(body_frame)

        root_layout.addWidget(content_frame)

    def add_table_row(self, id_code, classification, confidence, hazard, top=False):
        row_idx = 0 if top else self.table.rowCount()
        self.table.insertRow(row_idx)
        self.table.setItem(row_idx, 0, QTableWidgetItem(id_code))
        self.table.setItem(row_idx, 1, QTableWidgetItem(classification))
        self.table.setItem(row_idx, 2, QTableWidgetItem(confidence))
        self.table.setItem(row_idx, 3, QTableWidgetItem(hazard))

    def toggle_scan(self):
        self.cam_thread.is_scanning = not self.cam_thread.is_scanning
        if self.cam_thread.is_scanning:
            self.btn_status.setText("● Scanning Active")
            self.btn_status.setStyleSheet(f"background-color: {self.COLOR_ACCENT}; color: #102000; border: none; padding: 10px; border-radius: 4px;")
        else:
            self.btn_status.setText("○ Paused")
            self.btn_status.setStyleSheet("background-color: #444444; color: #ffffff; border: none; padding: 10px; border-radius: 4px;")

    def toggle_camera_source(self):
        success = self.cam_thread.toggle_webcam()
        if not success and not self.cam_thread.use_webcam:
            QMessageBox.warning(self, "카메라 연결 안내", "실물 웹캠 카메라가 감지되지 않아 시뮬레이션 모드로 동작합니다.")

    def run_detection(self):
        """수동 스캔/추론 트리거"""
        if self.model is not None and self.cam_thread.use_webcam and self.cam_thread.cap is not None:
            ret, frame = self.cam_thread.cap.read()
            if ret:
                results = self.model.predict(frame, conf=0.25)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id]
                        conf_val = float(box.conf[0]) * 100.0

                        self.lbl_match.setText(class_name)
                        self.lbl_conf.setText(f"{conf_val:.1f}%")

                        id_code = f"#BAT-{random.randint(8822, 9999)}{chr(random.randint(65, 90))}"
                        self.add_table_row(id_code, class_name, f"{conf_val:.1f}%", "Active Scanned", top=True)
                        return

        batteries = [
            ("Lithium-Ion (18650)", "99.3%", "High (Flammable)"),
            ("Alkaline AA", "98.1%", "Low"),
            ("Ni-MH Pack", "97.6%", "Low"),
            ("LiFePO4 Cell", "99.5%", "Moderate"),
        ]
        match = random.choice(batteries)
        self.lbl_match.setText(match[0])
        self.lbl_conf.setText(match[1])

        id_code = f"#BAT-{random.randint(8822, 9999)}{chr(random.randint(65, 90))}"
        self.add_table_row(id_code, match[0], match[1], match[2], top=True)

    def on_frame_processed(self, frame, class_name, conf_val):
        """카메라 스레드에서 전송된 프레임을 QLabel에 바인딩"""
        if class_name:
            self.lbl_match.setText(class_name)
            self.lbl_conf.setText(f"{conf_val:.1f}%")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_canvas.setPixmap(pixmap)

    def closeEvent(self, event):
        """앱 종료 시 스레드 정지"""
        self.cam_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoltScanApp()
    window.show()
    sys.exit(app.exec())