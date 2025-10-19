import cv2
import sys
import time
import os
import base64
import json
import argparse
from typing import List, Tuple
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import numpy as np

import re
import difflib

BULLET_CHARS = "•·●○◦▪▫"  # common bullet symbols

def _send_to_frontend(text: str, endpoint: str):
    try:
        import requests
        print(f"[backend] posting to {endpoint}", flush=True)
        resp = requests.post(endpoint, json={"type": "send", "text": text}, timeout=1.5)
        print(f"[backend] post result {resp.status_code}", flush=True)
        if resp.status_code >= 400:
            try:
                print(f"[backend] response: {resp.text}", flush=True)
            except Exception:
                pass
    except Exception as e:
        import traceback
        print(f"[backend] post failed: {e}", file=sys.stderr)
        traceback.print_exc()

def _denoise_symbols(s: str) -> str:
    # Remove bullets and leading list markers
    s = re.sub(rf"^[\s\-\*{BULLET_CHARS}]+\s*", "", s)
    # Normalize fancy quotes to straight quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # Collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _beautify_display(s: str) -> str:
    s = _denoise_symbols(s)
    # Normalize multi-dots to ellipsis
    s = re.sub(r"\.{3,}", "...", s)
    # Normalize repeated ?! at end
    m = re.search(r"\s*([?!]{2,})$", s)
    if m:
        cluster = m.group(1)
        if ("?" in cluster) and ("!" in cluster):
            s = re.sub(r"\s*[?!]{2,}$", " ??!", s)
        else:
            # Keep cluster but ensure a single space before it
            s = re.sub(r"\s*[?!]{2,}$", " " + cluster, s)
    return s.strip()

def _coalesce_lines(lines):
    # Merge short/continuation lines to reduce false splits (e.g., "Not" + "• reading it?!!")
    out = []
    for ln in lines:
        ln = _denoise_symbols(ln)
        if not out:
            out.append(ln)
            continue
        prev = out[-1]
        prev_end = prev[-1] if prev else ""
        should_join = (
            (len(prev) < 10) or
            (prev_end not in ".!?") or
            (ln and ln[0].islower()) or
            (ln and ln[0] in BULLET_CHARS)
        )
        if should_join:
            out[-1] = (prev + " " + ln).strip()
        else:
            out.append(ln)
    return out

def _norm_line(s: str) -> str:
    s = s.lower().replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # 去掉结尾的孤立标点
    s = re.sub(r"[.,;:!?]+$", "", s)
    return s

def _is_noise(s: str) -> bool:
    t = s.strip()
    if len(t) < 3:
        return True
    if t.isdigit():
        return True
    return False

class TextStabilizer:
    def __init__(self, threshold: float = 2.0, boost: float = 1.0, decay: float = 0.85, fuzzy: float = 0.92, max_keys: int = 300, cooldown_s: float = 6.0):
        self.threshold = threshold
        self.boost = boost
        self.decay = decay
        self.fuzzy = fuzzy
        self.max_keys = max_keys
        self.cooldown_s = cooldown_s
        self.scores = {}       # key -> score
        self.canonical = {}    # key -> last best original variant
        self.emitted = set()   # key already printed
        self.last_emit_ts = {} # key -> last emission monotonic timestamp

    def _match_key(self, norm: str) -> str:
        if norm in self.scores:
            return norm
        # 近似匹配：和已有 key 相似则复用
        best_key, best = None, 0.0
        for k in self.scores.keys():
            r = difflib.SequenceMatcher(None, norm, k).ratio()
            if r > best:
                best_key, best = k, r
        if best_key and best >= self.fuzzy:
            return best_key
        return norm

    def update(self, lines):
        # 衰减所有分数
        for k in list(self.scores.keys()):
            self.scores[k] *= self.decay
            if self.scores[k] < 0.05:
                # 清理长期无贡献的键
                del self.scores[k]
                self.canonical.pop(k, None)
                self.emitted.discard(k)
                self.last_emit_ts.pop(k, None)

        # 本帧命中的 key
        seen_now = set()

        for raw in lines:
            if _is_noise(raw):
                continue
            norm = _norm_line(raw)
            if _is_noise(norm):
                continue
            key = self._match_key(norm)
            seen_now.add(key)
            self.scores[key] = self.scores.get(key, 0.0) + self.boost
            # 选择一个更“好看”的展示版本：优先更长、含标点/首字母等
            prev = self.canonical.get(key)
            if (prev is None) or (len(raw) > len(prev)):
                self.canonical[key] = raw.strip()

        # 达阈值且未过冷却的作为“稳定增量”
        out = []
        now_mono = time.monotonic()
        for k in seen_now:
            if self.scores.get(k, 0.0) >= self.threshold:
                last_ts = self.last_emit_ts.get(k, 0.0)
                if (k not in self.emitted) or ((now_mono - last_ts) >= self.cooldown_s):
                    out.append(self.canonical.get(k, k))
                    self.emitted.add(k)
                    self.last_emit_ts[k] = now_mono

        # 简单的容量控制
        if len(self.scores) > self.max_keys:
            # 丢掉分数最低的一批
            for k, _ in sorted(self.scores.items(), key=lambda kv: kv[1])[: len(self.scores) - self.max_keys]:
                self.scores.pop(k, None)
                self.canonical.pop(k, None)
                self.emitted.discard(k)
                self.last_emit_ts.pop(k, None)

        return out

# Optional: Google Cloud Vision client library (preferred when Application Default Credentials are configured)
try:
    from google.cloud import vision as gcv_vision
    from google.api_core.client_options import ClientOptions  # type: ignore
except Exception:  # library not installed or unavailable
    gcv_vision = None
    ClientOptions = None  # type: ignore
    pass


class HandwritingRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwriting Test")
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 2px solid #dfdfdf;
                font: 12pt 'Cascadia Code';
            }
            QPushButton#start_stop_button {
                background-color: #1e1e1e;
                color: #ffffff;
                font: 12pt 'Cascadia Code';
                border: none;
                margin: 5px 2px;
                padding: 10px 50px;
            }
            QPushButton#start_stop_button:hover {
                background-color: #dfdfdf;
            }

            QPushButton#exit_button {
                background-color: #1e1e1e;
                color: #ffffff;
                font: 12pt 'Cascadia Code';
                border: none;
                margin: 5px 2px;
                padding: 10px 50px;
            }
            QPushButton#exit_button:hover {
                background-color: #dfdfdf;
            }

            QLabel {
                background-color: #dfdfdf;
            }

            start
        """)
        self.video_running = True
        
        # Backend selection: prefer client library if available and credentials are set; otherwise use REST with API key
        self.api_key = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GCP_API_KEY") or os.getenv("VISION_API_KEY")
        # Allow overriding endpoint (e.g., eu-vision.googleapis.com)
        self.api_endpoint = os.getenv("GOOGLE_VISION_API_ENDPOINT", "https://vision.googleapis.com")

        self.ocr_backend = "rest" if self.api_key else "client"

        self.vision_client = None
        if self.ocr_backend == "client" and gcv_vision is not None:
            # Use custom endpoint if provided (strip protocol for client options)
            client_options = None
            if self.api_endpoint and self.api_endpoint.startswith("https://"):
                endpoint_host = self.api_endpoint.replace("https://", "")
                client_options = ClientOptions(api_endpoint=endpoint_host) if ClientOptions else None
            try:
                self.vision_client = gcv_vision.ImageAnnotatorClient(client_options=client_options) if client_options else gcv_vision.ImageAnnotatorClient()
            except Exception:
                # Fallback to REST if client initialization fails
                self.ocr_backend = "rest"

        # Throttle OCR calls (seconds)
        self.ocr_interval = 0.7
        self._last_ocr_ts = 0.0

        # Cache last OCR result to keep UI stable between calls
        self._last_text = ""
        self._last_boxes = []  # list of list[(x, y)]
        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # layouts
        self.main_layout = QHBoxLayout()  # Horizontal layout for main window
        self.video_layout = QVBoxLayout()  # Vertical layout for video frame
        self.text_layout = QVBoxLayout()  # Vertical layout for text field and buttons

        # video frame label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # text_field
        self.text_field = QTextEdit(self)
        self.text_field.setReadOnly(True)
        self.text_layout.addWidget(self.text_field)

        # buttons layout
        buttons_layout = QHBoxLayout()

        # start_stop_button
        self.start_stop_button = QPushButton("Stop Video", self)
        self.start_stop_button.setObjectName("start_stop_button")
        self.start_stop_button.setFixedSize(250, 60)
        self.start_stop_button.clicked.connect(self.toggle_video_feed)
        buttons_layout.addWidget(self.start_stop_button)

        # exit_button
        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setObjectName("exit_button")
        self.exit_button.setFixedSize(250, 60)
        self.exit_button.clicked.connect(self.close)
        buttons_layout.addWidget(self.exit_button)

        # Add buttons layout to text layout
        self.text_layout.addLayout(buttons_layout)

        # Add video layout and text layout to main layout
        self.main_layout.addLayout(self.video_layout)
        self.main_layout.addLayout(self.text_layout)

        self.central_widget.setLayout(self.main_layout)

        # Video capture
        self.cap = cv2.VideoCapture(5)

        # setting timer for video feed...
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_feed)
        self.timer.start(10)

    def _parse_text_annotations_boxes(self, annotations) -> List[List[Tuple[int, int]]]:
        boxes: List[List[Tuple[int, int]]] = []
        if not annotations:
            return boxes
        for entity in annotations[1:]:
            # gRPC: entity.bounding_poly.vertices; REST: entity['boundingPoly']['vertices']
            verts = None
            if hasattr(entity, "bounding_poly"):
                verts = getattr(entity.bounding_poly, "vertices", None)
                if verts:
                    pts = [(getattr(v, "x", 0) or 0, getattr(v, "y", 0) or 0) for v in verts]
                    if len(pts) == 4:
                        boxes.append(pts)
            elif isinstance(entity, dict):
                bp = entity.get("boundingPoly", {})
                verts = bp.get("vertices", [])
                pts = [(int(v.get("x", 0) or 0), int(v.get("y", 0) or 0)) for v in verts]
                if len(pts) == 4:
                    boxes.append(pts)
        return boxes

    def _call_vision_ocr(self, frame_rgb):
        # Convert to JPEG bytes for Vision API
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return self._last_text, self._last_boxes

        if self.ocr_backend == "rest":
            import requests  # lazy import
            try:
                content_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                url_base = self.api_endpoint.rstrip("/")
                url = f"{url_base}/v1/images:annotate?key={self.api_key}"
                payload = {
                    "requests": [
                        {
                            "image": {"content": content_b64},
                            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                        }
                    ]
                }
                headers = {"Content-Type": "application/json"}
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
                if resp.status_code != 200:
                    return self._last_text, self._last_boxes
                data = resp.json()
                responses = data.get("responses", [])
                if not responses:
                    return self._last_text, self._last_boxes
                r0 = responses[0]
                if "error" in r0:
                    return self._last_text, self._last_boxes
                full_text = r0.get("fullTextAnnotation", {}).get("text", "").strip()
                annotations = r0.get("textAnnotations", [])
                boxes = self._parse_text_annotations_boxes(annotations)
                return full_text, boxes
            except Exception:
                return self._last_text, self._last_boxes

        # Client library path
        if self.vision_client is None or gcv_vision is None:
            return self._last_text, self._last_boxes

        try:
            image = gcv_vision.Image(content=buf.tobytes())
            response = self.vision_client.document_text_detection(image=image)
            if response.error.message:
                return self._last_text, self._last_boxes

            full_text = ""
            if response.full_text_annotation and response.full_text_annotation.text:
                full_text = response.full_text_annotation.text.strip()

            boxes = self._parse_text_annotations_boxes(response.text_annotations)
            return full_text, boxes
        except Exception:
            return self._last_text, self._last_boxes

    def recognize_text(self, frame_rgb):
        now = time.time()
        need_call = (now - self._last_ocr_ts) >= self.ocr_interval

        if need_call:
            text, boxes = self._call_vision_ocr(frame_rgb)
            self._last_text = text
            self._last_boxes = boxes
            self._last_ocr_ts = now
            self.text_field.clear()
            if self._last_text:
                self.text_field.append(self._last_text)

        # draw cached boxes
        for pts in self._last_boxes:
            np_pts = np.array(pts, dtype=np.int32)
            cv2.polylines(frame_rgb, [np_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        return frame_rgb

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for displaying in PyQt5

            # Process frame to recognize text and draw bounding boxes
            frame_with_boxes = self.recognize_text(frame)

            # converting frame to QImage
            h, w, ch = frame_with_boxes.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(frame_with_boxes.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)

            self.video_label.setPixmap(QPixmap.fromImage(p))

    def toggle_video_feed(self):
        self.video_running = not self.video_running
        if self.video_running:
            self.timer.start(10)
            self.start_stop_button.setText("Stop Video")
        else:
            self.timer.stop()
            self.start_stop_button.setText("Start Video")

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


def run_console_mode(enable_send: bool = False, send_endpoint: str = None):
    idle_timeout_s = float(os.getenv("BACKEND_IDLE_SEND_S", "2.5"))
    cooldown_s = float(os.getenv("BACKEND_COOLDOWN_S", "6.0"))
    segment_buffer = []
    last_new_emit_ts = None  # use monotonic timestamps when we emit new stable text
    stabilizer = TextStabilizer(threshold=2.0, boost=1.0, decay=0.85, fuzzy=0.92, cooldown_s=cooldown_s)
    # Backend selection
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GCP_API_KEY") or os.getenv("VISION_API_KEY")
    api_endpoint = os.getenv("GOOGLE_VISION_API_ENDPOINT", "https://vision.googleapis.com")

    use_rest = api_key is not None

    vision_client = None
    if not use_rest and gcv_vision is not None:
        client_options = None
        if api_endpoint and api_endpoint.startswith("https://") and ClientOptions is not None:
            endpoint_host = api_endpoint.replace("https://", "")
            client_options = ClientOptions(api_endpoint=endpoint_host)
        try:
            vision_client = gcv_vision.ImageAnnotatorClient(client_options=client_options) if client_options else gcv_vision.ImageAnnotatorClient()
        except Exception:
            use_rest = True

    ocr_interval = 0.7
    last_ocr_ts = time.monotonic() - ocr_interval  # so first loop triggers immediately
    # last_text removed in favor of stabilizer-only output

    cap = cv2.VideoCapture(5)
    if not cap.isOpened():
        print("Failed to open camera", file=sys.stderr)
        return 1

    print("Console mode started. Press Ctrl+C to stop.", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()

            
            if (now - last_ocr_ts) >= ocr_interval:
                # Encode once
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue

                if use_rest:
                    try:
                        import requests
                        content_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        url_base = api_endpoint.rstrip("/")
                        url = f"{url_base}/v1/images:annotate?key={api_key}"
                        payload = {
                            "requests": [
                                {
                                    "image": {"content": content_b64},
                                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                                }
                            ]
                        }
                        headers = {"Content-Type": "application/json"}
                        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            responses = data.get("responses", [])
                            if responses:
                                text = responses[0].get("fullTextAnnotation", {}).get("text", "").strip()
                                if text:
                                    lines = [ln for ln in text.splitlines() if ln.strip()]
                                    lines = _coalesce_lines(lines)
                                    beautified = [_beautify_display(ln) for ln in lines]
                                    stable_lines = stabilizer.update(beautified)
                                    if stable_lines:
                                        # de-dup within this emission batch (preserve order)
                                        seen_keys, unique = set(), []
                                        for s in stable_lines:
                                            s2 = _beautify_display(s)
                                            k = _norm_line(s2)
                                            if k in seen_keys:
                                                continue
                                            seen_keys.add(k)
                                            unique.append(s2)
                                        for s2 in unique:
                                            print(s2, flush=True)
                                            segment_buffer.append(s2)
                                        last_new_emit_ts = now
                    except Exception:
                        pass
                else:
                    try:
                        image = gcv_vision.Image(content=buf.tobytes())
                        response = vision_client.document_text_detection(image=image)
                        if not response.error.message:
                            text = ""
                            if response.full_text_annotation and response.full_text_annotation.text:
                                text = response.full_text_annotation.text.strip()
                            if text:
                                lines = [ln for ln in text.splitlines() if ln.strip()]
                                lines = _coalesce_lines(lines)
                                beautified = [_beautify_display(ln) for ln in lines]
                                stable_lines = stabilizer.update(beautified)
                                if stable_lines:
                                    seen_keys, unique = set(), []
                                    for s in stable_lines:
                                        s2 = _beautify_display(s)
                                        k = _norm_line(s2)
                                        if k in seen_keys:
                                            continue
                                        seen_keys.add(k)
                                        unique.append(s2)
                                    for s2 in unique:
                                        print(s2, flush=True)
                                        segment_buffer.append(s2)
                                    last_new_emit_ts = now
                    except Exception:
                        pass

                last_ocr_ts = now

            # Idle-send check: no new stable lines for idle_timeout_s
            if segment_buffer and (last_new_emit_ts is not None) and ((time.monotonic() - last_new_emit_ts) >= idle_timeout_s):
                # de-duplicate inside the segment buffer
                seen_keys, agg = set(), []
                for s in segment_buffer:
                    k = _norm_line(s)
                    if k in seen_keys:
                        continue
                    seen_keys.add(k)
                    agg.append(s)
                send_text = " ".join(agg).strip()
                if send_text:
                    print(f"(send) {send_text}", flush=True)
                    # Actually send to frontend if enabled
                    if enable_send:
                        target = (
                            send_endpoint
                            or os.getenv("BACKEND_SEND_ENDPOINT")
                            or "http://127.0.0.1:5173/api/backend/send"
                        )
                        _send_to_frontend(send_text, target)
                segment_buffer.clear()

            # Small sleep to reduce CPU usage
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    return 0


# Run the application
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting OCR")
    parser.add_argument("--console", action="store_true", help="Run in headless console mode (print recognized text)")
    parser.add_argument("--send", action="store_true", help="Enable sending aggregated idle text to frontend")
    parser.add_argument("--send-endpoint", type=str, default=None, help="Override frontend send endpoint (default env BACKEND_SEND_ENDPOINT or http://127.0.0.1:5173/api/backend/send)")
    args = parser.parse_args()

    if args.console:
        sys.exit(run_console_mode(enable_send=bool(args.send or os.getenv("BACKEND_SEND_ENDPOINT")), send_endpoint=args.send_endpoint))
    else:
        app = QApplication(sys.argv)
        window = HandwritingRecognitionApp()
        window.show()
        sys.exit(app.exec_())

