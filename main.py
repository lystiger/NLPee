# -*- coding: utf-8 -*-
"""
main.py — GUI Tkinter cho hệ thống nhận diện ngôn ngữ ký hiệu
Tích hợp PyTorch + MediaPipe (2 tay, không cần YOLO)
"""

from tkinter import Tk, RIGHT, LEFT, BOTH, X, filedialog, StringVar, GROOVE
from tkinter.ttk import Style, Entry, Combobox
import tkinter.font as TkFont
import tkinter as tk
import threading
from collections import Counter, deque
import imageio
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import torch

from predict_word     import predict_single_video, CLASSES
from predict_sequence import predict_sequence, pad_or_trim
from extractor import HandKeypointExtractor, FEATURE_DIM, SEQ_LENGTH
from model_bundle import DEFAULT_MODEL_PATH, load_model
from nlp_refiner import DEFAULT_OLLAMA_MODEL, available_backends, refine_text


# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
MODEL_PATH = str(DEFAULT_MODEL_PATH)
CACHE_DIR  = 'cache/gui/'


class Window(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, bg='#b3b3b3')
        self.master = master
        self.model = None
        self.extractor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.webcam = None
        self.webcam_running = False
        self.feature_buffer = deque(maxlen=SEQ_LENGTH)
        self.prediction_history = deque(maxlen=5)
        self.frame_counter = 0
        self.infer_every_n_frames = 3
        self.current_prediction = ''
        self.live_segment = []
        self.silence_counter = 0
        self.sequence_words = []
        self.min_segment_frames = 8
        self.silence_threshold = 10
        self.segment_status = 'Ready'
        self.init_window()

    def init_window(self):
        self.master.title("Viet Nam Sign Language Translator")
        self.font0 = TkFont.Font(self, size=10)
        self.font  = TkFont.Font(self, size=11)
        self.button_font = TkFont.Font(self, size=10)
        self.style = Style()
        self.style.theme_use("clam")
        self.pack(fill=BOTH, expand=1)

        self.inputfilepath    = StringVar()
        self.inputvideofile   = StringVar()
        self.outputfilepath   = StringVar()
        self.outputvideofile  = StringVar()
        self.display_sign     = StringVar()
        self.display_sequence = StringVar()
        self.display_refined  = StringVar()
        self.nlp_backend      = StringVar(value="rules")
        self.control_frames = []

        # Input directory
        self.frame1 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame1.pack(fill=X)
        self.control_frames.append(self.frame1)
        tk.Button(self.frame1, text='Input directory', bg='#b3b3b3', font=self.button_font,
                  command=self.input_browser, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame1, textvariable=self.inputfilepath,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Input video file
        self.frame2 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame2.pack(fill=X)
        self.control_frames.append(self.frame2)
        tk.Button(self.frame2, text='Input video file', bg='#b3b3b3', font=self.button_font,
                  command=self.inputvideo_browser, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame2, textvariable=self.inputvideofile,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Output directory
        self.frame3 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame3.pack(fill=X)
        self.control_frames.append(self.frame3)
        tk.Button(self.frame3, text='Output directory', bg='#b3b3b3', font=self.button_font,
                  command=self.output_browser, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame3, textvariable=self.outputfilepath,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Output video file
        self.frame9 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame9.pack(fill=X)
        self.control_frames.append(self.frame9)
        tk.Button(self.frame9, text='Output video file', bg='#b3b3b3', font=self.button_font,
                  command=self.outputvideo_browser, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame9, textvariable=self.outputvideofile,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Action buttons
        self.frame4 = tk.Frame(self)
        self.frame4.pack(fill=X)
        self.control_frames.append(self.frame4)
        tk.Button(self.frame4, text='Open input video',  bg='#b3b3b3', font=self.button_font,
                  command=self.open_invideo,  padx=6, pady=2).grid(row=0, column=0, padx=3, pady=3)
        tk.Button(self.frame4, text='Extract Features',  bg='#b3b3b3', font=self.button_font,
                  command=self.extract_features, padx=6, pady=2).grid(row=0, column=1, padx=3, pady=3)
        tk.Button(self.frame4, text='Open output video', bg='#b3b3b3', font=self.button_font,
                  command=self.open_outvideo, padx=6, pady=2).grid(row=0, column=2, padx=3, pady=3)
        tk.Button(self.frame4, text='Reset',             bg='#b3b3b3', font=self.button_font,
                  command=self.reset, padx=6, pady=2).grid(row=0, column=3, padx=3, pady=3)
        self.start_webcam_button = tk.Button(self.frame4, text='Start webcam', bg='#b3b3b3', font=self.button_font,
                  command=self.start_webcam, padx=6, pady=2)
        self.start_webcam_button.grid(row=0, column=4, padx=3, pady=3)
        self.stop_webcam_button = tk.Button(self.frame4, text='Stop webcam', bg='#b3b3b3', font=self.button_font,
                  command=self.stop_webcam, padx=6, pady=2)
        self.stop_webcam_button.grid(row=0, column=5, padx=3, pady=3)

        # Predict sign
        self.frame5 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame5.pack(fill=X)
        self.control_frames.append(self.frame5)
        tk.Button(self.frame5, text='Predict sign', bg='#b3b3b3', font=self.button_font,
                  command=self.sign_predict, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame5, textvariable=self.display_sign,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Predict sequence
        self.frame6 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame6.pack(fill=X)
        self.control_frames.append(self.frame6)
        tk.Button(self.frame6, text='Predict sequence', bg='#b3b3b3', font=self.button_font,
                  command=self.sequence_predict, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame6, textvariable=self.display_sequence,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Refined text / NLP backend
        self.frame10 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame10.pack(fill=X)
        self.control_frames.append(self.frame10)
        tk.Label(self.frame10, text='NLP backend', bg='#b3b3b3', font=self.button_font).pack(side=LEFT, padx=4, pady=3)
        self.backend_combo = Combobox(
            self.frame10,
            textvariable=self.nlp_backend,
            values=available_backends(),
            state='readonly',
            width=12,
            font=self.button_font,
        )
        self.backend_combo.pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame10, textvariable=self.display_refined,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Video display
        self.webcam_toolbar = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.webcam_stop_only_button = tk.Button(
            self.webcam_toolbar,
            text='Stop webcam',
            bg='#b3b3b3',
            font=self.button_font,
            command=self.stop_webcam,
            padx=8,
            pady=2,
        )
        self.webcam_stop_only_button.pack(side=LEFT, padx=4, pady=3)
        self.webcam_word_label = tk.Label(
            self.webcam_toolbar,
            textvariable=self.display_sign,
            bg='#d9d9d9',
            font=self.font,
            anchor='w',
        )
        self.webcam_word_label.pack(side=LEFT, fill=X, expand=True, padx=4, pady=2)
        self.webcam_refined_label = tk.Label(
            self.webcam_toolbar,
            textvariable=self.display_refined,
            bg='#d9d9d9',
            font=self.font,
            anchor='w',
        )
        self.webcam_refined_label.pack(side=LEFT, fill=X, expand=True, padx=4, pady=2)

        self.frame7 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame7.pack(fill=BOTH, expand=True)
        self.my_label = tk.Label(self.frame7)
        self.my_label.pack(fill=BOTH, expand=True)

        # Status bar
        self.status_var = StringVar(value="Sẵn sàng.")
        tk.Label(self, textvariable=self.status_var, bg='#d9d9d9',
                 font=self.font0, anchor='w').pack(fill=X, padx=5)

        # Close button
        self.frame8 = tk.Frame(self)
        self.frame8.pack(fill=X)
        self.control_frames.append(self.frame8)
        tk.Button(self.frame8, text='Close', font=self.font0,
                  command=self.close_window, bg='#b3b3b3', padx=6, pady=2).pack(side=RIGHT, padx=4, pady=3)

    # ── Browsers ──
    def _ask_directory(self):
        return filedialog.askdirectory()

    def _ask_video(self):
        return filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.MOV"), ("All files", "*.*")]
        )

    def input_browser(self):
        d = self._ask_directory()
        if d: self.inputfilepath.set(d)

    def output_browser(self):
        d = self._ask_directory()
        if d: self.outputfilepath.set(d)

    def inputvideo_browser(self):
        f = self._ask_video()
        if f: self.inputvideofile.set(f)

    def outputvideo_browser(self):
        f = self._ask_video()
        if f: self.outputvideofile.set(f)

    # ── Video playback ──
    def _stream_video(self, path):
        try:
            video = imageio.get_reader(path)
            for image in video.iter_data():
                image = cv2.resize(image, (1200, 650))
                frame_image = ImageTk.PhotoImage(Image.fromarray(image))
                self.my_label.config(image=frame_image)
                self.my_label.image = frame_image
        except Exception as e:
            self._set_status(f"Lỗi phát video: {e}")

    def open_invideo(self):
        path = self.inputvideofile.get()
        if not path:
            self._set_status("Chưa chọn video đầu vào.")
            return
        threading.Thread(target=self._stream_video, args=(path,), daemon=True).start()

    def open_outvideo(self):
        path = self.outputvideofile.get()
        if not path:
            self._set_status("Chưa chọn video đầu ra.")
            return
        threading.Thread(target=self._stream_video, args=(path,), daemon=True).start()

    # ── Webcam realtime ──
    def _ensure_realtime_runtime(self):
        if self.model is None:
            self.model, _ = load_model(model_path=MODEL_PATH, device=self.device)
        if self.extractor is None:
            self.extractor = HandKeypointExtractor()

    def _set_webcam_mode(self, enabled):
        if enabled:
            for frame in self.control_frames:
                frame.pack_forget()
            self.webcam_toolbar.pack_forget()
            self.frame7.pack_forget()
            self.webcam_toolbar.pack(fill=X)
            self.frame7.pack(fill=BOTH, expand=True)
        else:
            self.webcam_toolbar.pack_forget()
            self.frame7.pack_forget()
            self.frame1.pack(fill=X)
            self.frame2.pack(fill=X)
            self.frame3.pack(fill=X)
            self.frame9.pack(fill=X)
            self.frame4.pack(fill=X)
            self.frame5.pack(fill=X)
            self.frame6.pack(fill=X)
            self.frame10.pack(fill=X)
            self.frame7.pack(fill=BOTH, expand=True)
            self.frame8.pack(fill=X)

    def start_webcam(self):
        if self.webcam_running:
            self._set_status("Webcam đang chạy.")
            return

        try:
            self._ensure_realtime_runtime()
        except Exception as e:
            self._set_status(f"Không thể khởi tạo realtime runtime: {e}")
            return

        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            self.webcam = None
            self._set_status("Không mở được webcam.")
            return

        self.webcam_running = True
        self.feature_buffer.clear()
        self.prediction_history.clear()
        self.frame_counter = 0
        self.current_prediction = ''
        self.live_segment = []
        self.silence_counter = 0
        self.sequence_words = []
        self.segment_status = 'Ready'
        self.display_sign.set('')
        self.display_sequence.set('')
        self.display_refined.set('')
        self._set_webcam_mode(True)
        self._set_status("Đã bật webcam realtime.")
        self._update_webcam_frame()

    def stop_webcam(self):
        self.webcam_running = False
        if self.webcam is not None:
            self.webcam.release()
            self.webcam = None
        self._set_webcam_mode(False)
        self._set_status("Đã dừng webcam.")

    def _predict_realtime_label(self):
        if len(self.feature_buffer) < SEQ_LENGTH:
            return

        feat = np.stack(self.feature_buffer, axis=0).astype(np.float32)
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.model(x)
            pred_idx = out.argmax(1).item()

        self.prediction_history.append(CLASSES[pred_idx])
        stable_label = Counter(self.prediction_history).most_common(1)[0][0]
        self.current_prediction = stable_label
        self.display_sign.set(stable_label)

    def _commit_live_segment(self):
        if len(self.live_segment) < self.min_segment_frames:
            self.live_segment = []
            self.silence_counter = 0
            return

        feat = pad_or_trim(np.stack(self.live_segment, axis=0).astype(np.float32), SEQ_LENGTH)
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            out = self.model(x)
            pred_idx = out.argmax(1).item()

        word = CLASSES[pred_idx]
        self.sequence_words.append(word)
        self.display_sequence.set(' '.join(self.sequence_words))
        self.display_sign.set(word)
        self.current_prediction = word
        self.live_segment = []
        self.silence_counter = 0
        self.display_refined.set(self._refine_sequence_text(' '.join(self.sequence_words)))
        self.segment_status = f"Captured: {word}"

    def _refine_sequence_text(self, text):
        backend = self.nlp_backend.get() or "rules"
        try:
            return refine_text(text, backend=backend, model=DEFAULT_OLLAMA_MODEL)
        except Exception as e:
            self._set_status(f"NLP {backend} lỗi, fallback rules: {e}")
            if backend != "rules":
                self.nlp_backend.set("rules")
                self.backend_combo.configure(values=available_backends())
            return refine_text(text, backend="rules")

    def _update_webcam_frame(self):
        if not self.webcam_running or self.webcam is None:
            return

        ok, frame = self.webcam.read()
        if not ok:
            self._set_status("Không đọc được frame từ webcam.")
            self.stop_webcam()
            return

        frame = cv2.flip(frame, 1)
        feat = self.extractor.extract(frame)
        self.feature_buffer.append(feat)

        has_hand = not np.allclose(feat, 0.0)
        self.frame_counter += 1

        if has_hand:
            self.live_segment.append(feat)
            self.silence_counter = 0
            self.segment_status = "Sign word"
            if len(self.feature_buffer) == SEQ_LENGTH and self.frame_counter % self.infer_every_n_frames == 0:
                self._predict_realtime_label()
        elif self.live_segment:
            self.silence_counter += 1
            pause_pct = int((self.silence_counter / self.silence_threshold) * 100)
            pause_pct = min(pause_pct, 100)
            self.segment_status = f"Pause to confirm: {pause_pct}%"
            if self.silence_counter >= self.silence_threshold:
                self._commit_live_segment()
        else:
            self.segment_status = "Next word"

        display_frame = frame.copy()
        live_text = self.current_prediction if self.current_prediction else "Detecting..."
        cv2.putText(
            display_frame,
            f"Prediction: {live_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        status_color = (255, 220, 0) if has_hand else (0, 200, 255)
        cv2.putText(
            display_frame,
            self.segment_status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if not has_hand:
            hint_text = "Pause, then move to the next word"
        else:
            hint_text = "Hold the sign steady"

        sequence_text = self.display_sequence.get() or ""
        refined_text = self.display_refined.get() or ""
        cv2.putText(
            display_frame,
            f"Sequence: {sequence_text}"[:80],
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_frame,
            f"Refined: {refined_text}"[:80],
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 255, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display_frame,
            hint_text,
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (1200, 650))
        frame_image = ImageTk.PhotoImage(Image.fromarray(display_frame))
        self.my_label.config(image=frame_image)
        self.my_label.image = frame_image

        status_msg = "Webcam realtime đang chạy"
        if self.current_prediction:
            status_msg += f" - {self.current_prediction}"
        if self.sequence_words:
            status_msg += f" | {' '.join(self.sequence_words)}"
        status_msg += f" | {self.segment_status}"
        self._set_status(status_msg)
        self.master.after(15, self._update_webcam_frame)

    # ── Extract Features ──
    def extract_features(self):
        input_dir  = self.inputfilepath.get()
        output_dir = self.outputfilepath.get()
        if not input_dir or not output_dir:
            self._set_status("Cần chọn cả Input directory và Output directory.")
            return

        def _run():
            from extractor import HandKeypointExtractor, VideoFeatureCache, precompute_features, SEQ_LENGTH
            self._set_status("Đang trích xuất features... (có thể mất vài phút)")
            try:
                vids = []
                for cls in CLASSES:
                    cls_dir = os.path.join(input_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    for f in os.listdir(cls_dir):
                        if f.lower().endswith(('.mp4', '.avi', '.mov')):
                            vids.append(os.path.join(cls_dir, f))

                if not vids:
                    self._set_status("Không tìm thấy video nào trong Input directory.")
                    return

                cache_dir = os.path.join(output_dir, 'cache')
                extractor = HandKeypointExtractor()
                cache     = VideoFeatureCache(cache_dir)
                precompute_features(vids, extractor, cache, seq_length=SEQ_LENGTH)
                del extractor
                self._set_status(f"Trích xuất xong {len(vids)} videos → {cache_dir}")
            except Exception as e:
                self._set_status(f"Lỗi trích xuất: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Predict sign (1 từ) ──
    def sign_predict(self):
        video_path = self.inputvideofile.get()
        if not video_path:
            self._set_status("Chưa chọn video đầu vào.")
            return

        def _run():
            self._set_status("Đang dự đoán từ...")
            try:
                result = predict_single_video(
                    video_path = video_path,
                    model_path = MODEL_PATH,
                    cache_dir  = CACHE_DIR,
                )
                self.display_sign.set(result)
                self._set_status(f"Dự đoán xong: {result}")
            except Exception as e:
                self._set_status(f"Lỗi dự đoán: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Predict sequence (chuỗi từ) ──
    def sequence_predict(self):
        video_path = self.inputvideofile.get()
        if not video_path:
            self._set_status("Chưa chọn video đầu vào.")
            return

        output_dir = self.outputfilepath.get()
        output_txt = os.path.join(output_dir, 'sequence.txt') if output_dir else None
        backend = self.nlp_backend.get() or "rules"

        def _run():
            self._set_status("Đang dự đoán chuỗi từ...")
            try:
                result, refined = predict_sequence(
                    video_path  = video_path,
                    model_path  = MODEL_PATH,
                    output_path = output_txt,
                    nlp_backend = backend,
                )
                self.display_sequence.set(result)
                self.display_refined.set(refined)
                self._set_status(f"Dự đoán xong: {refined or result}")
            except Exception as e:
                self._set_status(f"Lỗi dự đoán chuỗi: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Helpers ──
    def _set_status(self, msg):
        self.master.after(0, lambda: self.status_var.set(msg))

    def reset(self):
        self.stop_webcam()
        self.inputfilepath.set('')
        self.inputvideofile.set('')
        self.outputfilepath.set('')
        self.outputvideofile.set('')
        self.display_sign.set('')
        self.display_sequence.set('')
        self.display_refined.set('')
        self._set_status("Đã reset.")

    def close_window(self):
        self.stop_webcam()
        self.master.destroy()


if __name__ == '__main__':
    form = Tk()
    screen_w = form.winfo_screenwidth()
    screen_h = form.winfo_screenheight()
    width = int(screen_w * 0.7)
    height = int(screen_h * 0.7)
    x = max((screen_w - width) // 2, 0)
    y = max((screen_h - height) // 2, 0)
    form.geometry(f"{width}x{height}+{x}+{y}")
    app = Window(form)
    form.mainloop()
