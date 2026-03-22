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
try:
    import imageio  # type: ignore
except ModuleNotFoundError:
    imageio = None
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import torch
import math
try:
    from matplotlib.figure import Figure  # type: ignore
except ModuleNotFoundError:
    Figure = None

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
        self.reference_text   = StringVar()
        self.display_bleu     = StringVar()
        self.display_rouge    = StringVar()
        self._illustration_win = None
        self._illustration_text = None
        self.nlp_backend      = StringVar(value="rules")
        self.nlp_model        = StringVar(value=DEFAULT_OLLAMA_MODEL)
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
        tk.Label(self.frame10, text='Model', bg='#b3b3b3', font=self.button_font).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame10, textvariable=self.nlp_model, font=self.button_font, width=30).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame10, textvariable=self.display_refined,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)

        # Evaluation (BLEU / ROUGE)
        self.frame11 = tk.Frame(self, relief=GROOVE, borderwidth=1)
        self.frame11.pack(fill=X)
        self.control_frames.append(self.frame11)
        tk.Label(self.frame11, text='Reference text', bg='#b3b3b3', font=self.button_font).pack(side=LEFT, padx=4, pady=3)
        Entry(self.frame11, textvariable=self.reference_text,
              font=self.font).pack(fill=X, padx=4, pady=2, expand=True)
        tk.Button(self.frame11, text='Compute BLEU/ROUGE', bg='#b3b3b3', font=self.button_font,
                  command=self.compute_scores, padx=6, pady=2).pack(side=LEFT, padx=4, pady=3)
        tk.Button(self.frame11, text='Minh hoạ kết quả', bg='#b3b3b3', font=self.button_font,
                  command=self.show_result_illustration, padx=6, pady=2).pack(side=LEFT, padx=2, pady=3)
        tk.Button(self.frame11, text='Xuất graph (PNG)', bg='#b3b3b3', font=self.button_font,
                  command=self.export_metric_graph, padx=6, pady=2).pack(side=LEFT, padx=2, pady=3)
        tk.Button(self.frame11, text='Demo', bg='#b3b3b3', font=self.button_font,
                  command=self.show_metric_demo, padx=6, pady=2).pack(side=LEFT, padx=2, pady=3)
        tk.Label(self.frame11, textvariable=self.display_bleu, bg='#d9d9d9', font=self.font0, width=12).pack(side=LEFT, padx=4, pady=2)
        tk.Label(self.frame11, textvariable=self.display_rouge, bg='#d9d9d9', font=self.font0, width=12).pack(side=LEFT, padx=4, pady=2)

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
            if imageio is not None:
                video = imageio.get_reader(path)
                for image in video.iter_data():
                    image = cv2.resize(image, (1200, 650))
                    frame_image = ImageTk.PhotoImage(Image.fromarray(image))
                    self.my_label.config(image=frame_image)
                    self.my_label.image = frame_image
                return

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(
                    "Không mở được video. Cài `imageio` hoặc kiểm tra đường dẫn/tệp video."
                )
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (1200, 650))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = ImageTk.PhotoImage(Image.fromarray(frame))
                self.my_label.config(image=frame_image)
                self.my_label.image = frame_image
            cap.release()
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
        model = (self.nlp_model.get() or "").strip() or DEFAULT_OLLAMA_MODEL
        try:
            return refine_text(text, backend=backend, model=model)
        except Exception as e:
            self._set_status(f"NLP {backend} lỗi, fallback rules: {e}")
            if backend != "rules":
                self.nlp_backend.set("rules")
                self.backend_combo.configure(values=available_backends())
            return refine_text(text, backend="rules")

    # ── BLEU / ROUGE ──
    @staticmethod
    def _tokenize(text):
        return [t for t in text.lower().split() if t]

    def _bleu_score(self, hypothesis, reference, max_n=4):
        hyp = self._tokenize(hypothesis)
        ref = self._tokenize(reference)
        if not hyp or not ref:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1))
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = max(len(hyp) - n + 1, 1)
            precisions.append((overlap + 1e-9) / total)  # smoothing to avoid zero

        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
        bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
        return bp * geo_mean

    def _bleu_components(self, hypothesis, reference, max_n=4):
        hyp = self._tokenize(hypothesis)
        ref = self._tokenize(reference)
        if not hyp or not ref:
            return 0.0, 0.0, [0.0] * max_n

        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1))
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = max(len(hyp) - n + 1, 1)
            precisions.append((overlap + 1e-9) / total)

        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
        bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
        return bp * geo_mean, bp, precisions

    def _lcs_length(self, a, b):
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[m][n]

    def _lcs_sequence(self, a, b):
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return []
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        i, j = m, n
        seq = []
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                seq.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        seq.reverse()
        return seq

    def _lcs_match_indices(self, a, b):
        """
        Return one possible LCS alignment as index lists:
        - indices in `a` and `b` that participate in the LCS.
        """
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return [], []

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        i, j = m, n
        a_idx = []
        b_idx = []
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                a_idx.append(i - 1)
                b_idx.append(j - 1)
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        a_idx.reverse()
        b_idx.reverse()
        return a_idx, b_idx

    def _rouge_l(self, hypothesis, reference):
        hyp = self._tokenize(hypothesis)
        ref = self._tokenize(reference)
        if not hyp or not ref:
            return 0.0
        lcs = self._lcs_length(hyp, ref)
        prec = lcs / len(hyp)
        rec  = lcs / len(ref)
        if prec == 0 and rec == 0:
            return 0.0
        return (2 * prec * rec) / (prec + rec)

    def compute_scores(self):
        reference = self.reference_text.get().strip()
        candidate = self.display_refined.get().strip() or self.display_sequence.get().strip()
        if not candidate:
            self._set_status("Không có câu dự đoán để chấm điểm.")
            return
        if not reference:
            self._set_status("Nhập Reference text để tính điểm.")
            return

        bleu  = self._bleu_score(candidate, reference)
        rouge = self._rouge_l(candidate, reference)
        self.display_bleu.set(f"BLEU: {bleu*100:.1f}")
        self.display_rouge.set(f"ROUGE-L: {rouge*100:.1f}")
        self._set_status("Đã tính BLEU/ROUGE.")
        self._update_result_illustration_if_open()

    def _current_candidate_reference(self):
        reference = self.reference_text.get().strip()
        candidate = self.display_refined.get().strip() or self.display_sequence.get().strip()
        return candidate, reference

    def _build_metric_figure(self, candidate, reference):
        if Figure is None:
            raise RuntimeError("Thiếu matplotlib. Cài `matplotlib` để xuất graph.")

        bleu, bp, precisions = self._bleu_components(candidate, reference)
        rouge = self._rouge_l(candidate, reference)

        fig = Figure(figsize=(9.5, 5.2), dpi=120)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Top: main scores
        labels = ["BLEU", "ROUGE-L"]
        values = [bleu * 100.0, rouge * 100.0]
        colors = ["#2a6fdb", "#0b6b2b"]
        ax1.bar(labels, values, color=colors)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Score (%)")
        ax1.set_title("Evaluation Scores")
        for i, v in enumerate(values):
            ax1.text(i, min(v + 2, 98), f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

        # Bottom: BLEU components
        p_labels = ["p1", "p2", "p3", "p4"]
        p_vals = [p * 100.0 for p in precisions]
        ax2.bar(p_labels, p_vals, color="#666666")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Precision (%)")
        ax2.set_title(f"BLEU components (BP={bp:.3f})")
        for i, v in enumerate(p_vals):
            ax2.text(i, min(v + 2, 98), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        return fig

    def export_metric_graph(self):
        candidate, reference = self._current_candidate_reference()
        if not candidate or not reference:
            self._set_status("Cần có cả Candidate (dự đoán) và Reference text để xuất graph.")
            return

        if Figure is None:
            self._set_status("Thiếu matplotlib nên chưa xuất graph được. Cài `matplotlib` trước nhé.")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            title="Lưu graph BLEU/ROUGE (PNG)",
        )
        if not out_path:
            return

        try:
            fig = self._build_metric_figure(candidate, reference)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            self._set_status(f"Đã lưu graph: {out_path}")
        except Exception as e:
            self._set_status(f"Lỗi xuất graph: {e}")

    def _update_result_illustration_if_open(self):
        if self._illustration_win is None or self._illustration_text is None:
            return
        try:
            self._render_result_illustration(self._illustration_text)
        except Exception:
            # If anything goes weird, don't crash the GUI
            pass

    def _render_highlighted_tokens(self, text_widget, tokens, match_set, tag_match, tag_other):
        for i, tok in enumerate(tokens):
            tag = tag_match if i in match_set else tag_other
            text_widget.insert("end", tok, tag)
            if i != len(tokens) - 1:
                text_widget.insert("end", " ")

    def _render_result_illustration(self, text_widget):
        candidate, reference = self._current_candidate_reference()
        text_widget.configure(state="normal")
        text_widget.delete("1.0", "end")

        if not candidate or not reference:
            text_widget.insert(
                "end",
                "Cần có cả Candidate (câu dự đoán) và Reference (câu chuẩn).\n"
                "1) Bấm Predict sequence để có kết quả.\n"
                "2) Nhập Reference text.\n"
                "3) Bấm Minh hoạ kết quả.\n",
            )
            text_widget.configure(state="disabled")
            return

        cand_tokens = self._tokenize(candidate)
        ref_tokens = self._tokenize(reference)
        cand_idx, ref_idx = self._lcs_match_indices(cand_tokens, ref_tokens)
        cand_match = set(cand_idx)
        ref_match = set(ref_idx)

        bleu, bp, precisions = self._bleu_components(candidate, reference)
        rouge = self._rouge_l(candidate, reference)

        text_widget.insert("end", "Minh hoạ so khớp (theo ROUGE-L/LCS)\n", "hdr")
        text_widget.insert("end", f"BLEU={bleu*100:.1f}% (BP={bp:.3f}, p1..p4={[round(p, 3) for p in precisions]})\n", "meta")
        text_widget.insert("end", f"ROUGE-L={rouge*100:.1f}%\n\n", "meta")

        text_widget.insert("end", "Reference:  ", "lbl")
        self._render_highlighted_tokens(text_widget, ref_tokens, ref_match, "match", "missing")
        text_widget.insert("end", "\n")

        text_widget.insert("end", "Candidate:  ", "lbl")
        self._render_highlighted_tokens(text_widget, cand_tokens, cand_match, "match", "extra")
        text_widget.insert("end", "\n\n")

        extras = [t for i, t in enumerate(cand_tokens) if i not in cand_match]
        missing = [t for i, t in enumerate(ref_tokens) if i not in ref_match]

        text_widget.insert("end", "Chú giải màu:\n", "hdr2")
        text_widget.insert("end", "Xanh = từ khớp theo đúng thứ tự (LCS)\n", "match")
        text_widget.insert("end", "Đỏ = từ dư (có trong Candidate nhưng không nằm trong LCS)\n", "extra")
        text_widget.insert("end", "Cam = từ thiếu (có trong Reference nhưng không nằm trong LCS)\n\n", "missing")

        text_widget.insert("end", f"Từ dư (Candidate): {extras}\n", "extra")
        text_widget.insert("end", f"Từ thiếu (Reference): {missing}\n", "missing")

        text_widget.configure(state="disabled")

    def show_result_illustration(self):
        if self._illustration_win is not None and self._illustration_text is not None:
            try:
                self._illustration_win.lift()
                self._render_result_illustration(self._illustration_text)
                return
            except Exception:
                self._illustration_win = None
                self._illustration_text = None

        win = tk.Toplevel(self.master)
        win.title("Minh hoạ kết quả BLEU / ROUGE-L")
        win.geometry("980x560")
        self._illustration_win = win

        text = tk.Text(win, wrap="word", font=self.font)
        text.pack(fill=BOTH, expand=True)
        self._illustration_text = text

        text.tag_configure("hdr", font=self.font, foreground="#222222")
        text.tag_configure("hdr2", font=self.font, foreground="#222222")
        text.tag_configure("meta", font=self.font0, foreground="#333333")
        text.tag_configure("lbl", font=self.font, foreground="#222222")
        text.tag_configure("match", foreground="#0b6b2b")
        text.tag_configure("extra", foreground="#b00020")
        text.tag_configure("missing", foreground="#b15a00")

        def _on_close():
            self._illustration_win = None
            self._illustration_text = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)
        self._render_result_illustration(text)

    def show_metric_demo(self):
        win = tk.Toplevel(self.master)
        win.title("Demo BLEU / ROUGE-L")
        win.geometry("900x520")

        text = tk.Text(win, wrap="word", font=self.font)
        text.pack(fill=BOTH, expand=True)

        def _write(s=""):
            text.insert("end", s + "\n")

        _write("Mục tiêu: minh hoạ cách BLEU và ROUGE-L được tính (đơn giản hoá, token = tách theo dấu cách).")
        _write("Gợi ý dùng trong app: nhập 'Reference text' (câu đúng) -> dự đoán -> bấm 'Compute BLEU/ROUGE'.")
        _write("")

        examples = [
            ("Tôi đi học", "Tôi đi học"),
            ("Tôi đi học", "Tôi đi làm"),
            ("Hôm nay tôi đi học ở trường", "Hôm nay tôi đi học"),
        ]

        for idx, (ref, hyp) in enumerate(examples, start=1):
            hyp_tokens = self._tokenize(hyp)
            ref_tokens = self._tokenize(ref)
            bleu, bp, precisions = self._bleu_components(hyp, ref)
            rouge = self._rouge_l(hyp, ref)
            lcs_seq = self._lcs_sequence(hyp_tokens, ref_tokens)

            _write(f"Ví dụ {idx}:")
            _write(f"  Reference:  {ref}")
            _write(f"  Candidate:  {hyp}")
            _write(f"  Tokens(ref): {ref_tokens}")
            _write(f"  Tokens(cand): {hyp_tokens}")
            _write(f"  ROUGE-L ~ LCS tokens: {lcs_seq}  -> ROUGE-L = {rouge*100:.1f}%")
            _write(f"  BLEU: BP={bp:.3f}, p1..p4={[round(p, 3) for p in precisions]} -> BLEU = {bleu*100:.1f}%")
            _write("")

        _write("Cách hiểu nhanh:")
        _write("- BLEU cao khi cụm từ 1-4 từ trong Candidate trùng Reference nhiều (và không quá ngắn).")
        _write("- ROUGE-L cao khi Candidate giữ được nhiều từ theo đúng thứ tự của Reference (LCS dài).")

        text.configure(state="disabled")

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
        nlp_model = (self.nlp_model.get() or "").strip() or DEFAULT_OLLAMA_MODEL

        def _run():
            self._set_status("Đang dự đoán chuỗi từ...")
            try:
                result, refined = predict_sequence(
                    video_path  = video_path,
                    model_path  = MODEL_PATH,
                    output_path = output_txt,
                    nlp_backend = backend,
                    nlp_model   = nlp_model,
                )
                self.display_sequence.set(result)
                self.display_refined.set(refined)
                if self.reference_text.get().strip():
                    self.compute_scores()
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
