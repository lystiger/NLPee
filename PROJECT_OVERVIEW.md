
# Vietnamese Sign Language Demo Overview

## 1. Project Goal

This project is a Vietnamese Sign Language recognition demo built around:

- computer vision for hand landmark extraction
- sequence classification for sign recognition
- lightweight rule-based NLP refinement for more natural Vietnamese text

The system can work in two modes:

- offline video prediction from an input `.mp4`
- live webcam prediction with word-by-word sequence building

## 2. Main Technologies

### Core stack

- Python 3.12
- OpenCV
- NumPy
- Tkinter
- Pillow

### Vision / feature extraction

- MediaPipe Hands
- 2-hand landmark extraction
- 126 features per frame
  - left hand: 63
  - right hand: 63

### Deep learning

- PyTorch
- BiLSTM classifier
- exported model bundle in `output/_output`

### NLP layer

- rule-based Vietnamese sentence refinement
- duplicate cleanup
- word-order cleanup
- punctuation
- Vietnamese diacritic restoration

## 3. Model Overview

The exported sign model is a BiLSTM classifier trained on fixed-length sign sequences.

### Input

- shape: `1 x 35 x 126`
- `35` frames per sign
- `126` landmark features per frame

### Output classes

- `BAN`
- `CAM_DIEC`
- `DI`
- `GI`
- `HOC`
- `HOM_TRUOC`
- `KHONG`
- `THICH`
- `TOI`

### Exported model files

- `output/_output/vsl_bilstm_state_dict.pth`
- `output/_output/vsl_bilstm_metadata.json`
- `output/_output/model.py`

## 4. Current System Architecture

### A. Offline video path

1. User selects an input video
2. OpenCV reads video frames
3. MediaPipe extracts 2-hand landmarks per frame
4. Sequence segmentation tries to detect word boundaries from silent gaps
5. Each segment is padded or trimmed to 35 frames
6. BiLSTM predicts one sign label per segment
7. Predicted labels are joined into a raw sequence
8. Rule-based NLP refines the raw sequence into more natural Vietnamese text

### B. Live webcam path

1. User clicks `Start webcam`
2. Webcam opens with OpenCV
3. MediaPipe extracts hand landmarks frame by frame
4. A rolling buffer is used for live single-word prediction
5. A short pause between words is used to confirm a segment
6. Confirmed words are appended into a live sequence
7. Rule-based NLP refines the live sequence

Important:

- live webcam sequence is not fully continuous sentence recognition
- it is currently word-by-word live sequence entry
- user flow is:
  - sign word
  - pause to confirm
  - next word

## 5. Important Source Files

- `main.py`
  - Tkinter GUI
  - webcam mode
  - live sequence flow

- `extractor.py`
  - MediaPipe hand landmark extraction
  - feature extraction logic

- `predict_word.py`
  - single-video sign prediction

- `predict_sequence.py`
  - offline sequence prediction from video

- `model_bundle.py`
  - exported model loading
  - metadata-driven runtime configuration

- `nlp_refiner.py`
  - rule-based NLP cleanup
  - Vietnamese diacritic restoration

## 6. How To Run

### Activate environment

```bash
source venv/bin/activate
```

### Start GUI

```bash
python main.py
```

## 7. How To Use The GUI

### A. Offline single-word prediction

1. Click `Input video file`
2. Choose a video
3. Click `Predict sign`

### B. Offline sequence prediction

1. Click `Input video file`
2. Choose a multi-sign video
3. Click `Predict sequence`

You will get:

- raw predicted sequence
- refined Vietnamese sentence

### C. Live webcam prediction

1. Click `Start webcam`
2. Show a sign clearly
3. For sequence mode:
   - sign word
   - pause to confirm
   - next word
4. Click `Stop webcam` when finished

## 8. Live Webcam Guidance

The webcam overlay uses these cues:

- `Sign word`
- `Pause to confirm: XX%`
- `Next word`

Recommended user behavior:

- keep one hand clearly visible
- sign at a moderate speed
- avoid very slow exaggerated signing
- pause briefly between words
- keep background simple if possible

## 9. NLP Refinement Layer

The project includes a lightweight rule-based NLP layer.

### What it does

- removes duplicate adjacent words
- reorders `HOM_TRUOC` to the front
- applies simple phrase rules
- adds punctuation
- restores Vietnamese diacritics

### Example

- raw: `TOI TOI DI HOC`
- refined: `Tôi đi học.`

- raw: `BAN THICH GI`
- refined: `Bạn thích gì?`

### Important limitation

The NLP layer does not fix wrong recognition upstream.

If the sign recognizer produces wrong labels, the refined sentence may still be wrong.

## 10. Known Error Cases / Limitations

### A. Slow signing

Problem:

- the model was trained on fixed-length short clips
- very slow signing can confuse segmentation and classification

Symptoms:

- one word becomes multiple predicted words
- wrong extra words appear

### B. Noisy word boundaries

Problem:

- the sequence path uses silent-gap heuristics
- hand dropouts or transition motion can be treated as new words

Symptoms:

- extra segments
- duplicated or strange sequences

### C. Background / hand visibility issues

Problem:

- MediaPipe may fail if the hand is too small, too dark, blurred, or blended into the background

Symptoms:

- no detection
- unstable predictions
- sequence confirmation fails

### D. Recognition is classification, not full NLP

Problem:

- this system recognizes from a fixed vocabulary
- it does not understand full natural language semantics

Symptoms:

- sentence may be readable but semantically odd
- grammar improvement is limited by raw prediction quality

### E. Live webcam sequence is turn-based

Problem:

- the webcam mode needs a short pause between words

Symptoms:

- continuous unpaused signing may not separate words correctly

## 11. Common Troubleshooting

### Webcam opens but does not detect signs

Try:

- move hand closer to camera
- use brighter lighting
- reduce cluttered background
- hold the sign more steadily

### Sequence output contains extra words

Try:

- sign faster and more compactly
- avoid long transitions
- pause clearly between words

### Offline video works better than webcam

This is normal.

Reason:

- offline sequence prediction has the full clip available
- webcam mode must segment in real time

### Refined sentence still looks strange

Reason:

- NLP only cleans the predicted words
- it cannot fully repair incorrect recognition

## 12. Demo Notes

Best clips for demo:

- short
- clear
- moderate speed
- strong hand visibility
- distinct pauses between words

Best way to explain the system:

- Stage 1: computer vision extracts hand landmarks
- Stage 2: BiLSTM classifies sign sequences
- Stage 3: rule-based NLP refines the output into natural Vietnamese text

## 13. Current Project Status

### Working

- offline single-word prediction
- offline sequence prediction
- live webcam sign prediction
- live webcam word-by-word sequence building
- Vietnamese rule-based refined sentence output

### Not fully solved

- robust continuous real-time sentence segmentation
- full natural language understanding
- large-vocabulary sign recognition
- model robustness to very slow signing

