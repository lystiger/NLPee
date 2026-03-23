# Vietnamese Sign Language Recognition

Vietnamese Sign Language recognition system built for an NLP course project at the University of Science and Technology Hanoi. The project combines computer vision, sequence modeling, and lightweight NLP to translate sign videos into short Vietnamese sentences.

The current repository already includes:

- a trained MediaPipe + BiLSTM inference pipeline
- a Tkinter desktop GUI for demo and evaluation
- offline prediction for single-sign and multi-sign videos
- live webcam prediction with word-by-word sequence building
- a rule-based Vietnamese refinement layer
- an optional Ollama-based local LLM refinement backend
- exported model artifacts in [`output/_output`](output/_output)
- the course report in [`NLP_course.pdf`](NLP_course.pdf)

## Project Summary

The system runs in three stages:

1. MediaPipe Hands extracts 2-hand landmarks from each frame.
2. A BiLSTM classifies a fixed-length sequence of landmark features into sign labels.
3. An NLP post-processing layer converts raw gloss-like outputs into more natural Vietnamese text.

This repo is an in-progress course project, but the inference demo, evaluation tools, and exported model bundle are already usable.

## Main Features

- 2-hand landmark extraction with MediaPipe, without YOLO in the current pipeline
- 126 features per frame: 63 for the left hand and 63 for the right hand
- fixed input shape of `1 x 35 x 126`
- 9-class Vietnamese Sign Language classifier
- offline single-word prediction from video
- offline multi-word sequence prediction from video
- live webcam mode with pause-based word confirmation
- BLEU and ROUGE-L scoring inside the GUI
- CSV logging of evaluation results to [`results_log.csv`](results_log.csv)
- per-case and aggregate metric plotting

## Vocabulary

The exported model currently predicts these 9 labels:

| Label | Intended meaning |
| --- | --- |
| `BAN` | you |
| `CAM_DIEC` | inconsistent in current materials; see note below |
| `DI` | go |
| `GI` | what |
| `HOC` | study |
| `HOM_TRUOC` | yesterday |
| `KHONG` | no / not |
| `THICH` | like |
| `TOI` | I / me |

Note: the `CAM_DIEC` label is described inconsistently across the current code, logs, and report. The rule-based NLP layer currently maps it to `cảm ơn`, while parts of the report and example references also suggest `câm điếc`. This should be standardized before the final project release.

## Reported Results

According to [`NLP_course.pdf`](NLP_course.pdf), the main BiLSTM classifier achieved:

- `95.02%` test accuracy
- `95.57%` balanced accuracy
- macro Precision `0.95`
- macro Recall `0.96`
- macro F1-score `0.95`
- evaluation on `241` held-out test samples
- early stopping at epoch `88`

The paper also reports the main error patterns:

- `TOI` is sometimes confused with `BAN`
- `DI` is sometimes confused with `CAM_DIEC`

Generated training artifacts included in the repo:

- [`training_history.png`](output/_output/training_history.png)
- [`confusion_matrix.png`](output/_output/confusion_matrix.png)

## Architecture

### 1. Feature Extraction

- MediaPipe Hands detects up to 2 hands directly on the full frame.
- Each hand provides 21 landmarks with `(x, y, z)` coordinates.
- Missing hands are padded with zeros.
- Videos are sampled or padded to exactly `35` frames.

### 2. Sequence Classifier

The exported classifier in [`model_bundle.py`](model_bundle.py) is a BiLSTM with:

- hidden size `128`
- `2` LSTM layers
- bidirectional encoding
- dropout `0.3`
- layer normalization before final classification

The metadata bundle is stored in:

- [`vsl_bilstm_state_dict.pth`](output/_output/vsl_bilstm_state_dict.pth)
- [`vsl_bilstm_metadata.json`](output/_output/vsl_bilstm_metadata.json)
- [`model.py`](output/_output/model.py)

### 3. NLP Refinement

The NLP layer in [`nlp_refiner.py`](nlp_refiner.py) currently supports:

- duplicate removal
- simple word reordering, especially `HOM_TRUOC`
- Vietnamese diacritic restoration
- punctuation insertion
- fixed phrase templates for common outputs
- optional Ollama-based local LLM rewriting

## Repository Structure

| Path | Purpose |
| --- | --- |
| [`main.py`](main.py) | Tkinter GUI, webcam flow, evaluation tools |
| [`extractor.py`](extractor.py) | MediaPipe landmark extraction and feature caching |
| [`predict_word.py`](predict_word.py) | single-word prediction utilities |
| [`predict_sequence.py`](predict_sequence.py) | multi-word sequence prediction from video |
| [`model_bundle.py`](model_bundle.py) | model loading and metadata-driven configuration |
| [`nlp_refiner.py`](nlp_refiner.py) | rule-based and Ollama-based text refinement |
| [`plot_overall.py`](plot_overall.py) | aggregate plotting from `results_log.csv` |
| [`train_colab_updated.ipynb`](train_colab_updated.ipynb) | Colab training notebook |
| [`Code_emBao.py`](Code_emBao.py) | older YOLO + MediaPipe experimental code, not the main current pipeline |
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | concise project summary |
| [`NLP_course.pdf`](NLP_course.pdf) | course report / write-up |

## Requirements

- Python `3.11+` recommended
- Tkinter available in your Python installation
- webcam access for live mode
- optional: Ollama for local LLM refinement

Python dependencies are listed in [`requirements.txt`](requirements.txt):

- OpenCV
- NumPy
- MediaPipe
- PyTorch
- TorchVision
- Pillow
- imageio
- imageio-ffmpeg
- tqdm
- scikit-learn
- matplotlib
- seaborn
- pandas

## Installation

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your Python build does not include Tkinter, install it through your OS package manager before running the GUI.

If you see this error:

```bash
module 'mediapipe' has no attribute 'solutions'
```

reinstall the official package:

```bash
python -m pip uninstall -y mediapipe
python -m pip install mediapipe
```

## Quick Start

### Run the GUI

```bash
python main.py
```

The GUI supports:

- offline single-word prediction
- offline multi-word sequence prediction
- live webcam prediction
- reference-based BLEU / ROUGE-L evaluation
- PNG export of score graphs

### Model Path

By default, the app loads the exported model bundle from:

```text
output/_output/vsl_bilstm_state_dict.pth
```

You can run inference immediately as long as the dependencies are installed.

## How To Use the GUI

### A. Single-word prediction

1. Click `Input video file`.
2. Choose a video containing one sign.
3. Click `Predict sign`.

The predicted class label will appear in the `Predict sign` field.

### B. Multi-word sequence prediction

1. Click `Input video file`.
2. Choose a video containing several signs.
3. Optionally choose an `Output directory`.
4. Select an NLP backend.
5. Click `Predict sequence`.

The GUI will show:

- the raw predicted gloss sequence
- the refined Vietnamese sentence

If an output directory is selected, the sequence result is also written to `sequence.txt` in that directory.

### C. Live webcam mode

1. Click `Start webcam`.
2. Sign one word clearly.
3. Pause briefly so the system can confirm the word.
4. Continue with the next word.
5. Click `Stop webcam` when finished.

Important: the webcam mode is currently word-by-word and pause-based. It is not full continuous sentence recognition.

### D. Evaluation inside the GUI

1. Enter the expected sentence in `Reference text`.
2. Run sequence prediction or webcam prediction.
3. Click `Compute BLEU/ROUGE`.

The GUI will:

- compute BLEU before and after NLP refinement
- compute ROUGE-L
- append the result to [`results_log.csv`](results_log.csv)
- generate a comparison chart

Additional evaluation buttons:

- `Minh hoạ kết quả`: shows token-level overlap using an LCS-style visualization
- `Xuất graph (PNG)`: exports a score graph to a PNG file
- `Demo`: opens a short BLEU / ROUGE-L explanation panel

Note: the scoring code normalizes case, underscores, punctuation, and Vietnamese diacritics before comparison, so references without accents can still score correctly against accent-restored outputs.

## Optional Ollama Backend

The default NLP backend is `rules`. If Ollama is available locally, the GUI also exposes an `ollama` backend.

Default Ollama model used by the app:

```text
gbenson/qwen2.5-0.5b-instruct:Q2_K
```

Example setup:

```bash
ollama pull gbenson/qwen2.5-0.5b-instruct:Q2_K
```

Then in the GUI:

- choose `ollama` in `NLP backend`
- keep or edit the model name in the `Model` field

The Ollama backend can produce more natural text in some cases, but it is still limited by upstream recognition errors.

## Command-Line Usage

### Batch single-word prediction from a labeled dataset

```bash
python predict_word.py \
  --data_dir /path/to/NLP_DATASET \
  --model_path output/_output/vsl_bilstm_state_dict.pth \
  --output_path output/result.txt
```

Expected dataset layout:

```text
/path/to/NLP_DATASET/
  BAN/
  CAM_DIEC/
  DI/
  GI/
  HOC/
  HOM_TRUOC/
  KHONG/
  THICH/
  TOI/
```

### Sequence prediction from one video

```bash
python predict_sequence.py \
  --video_path /path/to/video.mp4 \
  --model_path output/_output/vsl_bilstm_state_dict.pth \
  --output_path output/sequence.txt \
  --nlp_backend rules
```

Using Ollama instead of rules:

```bash
python predict_sequence.py \
  --video_path /path/to/video.mp4 \
  --model_path output/_output/vsl_bilstm_state_dict.pth \
  --output_path output/sequence.txt \
  --nlp_backend ollama \
  --nlp_model gbenson/qwen2.5-0.5b-instruct:Q2_K
```

### Batch sequence prediction

```bash
python predict_sequence.py \
  --data_dir /path/to/sequence_dataset \
  --model_path output/_output/vsl_bilstm_state_dict.pth \
  --output_path output/sequence_batch.txt
```

## Feature Extraction and Caching

The current inference pipeline caches extracted features as `.npy` files. This is used both in the GUI and prediction scripts to avoid recomputing landmarks repeatedly.

- GUI feature extraction writes cache files under `<output_dir>/cache`
- single-video prediction uses `cache/predict/`
- GUI runtime prediction uses `cache/gui/`

This behavior is implemented in [`extractor.py`](extractor.py).

## Training

The training workflow lives in [`train_colab_updated.ipynb`](train_colab_updated.ipynb) and is designed for Google Colab.

The notebook expects a dataset structure like:

```text
MyDrive/NLP_DATASET/
  BAN/
  CAM_DIEC/
  DI/
  GI/
  HOC/
  HOM_TRUOC/
  KHONG/
  THICH/
  TOI/
  VALIDATION/
    BAN/
    CAM_DIEC/
    ...
  TEST/
    BAN/
    CAM_DIEC/
    ...
```

Training details described in the report include:

- AdamW optimizer
- learning rate `1e-3`
- batch size `32`
- label smoothing `0.1`
- weight decay `1e-2`
- gradient clipping
- early stopping
- augmentation with temporal shifting, keypoint masking, Gaussian noise, and frame dropout

Exported artifacts are saved under [`output/_output`](output/_output).

## Evaluation and Plotting

Per-sample evaluation logs are stored in [`results_log.csv`](results_log.csv).

To draw an aggregate comparison chart from that CSV:

```bash
python plot_overall.py
```

This script averages:

- `BLEU_Before`
- `BLEU_After`
- `ROUGE-L`

grouped by NLP backend.

## Current Limitations

- the classifier only supports a fixed 9-class vocabulary
- webcam mode is pause-based and not true continuous sign-to-sentence decoding
- segmentation for multi-word videos relies on silent-gap heuristics and can fail on slow or transition-heavy signing
- performance drops when hands are small, blurred, dark, or poorly separated from the background
- the NLP layer improves fluency but cannot reliably repair wrong sign predictions
- `CAM_DIEC` label semantics are not yet fully standardized across the project materials
- the repository includes inference assets, but the training dataset itself is not included

## Current Status

Working now:

- offline single-word prediction
- offline sequence prediction
- live webcam prediction
- word-by-word sequence building
- rule-based Vietnamese refinement
- optional Ollama refinement
- GUI-based BLEU / ROUGE-L evaluation

Not fully solved yet:

- robust continuous real-time sentence segmentation
- large-vocabulary recognition
- stronger robustness to slow signing and directionally similar gestures
- a finalized, fully consistent gloss-to-text mapping for all labels

## Documentation

For more detail, see:

- [`NLP_course.pdf`](NLP_course.pdf) for the course paper
- [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for a short engineering overview
