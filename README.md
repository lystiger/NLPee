# Vietnamese Sign Language Recognition via MediaPipe Landmark Extraction, Bidirectional LSTM Sequence Classification, and Rule-Based NLP Refinement

This project presents a Vietnamese Sign Language recognition system developed for an NLP course project. Although the pipeline begins with landmark extraction from sign videos, the main contribution is framed from an NLP perspective: sign language is treated as a temporal sequence understanding problem, and the final objective is not only to classify gestures but also to generate readable Vietnamese text from gloss-like outputs.

The system is designed as a lightweight proof-of-concept for short conversational expressions in Vietnamese. It combines sequence modeling with a small post-processing language module so that raw classifier predictions can be converted into more natural text. In this sense, the project sits between isolated sign recognition and low-resource sign language translation.

## Project Idea

The central motivation of this work is that sign recognition should not be viewed only as a visual classification problem. A sign unfolds over time, just like a spoken word or a short phrase unfolds as a sequence. Because of that, Vietnamese Sign Language recognition can be interpreted in NLP terms:

- each frame is analogous to a token in a sequence
- each landmark vector is analogous to a token embedding
- a short sign clip forms a sequence input for a temporal encoder
- predicted labels function like glosses
- a final refinement stage maps glosses into natural Vietnamese text

This NLP framing is especially useful in a low-resource setting, where a full gloss-to-sentence neural translation model is not yet realistic due to the lack of large parallel datasets.

## System Overview

The pipeline consists of three main stages:

1. MediaPipe extracts hand landmarks from video frames.
2. A Bidirectional LSTM classifies the resulting landmark sequence into sign labels.
3. An NLP refinement module rewrites the raw output into a short Vietnamese sentence.

The current system works on a small vocabulary of 9 sign classes and focuses on short utterances. It is therefore best understood as a controlled proof-of-concept rather than a complete sign language translation system.

## Why This Is An NLP Project

From an NLP viewpoint, the key task is sequence modeling. The classifier does not make a decision from a single frame alone; instead, it processes an ordered sequence and learns temporal dependencies between frames. This makes the task conceptually close to sequence classification problems such as sentence classification, sequence labeling, or phoneme-to-word modeling.

The second NLP aspect appears after recognition. The raw output labels are not yet natural Vietnamese. They need to be normalized, reordered when necessary, restored with diacritics, and punctuated correctly. This post-processing step is a small gloss-to-text refinement problem. In the current project, that stage is handled by:

- a deterministic rule-based backend
- an optional local Qwen backend through Ollama

This allows the project to compare interpretable symbolic refinement against a lightweight generative approach.

## Sequence Representation

Each input video is converted into a fixed-length sequence of 35 frames. For every frame, the system extracts a 126-dimensional feature vector corresponding to two-hand landmarks. The final model input has shape:

```text
1 x 35 x 126
```

In NLP language, this is similar to passing a short sentence of length 35 into a sequence encoder, where each timestep has its own learned representation. Padding shorter videos to a fixed length is also conceptually similar to using padding tokens in NLP pipelines.

## BiLSTM Sequence Classifier

The main classifier is a Bidirectional LSTM. This choice is motivated by the small dataset size and the short sequence length. Compared with larger transformer-style models, BiLSTM is more suitable here because:

- the dataset is relatively limited
- temporal order is essential
- sequence length is short enough that recurrent modeling is efficient
- the model remains lightweight and practical for CPU deployment

The classifier uses:

- hidden size `128`
- `2` LSTM layers
- bidirectional encoding
- dropout `0.3`
- a final linear layer over `9` output classes

In the project report, BiLSTM is presented as a strong and appropriate sequence encoder for this low-resource VSL setting.

## NLP Refinement Layer

After classification, the model outputs gloss-like labels rather than fluent Vietnamese text. For example:

```text
TOI TOI DI HOC
```

The rule-based NLP layer converts this into:

```text
Tôi đi học.
```

This refinement stage is important because raw sign labels are often repetitive, incomplete, or unnatural when displayed directly. The rule-based module handles several common operations:

- duplicate suppression
- reordering of time markers such as `HOM_TRUOC`
- Vietnamese diacritic restoration
- punctuation insertion
- simple template-based phrase rewriting

For example:

```text
BAN THICH GI -> Bạn thích gì?
TOI DI HOC -> Tôi đi học.
TOI TOI DI HOC -> Tôi đi học.
HOM_TRUOC TOI DI HOC -> Hôm trước tôi đi học.
```

An optional local Qwen backend is also supported. Its role is not to replace the classifier, but to refine the predicted gloss sequence into smoother Vietnamese text. This gives the project a useful comparison between rule-based and generative NLP refinement.

## Vocabulary

The current system predicts 9 labels:

- `BAN`
- `CAM_DIEC`
- `DI`
- `GI`
- `HOC`
- `HOM_TRUOC`
- `KHONG`
- `THICH`
- `TOI`

This vocabulary is intentionally small. The goal of the project is to validate the pipeline design and the NLP refinement idea, not to claim full-scale sign language translation.

## Training and Model Behavior

According to the latest project report, the model was trained on a dataset split into:

- `80%` training
- `10%` validation
- `10%` test

The held-out test set contains `241` samples. Training uses:

- AdamW optimizer
- learning rate `1e-3`
- weight decay `1e-2`
- label smoothing
- early stopping
- regularization and augmentation designed for sequence robustness

Training stopped at epoch `88` through early stopping.

The final reported classification performance is:

- test accuracy: `95.02%`
- balanced accuracy: `95.57%`
- macro precision: `0.95`
- macro recall: `0.96`
- macro F1-score: `0.95`

These results show that even with a relatively compact architecture, the sequence classification stage can perform strongly on a small fixed vocabulary.

## NLP Evaluation

Beyond recognition accuracy, the report evaluates the system from an NLP perspective by measuring how well the refinement layer transforms gloss sequences into natural Vietnamese sentences. This is an extrinsic evaluation setup: the refinement backend is judged based on downstream sentence quality rather than only classifier accuracy.

The evaluation compares two backends:

- rule-based refinement
- Qwen 2.5B local refinement

The main metrics are:

- `BLEUbefore`: similarity between raw gloss prediction and the reference
- `BLEUafter`: similarity between refined text and the reference
- `ROUGE-L`: overlap in sequence structure and content

Average results reported in the project are:

- Rule-based backend: `BLEUbefore 80.6%`, `BLEUafter 60.9%`, `ROUGE-L 81.2%`
- Qwen 2.5B backend: `BLEUbefore 64.5%`, `BLEUafter 42.3%`, `ROUGE-L 75.0%`

The current findings suggest that the rule-based backend performs better overall on the present dataset. This is mainly because the dataset contains many short and relatively regular outputs, where deterministic templates are reliable. Qwen can generate more natural sentences in some longer or noisier cases, but it is also more likely to paraphrase incorrectly or over-interpret an already wrong gloss input.

## Notes on BLEU and Short Sequences

One important observation from the report is that BLEU-4 is not always a good metric for this project. Many predicted Vietnamese outputs are only 2 to 3 words long. In such cases, BLEU-4 can become artificially low or even misleading, not because the sentence is semantically wrong, but because the metric is poorly matched to very short outputs.

For that reason, ROUGE-L is often a more informative metric in this project. It captures ordered token overlap and is more stable for short sign-derived sentences. The report therefore treats BLEU as a useful indicator, but not the only measure of quality.

## Error Analysis

The report highlights several important failure patterns.

The first major confusion is:

```text
TOI -> BAN
```

This is especially serious because it changes the subject of a sentence. For example:

```text
Tôi đi học. -> Bạn đi học.
```

This kind of error is hard for the NLP layer to fix, because once the classifier outputs the wrong gloss, the refinement stage no longer has access to the original visual evidence.

Another important confusion is:

```text
DI -> CAM_DIEC
```

This affects both recognition quality and downstream sentence quality.

The report also notes a semantic ambiguity around the label `CAM_DIEC`. In some places it is interpreted closer to "cảm ơn", while in others it aligns with "câm điếc". This ambiguity introduces systematic evaluation noise and lowers downstream NLP metrics. In other words, part of the error is not only model error, but also label-definition inconsistency.

## Running the Project

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the main application:

```bash
python main.py
```

## Main Workflows

The GUI currently supports the following workflows:

### 1. Single-sign prediction

- choose an input video
- click `Predict sign`
- the system outputs one predicted label

### 2. Multi-sign sequence prediction

- choose an input video
- click `Predict sequence`
- the system outputs a gloss sequence
- the NLP layer refines it into Vietnamese text

### 3. NLP evaluation

- enter a reference sentence
- run prediction
- compute BLEU and ROUGE-L
- compare raw gloss output and refined sentence output

### 4. Optional live mode

The application also supports webcam-based prediction. However, the current interaction is still turn-based rather than continuous: the user signs one word, pauses, then signs the next word. This is useful for demos, but it is not yet equivalent to full real-time sentence translation.

## Optional Ollama / Qwen Backend

The default refinement backend is `rules`. If Ollama is installed locally, the project can also use a Qwen model for local text refinement.

Example:

```bash
ollama pull gbenson/qwen2.5-0.5b-instruct:Q2_K
```

Then run the GUI and select the `ollama` backend. This backend is useful for qualitative comparison, especially when discussing rule-based NLP versus LLM-based rewriting.

## Main Files

Some key files in the repository are:

- `main.py`: GUI, prediction flow, and evaluation interface
- `predict_sequence.py`: sequence prediction logic
- `predict_word.py`: single-sign prediction
- `nlp_refiner.py`: rule-based and Ollama-based refinement
- `model_bundle.py`: model loading and metadata
- `train_colab_updated.ipynb`: training notebook

## Current Limitations

This project is still a proof-of-concept and has several important limitations:

- only 9 sign classes are supported
- the system is closer to isolated sign recognition plus NLP refinement than full sign language translation
- downstream text quality depends heavily on upstream recognition accuracy
- the NLP module improves readability, but cannot reliably repair a wrong gloss prediction
- BLEU-4 is not ideal for short outputs
- `CAM_DIEC` remains semantically ambiguous in the current project materials
- the live webcam mode is not yet robust continuous sentence recognition

## Conclusion

This project demonstrates that Vietnamese Sign Language recognition can be meaningfully framed as an NLP problem. The most important contribution is not only the sequence classifier itself, but the combination of sequence modeling and gloss-to-text refinement. In a low-resource setting, the rule-based NLP backend currently provides the most stable overall results: it is simple, interpretable, and effective for short Vietnamese outputs. At the same time, the comparison with Qwen shows a clear path for future work on more flexible gloss-to-text generation once larger and cleaner VSL datasets become available.
