## Run

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## NLP refine (rules / Ollama Qwen)

App có 2 backend để “refine” câu: `rules` (mặc định) và `ollama` (LLM).

- Cài Ollama (máy bạn) để bật backend `ollama`.
- Pull model Qwen (ví dụ default trong app):

```bash
ollama pull gbenson/qwen2.5-0.5b-instruct:Q2_K
```

Sau đó mở app:

- Ở `NLP backend` chọn `ollama`
- Ở ô `Model` nhập tên model Ollama (ví dụ: `gbenson/qwen2.5-0.5b-instruct:Q2_K`)

If you see an error like `module 'mediapipe' has no attribute 'solutions'`, uninstall/reinstall the official package:

```bash
python -m pip uninstall -y mediapipe
python -m pip install mediapipe
```
