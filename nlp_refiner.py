import os
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OLLAMA_DIR = BASE_DIR / "ollama_local"
OLLAMA_APP_DIR = OLLAMA_DIR / "app"
OLLAMA_MODELS_DIR = OLLAMA_DIR / "models"
OLLAMA_BINARY = OLLAMA_APP_DIR / "bin" / "ollama"
DEFAULT_OLLAMA_MODEL = "gbenson/qwen2.5-0.5b-instruct:Q2_K"


def available_backends():
    backends = ["rules"]
    if OLLAMA_BINARY.exists():
        backends.append("ollama")
    return backends


def normalize_gloss_sequence(text):
    words = [w.strip() for w in text.split() if w.strip()]
    cleaned = []
    for word in words:
        if not cleaned or cleaned[-1] != word:
            cleaned.append(word)
    return cleaned


def _reorder_time_marker(words):
    if "HOM_TRUOC" in words:
        return ["HOM_TRUOC"] + [w for w in words if w != "HOM_TRUOC"]
    return words


def _apply_phrase_rules(words):
    words = words[:]

    if words == ["CAM_DIEC"]:
        return "Cảm ơn."

    if words == ["KHONG"]:
        return "Không."

    if words == ["TOI", "THICH", "HOC"]:
        return "Tôi thích học."

    if words == ["TOI", "KHONG", "THICH", "HOC"]:
        return "Tôi không thích học."

    if words == ["BAN", "THICH", "GI"]:
        return "Bạn thích gì?"

    if words == ["TOI", "THICH", "GI"]:
        return "Tôi thích gì?"

    if words == ["BAN", "DI", "HOC"]:
        return "Bạn đi học."

    if words == ["TOI", "DI", "HOC"]:
        return "Tôi đi học."

    if words == ["HOM_TRUOC", "TOI", "DI", "HOC"]:
        return "Hôm trước tôi đi học."

    if words == ["HOM_TRUOC", "BAN", "DI", "HOC"]:
        return "Hôm trước bạn đi học."

    if len(words) >= 2 and words[-1] == "GI":
        base = [w for w in words[:-1] if w != "GI"]
        phrase = " ".join(_word_to_text(word) for word in base).strip()
        if phrase:
            return f"{phrase} gì?"
        return "Gì?"

    return ""


def _word_to_text(word):
    mapping = {
        "BAN": "bạn",
        "CAM_DIEC": "cảm ơn",
        "DI": "đi",
        "GI": "gì",
        "HOC": "học",
        "HOM_TRUOC": "hôm trước",
        "KHONG": "không",
        "THICH": "thích",
        "TOI": "tôi",
    }
    return mapping.get(word, word.lower())


def _cleanup_tokens(words):
    words = normalize_gloss_sequence(" ".join(words))
    words = _reorder_time_marker(words)
    return words


def rule_based_refine(text):
    words = normalize_gloss_sequence(text)
    if not words:
        return ""

    words = _cleanup_tokens(words)
    phrase_rule = _apply_phrase_rules(words)
    if phrase_rule:
        return phrase_rule

    sentence = " ".join(_word_to_text(word) for word in words).strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "?")):
        sentence += "."
    return sentence


def _ollama_env():
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)
    return env


def ollama_available():
    return OLLAMA_BINARY.exists()


def refine_with_ollama(text, model=DEFAULT_OLLAMA_MODEL, timeout=30):
    if not text.strip():
        return ""
    if not ollama_available():
        raise RuntimeError("Ollama local binary not found.")

    prompt = (
        "You are refining Vietnamese sign gloss output into a short natural Vietnamese sentence. "
        "Keep the meaning, remove duplicate words, improve word order lightly, and return only the final sentence.\n\n"
        f"Gloss: {text}\nSentence:"
    )

    result = subprocess.run(
        [str(OLLAMA_BINARY), "run", model, prompt],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_ollama_env(),
    )
    return result.stdout.strip()


def refine_text(text, backend="rules", model=DEFAULT_OLLAMA_MODEL):
    if backend == "rules":
        return rule_based_refine(text)
    if backend == "ollama":
        return refine_with_ollama(text, model=model)
    raise ValueError(f"Unsupported NLP backend: {backend}")
