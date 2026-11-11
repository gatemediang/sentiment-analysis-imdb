"""
Movie Review Sentiment Analyzer (Streamlit)

Classifies a movie review as positive or negative using a DistilBERT model
fine-tuned on IMDB reviews. The model is loaded
from the Hugging Face Hub at startup; no model files live in this repository.

Guard rails in this app:
    - input sanitisation (Unicode normalisation, control characters, HTML tags,
      URLs, character floods) before anything reaches the model
    - validation with clear user-facing messages for empty, non-text, emoji-only,
      link-only, repetitive and over-long input
    - warnings (not rejections) for short, non-English, truncated or partly
      stripped input
    - token-level truncation so long reviews can never crash the model
    - low-confidence predictions reported as uncertain instead of forced
    - per-client and global sliding-window rate limits, shared across sessions
    - thread-safe inference (Streamlit serves each session on its own thread)
    - structured logging to stdout with no raw user text, only length and a hash
    - model load failures shown with a retry option instead of a crash

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import hashlib
import logging
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field

import streamlit as st
from transformers import pipeline

# =============================================================================
# Page config (first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# =============================================================================
# Configuration
# =============================================================================
def get_setting(name: str, default=None):
    """Read a setting from Streamlit secrets, then environment, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml present
        pass
    return os.getenv(name, default)


MODEL_ID = get_setting("MODEL_ID", "gatemediang/distilbert-slang-sentiment")
# Pin to a commit id in secrets for fully reproducible deployments
MODEL_REVISION = get_setting("MODEL_REVISION", "main")
HF_TOKEN = get_setting("HF_TOKEN")  # only needed if the model repo is private

MAX_CHARS = 5000          # hard cap on input length (characters)
MAX_TOKENS = 512          # DistilBERT's maximum sequence length
MIN_LETTERS = 2           # fewer letters than this is not a review
SHORT_REVIEW_WORDS = 3    # below this, warn that predictions are less reliable
MAX_EMOJI_RATIO = 0.5     # reject if more than half the visible characters are emoji
LOW_CONFIDENCE = 0.65     # below this, report the prediction as uncertain

RATE_LIMIT_PER_CLIENT = 10    # predictions per client per window
RATE_LIMIT_GLOBAL = 120       # predictions across all users per window
RATE_LIMIT_WINDOW = 60        # seconds

LABEL_NAMES = {
    "NEGATIVE": "Negative",
    "POSITIVE": "Positive",
    "LABEL_0": "Negative",
    "LABEL_1": "Positive",
}


# =============================================================================
# Logging (configured once per process, not on every rerun)
# =============================================================================
@st.cache_resource
def setup_logging():
    """
    Create the app and security loggers.

    Logs go to stdout, which Streamlit Community Cloud captures under
    "Manage app". Its filesystem is temporary, so log files would be lost on
    restart; set LOG_TO_FILE=1 to also write rotating files when running locally.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_logger = logging.getLogger("sentiment_app")
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False

    if not app_logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        app_logger.addHandler(stream)

        if os.getenv("LOG_TO_FILE") == "1":
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                "sentiment_app.log", maxBytes=1_000_000, backupCount=3
            )
            file_handler.setFormatter(formatter)
            app_logger.addHandler(file_handler)

    # Child logger: inherits the handlers above, tagged "sentiment_app.security"
    security_logger = logging.getLogger("sentiment_app.security")
    security_logger.setLevel(logging.INFO)
    return app_logger, security_logger


logger, security_logger = setup_logging()


def fingerprint(text: str) -> str:
    """Short, irreversible hash so events can be correlated without storing text."""
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:12]


def log_security_event(event_type: str, detail: str, level: str = "info"):
    """
    Log a security-relevant event.

    Args:
        event_type: short category label, e.g. 'RATE_LIMIT', 'INPUT_REJECTED'
        detail:     description; must never contain raw user text
        level:      'info' | 'warning' | 'critical'
    """
    getattr(security_logger, level if level in ("warning", "critical") else "info")(
        f"[{event_type}] {detail}"
    )


# =============================================================================
# Rate limiting (shared across all sessions in this server process)
# =============================================================================
class SlidingWindowRateLimiter:
    """
    Thread-safe sliding-window limiter.

    Kept in st.cache_resource so every session shares it. A limit stored in
    st.session_state alone resets whenever the user refreshes the page.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._hits = defaultdict(deque)

    def _trim(self, key, window, now):
        q = self._hits[key]
        while q and now - q[0] >= window:
            q.popleft()
        return q

    def try_acquire(self, rules):
        """
        rules: list of (key, limit, window_seconds).
        Records a hit against every rule only if all of them allow it.
        Returns (allowed, retry_after_seconds, blocking_key).
        """
        now = time.monotonic()
        with self._lock:
            for key, limit, window in rules:
                q = self._trim(key, window, now)
                if len(q) >= limit:
                    retry_after = int(window - (now - q[0])) + 1
                    return False, retry_after, key
            for key, _, _ in rules:
                self._hits[key].append(now)
            # Drop empty queues so memory doesn't grow with one-off visitors
            for key in [k for k, q in self._hits.items() if not q]:
                del self._hits[key]
        return True, 0, None

    def remaining(self, key, limit, window):
        with self._lock:
            q = self._trim(key, window, time.monotonic())
            return max(limit - len(q), 0)


@st.cache_resource
def get_rate_limiter():
    return SlidingWindowRateLimiter()


def get_client_key() -> str:
    """
    Identify the client for rate limiting.

    Uses the first X-Forwarded-For address (hashed, never stored raw) when the
    app sits behind a proxy, as on Streamlit Community Cloud; otherwise falls
    back to a per-session id.
    """
    ip = ""
    try:
        forwarded = st.context.headers.get("X-Forwarded-For", "")
        ip = forwarded.split(",")[0].strip()
    except Exception:
        pass
    if ip:
        return "ip:" + fingerprint(ip)
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    return "session:" + st.session_state.session_id


# =============================================================================
# Model loading and inference
# =============================================================================
@st.cache_resource(show_spinner="Loading the model from the Hugging Face Hub...")
def load_model():
    """
    Download and initialise the classifier once per server process.

    Exceptions are raised, not displayed here: st.cache_resource does not cache
    failures, so the next rerun retries the download automatically.
    """
    start = time.perf_counter()
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        token=HF_TOKEN,
    )
    logger.info(
        f"Model loaded: {MODEL_ID}@{MODEL_REVISION} in {time.perf_counter() - start:.1f}s"
    )
    return classifier


@st.cache_resource
def get_inference_lock():
    """Fast tokenizers are not safe to call from several threads at once."""
    return threading.Lock()


def predict(classifier, text: str) -> dict:
    """Run the model and return the label, its confidence and both class scores."""
    with get_inference_lock():
        raw = classifier(text, top_k=None, truncation=True, max_length=MAX_TOKENS)
    if raw and isinstance(raw[0], list):  # older transformers nest one level deeper
        raw = raw[0]
    scores = {LABEL_NAMES.get(r["label"].upper(), r["label"]): float(r["score"]) for r in raw}
    label = max(scores, key=scores.get)
    return {"label": label, "confidence": scores[label], "scores": scores}


# =============================================================================
# Input sanitisation and validation
# =============================================================================
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
TAG_RE = re.compile(r"<[^<>]{0,200}>")
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]"
)
LETTER_FLOOD_RE = re.compile(r"([^\W\d_])\1{3,}")   # "soooooo" -> "sooo"
SYMBOL_FLOOD_RE = re.compile(r"([^\w\s])\1{2,}")    # "!!!!!!" -> "!!"
WHITESPACE_RE = re.compile(r"[ \t]+")
NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass
class ValidationResult:
    ok: bool
    text: str = ""
    error: str = ""
    reason: str = ""                      # short code for logs
    warnings: list = field(default_factory=list)


def sanitize_input(text: str) -> tuple[str, dict]:
    """
    Clean the input and report what was removed.

    Order matters: normalise Unicode first so look-alike characters can't
    slip past the later checks.
    """
    flags = {"html": False, "urls": False}

    text = unicodedata.normalize("NFKC", text)
    # Drop control and invisible characters, keep newlines and tabs
    text = "".join(
        ch for ch in text
        if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )
    if TAG_RE.search(text):
        flags["html"] = True
        text = TAG_RE.sub(" ", text)
    if URL_RE.search(text):
        flags["urls"] = True
        text = URL_RE.sub(" ", text)

    text = LETTER_FLOOD_RE.sub(r"\1\1\1", text)
    text = SYMBOL_FLOOD_RE.sub(r"\1\1", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = NEWLINES_RE.sub("\n\n", text)
    return text.strip(), flags


def validate_input(raw: str, tokenizer) -> ValidationResult:
    """Sanitise, then decide whether the text can be classified."""
    if raw is None or not raw.strip():
        return ValidationResult(False, error="Please enter a review first.", reason="EMPTY")

    if len(raw) > MAX_CHARS:
        return ValidationResult(
            False,
            error=f"Reviews are limited to {MAX_CHARS:,} characters. Please shorten yours.",
            reason="TOO_LONG",
        )

    text, flags = sanitize_input(raw)
    warnings = []

    if flags["html"]:
        log_security_event(
            "HTML_STRIPPED",
            f"Markup removed from input. len={len(raw)} fp={fingerprint(raw)}",
            level="warning",
        )

    visible = [ch for ch in text if not ch.isspace()]
    letters = [ch for ch in visible if ch.isalpha()]
    emoji_count = len(EMOJI_RE.findall(text))

    if not text or (not letters and emoji_count == 0):
        if flags["urls"]:
            return ValidationResult(
                False,
                error="Links can't be analysed. Paste the review text itself.",
                reason="URL_ONLY",
            )
        return ValidationResult(
            False,
            error="That doesn't contain any words. Describe what you thought of the film.",
            reason="NO_TEXT",
        )

    if visible and emoji_count / len(visible) > MAX_EMOJI_RATIO:
        return ValidationResult(
            False,
            error="The model reads words, not emoji. Add a sentence about the film.",
            reason="EMOJI_ONLY",
        )

    if len(letters) < MIN_LETTERS:
        return ValidationResult(
            False,
            error="That's too short to analyse. Write at least a few words.",
            reason="TOO_FEW_LETTERS",
        )

    words = re.findall(r"[^\W\d_]+", text)
    if len(words) >= 20 and len(set(w.lower() for w in words)) / len(words) < 0.2:
        return ValidationResult(
            False,
            error="The text is mostly the same words repeated. Please enter a genuine review.",
            reason="REPETITIVE",
        )

    # Accepted, with warnings where the prediction may be less reliable
    if flags["urls"]:
        warnings.append("Links were removed before analysis.")
    if emoji_count:
        warnings.append("Emoji are ignored by the model; only the words are analysed.")
    if len(words) < SHORT_REVIEW_WORDS:
        warnings.append("Very short reviews give less reliable predictions.")

    latin = sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, ""))
    if latin / len(letters) < 0.5:
        warnings.append("The model was trained on English reviews, so results for other languages are unreliable.")

    n_tokens = len(tokenizer(text, add_special_tokens=True)["input_ids"])
    if n_tokens > MAX_TOKENS:
        warnings.append(
            f"Your review is {n_tokens} tokens long. The model reads only the first {MAX_TOKENS}, "
            "so the end of the review is ignored."
        )

    return ValidationResult(True, text=text, warnings=warnings)


# =============================================================================
# UI helpers
# =============================================================================
EXAMPLES = {
    "Slang": [
        "No cap this film was amazing",
        "The plot was lowkey boring",
        "This movie ate and left no crumbs",
    ],
    "Sarcasm and mixed": [
        "Oh great another superhero movie",
        "Rough start but it got so good",
        "Visually stunning but emotionally hollow",
    ],
    "Edge cases": [
        "😂😂😂🔥🔥",
        "https://www.example.com/review",
        "Peak",
    ],
}


def use_example(text: str):
    """Sidebar button callback: put an example into the review box."""
    st.session_state["review"] = text


def show_result(result: dict):
    label, confidence = result["label"], result["confidence"]

    if confidence < LOW_CONFIDENCE:
        st.info(
            f"Uncertain, leaning {label.lower()} ({confidence:.0%} confidence). "
            "The review may be mixed or neutral, and this model only knows positive and negative."
        )
    elif label == "Positive":
        st.success(f"Positive 😊  ({confidence:.0%} confidence)")
    else:
        st.error(f"Negative 😞  ({confidence:.0%} confidence)")

    col1, col2 = st.columns(2)
    for col, name in zip((col1, col2), ("Positive", "Negative")):
        score = result["scores"].get(name, 0.0)
        col.metric(name, f"{score:.1%}")
        col.progress(score)


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Try an example")
    for group, items in EXAMPLES.items():
        st.caption(group)
        for i, example in enumerate(items):
            st.button(example, key=f"ex_{group}_{i}", on_click=use_example, args=(example,))

    with st.expander("About the model"):
        st.markdown(
            f"""
DistilBERT ([`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})), fine-tuned from
`distilbert-base-uncased-finetuned-sst-2-english` on 341 slang-heavy reviews
mixed with 800 IMDB reviews.

| Accuracy | Slang test (69) | IMDB sample (500) |
|---|---|---|
| Base model | 84.1% | 89.0% |
| Fine-tuned | 84.1% | 89.2% |

It is a binary classifier with no neutral class, trained on English text.
"""
        )


# =============================================================================
# Main page
# =============================================================================
st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Type a movie review, slang and sarcasm welcome, and the model will predict "
    "whether it's positive or negative."
)

try:
    classifier = load_model()
except Exception as exc:
    logger.exception(f"Model load failed for {MODEL_ID}@{MODEL_REVISION}")
    message = str(exc).lower()
    if any(s in message for s in ("not found", "404", "401", "403", "gated", "private", "revision")):
        st.error(
            f"The model `{MODEL_ID}` (revision `{MODEL_REVISION}`) couldn't be accessed. "
            "Check that the repo id and revision are correct and that the repo is public, "
            "or add an HF_TOKEN secret."
        )
    else:
        st.error(
            "The model couldn't be loaded right now. This is usually a temporary "
            "network problem reaching the Hugging Face Hub."
        )
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    if st.button("Try again"):
        st.rerun()
    st.stop()

limiter = get_rate_limiter()
client_key = get_client_key()

with st.form("review_form", clear_on_submit=False):
    st.text_area(
        "Your review",
        key="review",
        height=160,
        max_chars=MAX_CHARS,
        placeholder="e.g. The cinematography was bussin but the plot was mid",
    )
    submitted = st.form_submit_button("Analyse sentiment", type="primary")

remaining = limiter.remaining(client_key, RATE_LIMIT_PER_CLIENT, RATE_LIMIT_WINDOW)
st.caption(f"Analyses remaining this minute: {remaining}/{RATE_LIMIT_PER_CLIENT}")

if submitted:
    raw_text = st.session_state.get("review", "")
    check = validate_input(raw_text, classifier.tokenizer)

    if not check.ok:
        log_security_event(
            "INPUT_REJECTED",
            f"reason={check.reason} len={len(raw_text)} fp={fingerprint(raw_text)} client={client_key}",
            level="info",
        )
        st.warning(check.error)
        st.stop()

    allowed, retry_after, blocked_by = limiter.try_acquire([
        (client_key, RATE_LIMIT_PER_CLIENT, RATE_LIMIT_WINDOW),
        ("__global__", RATE_LIMIT_GLOBAL, RATE_LIMIT_WINDOW),
    ])
    if not allowed:
        log_security_event(
            "RATE_LIMIT",
            f"blocked_by={'global' if blocked_by == '__global__' else 'client'} "
            f"client={client_key} retry_after={retry_after}s",
            level="warning",
        )
        if blocked_by == "__global__":
            st.error(f"The app is busy right now. Please try again in {retry_after} seconds.")
        else:
            st.error(
                f"You've reached {RATE_LIMIT_PER_CLIENT} analyses per minute. "
                f"Please wait {retry_after} seconds."
            )
        st.stop()

    for warning in check.warnings:
        st.warning(warning)

    try:
        start = time.perf_counter()
        with st.spinner("Analysing..."):
            result = predict(classifier, check.text)
        latency_ms = (time.perf_counter() - start) * 1000
    except Exception:
        logger.exception(f"Inference failed. len={len(check.text)} fp={fingerprint(check.text)}")
        st.error("Something went wrong while analysing that review. Please try again.")
        st.stop()

    logger.info(
        f"Prediction label={result['label']} conf={result['confidence']:.3f} "
        f"len={len(check.text)} warnings={len(check.warnings)} latency_ms={latency_ms:.0f}"
    )
    st.divider()
    show_result(result)

st.divider()
st.caption(
    f"Model: [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) · revision `{MODEL_REVISION}` · "
    "Built with Streamlit and Hugging Face Transformers"
)
