"""Feature extraction configuration.

Single source of truth for:
- R2 paths and credentials
- FEATURE_ORDER (82 physical columns, the locked v2.1 schema)
- Extractor constants (frame counts, sample rates, thresholds)
- Pipeline/pod settings

Do not import anything from src.data_collection_v2 here; feature extraction is
a separate phase that reads the outputs of data_collection_v2 from R2.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── R2 credentials ──────────────────────────────────────────────────────────
R2_BUCKET = "clipwhy-data"
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")

# ── R2 paths (prefixes, no leading or trailing slash) ───────────────────────
R2_SEGMENTS_PREFIX = "segments"
R2_TRANSCRIPTS_PREFIX = "transcripts/segments"
R2_LABELED_PREFIX = "labeled"
R2_METADATA_PREFIX = "metadata"
R2_FEATURES_PREFIX = "features"
R2_CLIP_EMBEDDINGS_PREFIX = "clip_embeddings"
R2_CLAIMS_PREFIX = "features_claims"
R2_PROGRESS_PREFIX = "features_progress"
R2_MODELS_PREFIX = "models"

# ── Local paths on the pod ──────────────────────────────────────────────────
SHARED_ROOT = Path(os.environ.get("FEATURES_SHARED_ROOT", "/workspace/features_shared"))
LOCAL_SEGMENTS = SHARED_ROOT / "segments"
LOCAL_TRANSCRIPTS = SHARED_ROOT / "transcripts"
LOCAL_LABELED = SHARED_ROOT / "labeled"
LOCAL_METADATA = SHARED_ROOT / "metadata"
LOCAL_FEATURES_OUT = SHARED_ROOT / "features_out"
LOCAL_CLIP_EMBEDDINGS_OUT = SHARED_ROOT / "clip_embeddings_out"
LOCAL_MODELS = SHARED_ROOT / "models"
LOCAL_LOGS = SHARED_ROOT / "logs"

# ── Claim mechanism ─────────────────────────────────────────────────────────
CLAIM_TTL_SECONDS = 7200          # 2 h: stale claims auto-reclaimable
CLAIM_RENEW_SECONDS = 900         # renew heartbeat every 15 min while working
CLAIM_POLL_SECONDS = 15           # how often to poll for next creator when pool empty

# ── Segment & sampling settings ─────────────────────────────────────────────
SEGMENT_DURATION_SEC = 30
AUDIO_SAMPLE_RATE = 16000
FRAMES_PER_SEGMENT = 5                       # lever A (VQualA 2025 rank-4 convention)
FRAME_SAMPLE_TIMES_SEC = [0, 6, 12, 18, 24]  # aligned to 5-frame schedule
CLIP_BATCH_SIZE = 8                          # lever B (batch segments for CLIP+DOVER)
CLIP_PCA_DIMS = 32
WAV2VEC_WINDOW_SEC = 5
WAV2VEC_HOP_SEC = 2.5
PAUSE_THRESHOLD_SEC = 0.15                   # Goldman-Eisler 1968 convention
HOOK_WINDOW_SEC = 3                          # V1 energy_first_3s
OPENING_HOOK_WINDOW_SEC = 5                  # first_5s_hook_word_ratio

# Pitch analysis (V1-compatible)
PYIN_FMIN = 50
PYIN_FMAX = 500
MIN_VOICED_FRACTION = 0.10

# Silence detection (V1-compatible)
SILENCE_ADAPTIVE_RATIO = 0.1
SILENCE_FALLBACK = 0.005

# Audio event thresholds
MUSIC_FRAME_THRESHOLD = 0.3
YAMNET_MUSIC_CLASSES = [132, 137]            # "Music", "Musical instrument"
YAMNET_SPEECH_CLASSES = [0, 3, 4]            # "Speech", "Narration", "Conversation"

# Face detection
FACE_CONF_THRESHOLD = 0.5

# Scene change
TRANSNET_INPUT_HEIGHT = 27
TRANSNET_INPUT_WIDTH = 48

# Voice quality gating
MIN_VOICED_FRAC_FOR_PRAAT = 0.20             # skip jitter/shimmer below this

# ── Pipeline threads per pod ────────────────────────────────────────────────
PREFETCH_WORKERS = 1     # 1 creator ahead
CPU_WORKER_THREADS = 6   # librosa + parselmouth release the GIL; pod has 8 vCPU
GPU_WORKER_THREADS = 1   # GPU models are not thread-safe
CPU_TO_GPU_QUEUE_MAXSIZE = 16
GPU_OUT_QUEUE_MAXSIZE = 32

# ── Features version & category encoding ────────────────────────────────────
FEATURES_VERSION = "v2.1-locked"
CATEGORY_ORDER = ["tech", "education", "entertainment", "fitness", "commentary"]

# ── Text lexicons (embedded here so pod is self-contained; identical to V1) ─
HOOK_WORDS = {
    # curiosity_gap
    "secret", "hidden", "revealed", "truth", "reason", "actually",
    "really", "behind", "unknown", "mystery", "discover", "surprising",
    # high_arousal
    "shocking", "insane", "incredible", "unbelievable", "amazing",
    "crazy", "wild", "intense", "epic", "ridiculous", "brutal",
    "terrifying", "heartbreaking",
    # urgency
    "now", "urgent", "immediately", "warning", "breaking", "before",
    "hurry", "limited", "last", "emergency", "critical", "fast", "quick",
    # negative_framing
    "mistake", "wrong", "never", "worst", "stop", "fail", "avoid",
    "danger", "terrible", "toxic", "destroy", "ruin", "problem",
    "risk", "scam", "trap", "deadly",
    # value_promise
    "hack", "trick", "tip", "way", "method", "strategy", "technique",
    "guide", "step", "easy", "simple", "free", "save", "learn",
    "shortcut", "formula",
    # superlatives
    "best", "worst", "most", "ever", "ultimate", "only", "first",
    "biggest", "smallest", "fastest", "top", "greatest", "least",
    "extreme", "record",
    # direct_address
    "watch", "listen", "look", "stop", "try", "imagine", "notice",
    "remember", "think", "guess", "wait",
    # contrarian
    "actually", "wrong", "myth", "lie", "fake", "overrated", "underrated",
    "controversial", "debate", "unpopular", "nobody", "everyone", "supposed",
    # proof_credibility
    "proof", "proven", "science", "study", "research", "data", "evidence",
    "tested", "experiment", "results", "confirmed", "official", "fact",
    # temporal
    "new", "just", "finally", "update", "change", "today", "recently",
    "latest", "suddenly", "anymore",
}

SECOND_PERSON_WORDS = {"you", "your", "yours", "yourself", "yourselves"}
FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}
INTERROGATIVE_WORDS = {
    "who", "what", "where", "when", "why", "how", "which", "whose", "whom",
    "did", "do", "does", "is", "are", "was", "were",
    "can", "could", "would", "will", "shall", "should",
    "have", "has", "had",
}

# ── Feature column order (physical, 82 feature columns) ─────────────────────
_CLIP_COLS = [f"clip_pca_{i:02d}" for i in range(CLIP_PCA_DIMS)]

FEATURE_COLUMNS = (
    # Text (10)
    [
        "word_count", "words_per_second",
        "hook_word_count", "hook_word_ratio",
        "question_count", "question_density",
        "second_person_ratio", "first_person_ratio",
        "first_5s_hook_word_ratio", "articulation_rate",
    ]
    # Audio speech (7)
    + [
        "energy_mean", "energy_var", "energy_first_3s_ratio",
        "pitch_range", "pitch_std",
        "speaking_rate_audio", "silence_ratio",
    ]
    # Voice quality (2)
    + ["jitter_local", "shimmer_local"]
    # Audio events (4)
    + ["music_presence", "music_fraction", "speech_music_ratio", "laughter_peak"]
    # Audio emotion (7)
    + [
        "arousal_mean", "valence_mean", "dominance_mean",
        "arousal_std", "arousal_peak",
        "arousal_arc_direction", "valence_arc_direction",
    ]
    # Visual (39 physical = 2 DOVER + 32 CLIP + 2 low-level + 2 scene + 3 face)
    + ["dover_aesthetic_score", "dover_technical_score"]
    + _CLIP_COLS
    + ["colorfulness", "brightness_mean"]
    + ["cut_count", "cuts_per_second"]
    + ["face_present_ratio", "largest_face_area_ratio_max", "face_count_median"]
    # Structural (8)
    + [
        "position_ratio", "is_intro", "is_outro",
        "segment_duration", "video_duration",
        "is_first_segment", "is_last_segment",
        "segment_novelty_to_neighbors",
    ]
    # Creator context (5 physical, one-hot)
    + [f"creator_category_{c}" for c in CATEGORY_ORDER]
)

KEY_COLUMNS = ["segment_id", "video_id", "creator_id", "category", "segment_index"]
META_COLUMNS = ["features_version", "extracted_at"]
OUTPUT_COLUMN_ORDER = KEY_COLUMNS + ["label"] + FEATURE_COLUMNS + META_COLUMNS


def ensure_local_dirs() -> None:
    for d in [
        LOCAL_SEGMENTS, LOCAL_TRANSCRIPTS, LOCAL_LABELED, LOCAL_METADATA,
        LOCAL_FEATURES_OUT, LOCAL_CLIP_EMBEDDINGS_OUT, LOCAL_MODELS, LOCAL_LOGS,
    ]:
        d.mkdir(parents=True, exist_ok=True)
