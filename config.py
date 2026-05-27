import datetime as dt
import os
from pathlib import Path

_EXPERIMENT_DIR = Path(__file__).resolve().parent

CUTOFF_UTC = dt.datetime(2026, 4, 8, 0, 0, 0, tzinfo=dt.timezone.utc)
CUTOFF_MS = int(CUTOFF_UTC.timestamp() * 1000)
GAP_MS = 0

SUBMISSION_K = 160
RANDOM_SEED = 42

MAXLEN = int(os.environ.get("TISASREC_MAXLEN", "150"))
HIDDEN_UNITS = int(os.environ.get("TISASREC_HIDDEN", "256"))
NUM_BLOCKS = int(os.environ.get("TISASREC_BLOCKS", "3"))
NUM_HEADS = int(os.environ.get("TISASREC_HEADS", "4"))
DROPOUT_RATE = float(os.environ.get("TISASREC_DROPOUT", "0.2"))
L2_EMB = float(os.environ.get("TISASREC_L2", "0.0"))
TIME_SPAN = int(os.environ.get("TISASREC_TIME_SPAN", "512"))

VOCAB_SIZE = int(os.environ.get("TISASREC_VOCAB", "400000"))
BATCH_SIZE = int(os.environ.get("TISASREC_BATCH", "512"))
EPOCHS = int(os.environ.get("TISASREC_EPOCHS", "12"))
LR = float(os.environ.get("TISASREC_LR", "0.001"))
NUM_NEG = int(os.environ.get("TISASREC_NUM_NEG", "4"))
WEIGHT_DECAY = float(os.environ.get("TISASREC_WD", "0.01"))

ALPHA_CONTACT = 5.0
ALPHA_VIEW = 1.0
# Последние N позиций на user в train (0 = все; >0 снижает переобучение на длинном хвосте)
MAX_POSITIONS_PER_USER = int(os.environ.get("TISASREC_MAX_POS_PER_USER", "250"))
USE_TRAIN_FOR_VOCAB = os.environ.get("TISASREC_USE_TRAIN", "1") == "1"
USE_TRAIN_FOR_SEQUENCES = os.environ.get("TISASREC_USE_TRAIN_SEQ", "1") == "1"
TRAIN_SHARD_MOD = 100

TRAIN_MAX_USERS = int(os.environ.get("TISASREC_TRAIN_MAX_USERS", "0"))
TRAIN_MAX_STEPS = int(os.environ.get("TISASREC_TRAIN_MAX_STEPS", "0"))
PREDICT_BATCH = int(os.environ.get("TISASREC_PRED_BATCH", "256"))
SCORE_CHUNK = int(os.environ.get("TISASREC_SCORE_CHUNK", "100000"))
PRED_MAX_USERS = int(os.environ.get("TISASREC_PRED_MAX_USERS", "0"))
NUM_WORKERS = int(os.environ.get("TISASREC_NUM_WORKERS", "4"))
GRAD_CLIP = float(os.environ.get("TISASREC_GRAD_CLIP", "1.0"))

ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", str(_EXPERIMENT_DIR / "artifacts")),
).expanduser().resolve()

POLARS_MAX_THREADS = int(os.environ.get("POLARS_MAX_THREADS", "32"))


def resolve_data_dir() -> Path:
    """Каталог datafest: рядом с кодом, в подпапке или через DATA_DIR."""
    explicit = os.environ.get("DATA_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for name in ("datafest_2026_v2_v4", "data"):
        candidate = _EXPERIMENT_DIR / name
        if (candidate / "eval_users.csv").is_file():
            return candidate.resolve()
    if (_EXPERIMENT_DIR / "eval_users.csv").is_file():
        return _EXPERIMENT_DIR.resolve()
    return _EXPERIMENT_DIR.resolve()


DATA_DIR = resolve_data_dir()
