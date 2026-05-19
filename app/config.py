from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "biencoder_model"
INDEX_DIR = BASE_DIR / "index_data"
PRICE_LIST_DIR = BASE_DIR / "price_lists"

MAX_LEN = 128
DEFAULT_BATCH_SIZE = 16

