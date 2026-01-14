# config.py
import os

# =============================
# Environment Detection
# =============================
IS_COLAB = os.path.exists("/content")

# =============================
# Base Directory
# =============================
if IS_COLAB:
    BASE_DIR = "/content/ta-indobert-comparison"
    DATA_RAW_DIR = "/content/drive/MyDrive/ta-indobert-comparison/data/raw"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# =============================
# Dataset Paths
# =============================
DATA_CSV = os.path.join(DATA_RAW_DIR, "data.csv")
IN_HF_CSV = os.path.join(DATA_RAW_DIR, "in_hf.csv")
ABUSIVE_CSV = os.path.join(DATA_RAW_DIR, "abusive.csv")
KAMUS_ALAY_CSV = os.path.join(DATA_RAW_DIR, "new_kamusalay.csv")
INDOTOXIC2024_CSV = os.path.join(DATA_RAW_DIR, "indotoxic2024_annotated_data_v2_final.csv")
SYNTHETIC_CHAT_CSV = os.path.join(DATA_RAW_DIR, "synthetic_chat_id.csv")

# =============================
# Model Names
# =============================
INDOBERT_MODEL = "indolem/indobert-base-uncased"
INDOBERTWEET_MODEL = "indolem/indobertweet-base-uncased"

# =============================
# Training Hyperparameters
# =============================
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
PATIENCE = 2
LR = 2e-5
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
