# config.py
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# Paths dataset Anda
DATA_CSV = os.path.join(DATA_RAW_DIR, 'data.csv')  # Multi-label Twitter
IN_HF_CSV = os.path.join(DATA_RAW_DIR, 'in_hf.csv')  # HF superset
ABUSIVE_CSV = os.path.join(DATA_RAW_DIR, 'abusive.csv')  # Abusive only
KAMUS_ALAY_CSV = os.path.join(DATA_RAW_DIR, 'new_kamusalay.csv')  # Kamus alay 
INDOTOXIC2024_CSV = os.path.join(DATA_RAW_DIR, 'indotoxic2024_annotated_data_v2_final.csv') 
SYNTHETIC_CHAT_CSV = os.path.join(DATA_RAW_DIR, 'synthetic_chat_id.csv')

# Model names (Hugging Face)
INDOBERT_MODEL = "indolem/indobert-base-uncased"
INDOBERTWEET_MODEL = "indolem/indobertweet-base-uncased"

# Hyperparams
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
PATIENCE = 2
LR = 2e-5
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42