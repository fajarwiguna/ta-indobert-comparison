# main.py
import torch
import pandas as pd
from config import *
from src.data_loader import merge_datasets, split_data
from src.preprocessor import tokenize_data
from src.model_builder import build_indobert_modified, build_indobertweet_baseline
from src.trainer import train_model
from src.evaluator import evaluate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load dataset
full_df = merge_datasets()

# 2. Split
train_df, val_df, test_df = split_data(full_df)

# 3. Tokenization
train_enc_bert = tokenize_data(train_df, INDOBERT_MODEL, use_slang=True)
val_enc_bert = tokenize_data(val_df, INDOBERT_MODEL, use_slang=True)
test_enc_bert = tokenize_data(test_df, INDOBERT_MODEL, use_slang=True)

train_enc_tweet = tokenize_data(train_df, INDOBERTWEET_MODEL, use_slang=False)
val_enc_tweet = tokenize_data(val_df, INDOBERTWEET_MODEL, use_slang=False)
test_enc_tweet = tokenize_data(test_df, INDOBERTWEET_MODEL, use_slang=False)

# 4. Train models
model_bert = build_indobert_modified()
trained_bert = train_model(model_bert, train_enc_bert, val_enc_bert, device)

model_tweet = build_indobertweet_baseline()
trained_tweet = train_model(model_tweet, train_enc_tweet, val_enc_tweet, device)

# 5. Evaluation
metrics_bert, _ = evaluate_model(
    trained_bert, test_enc_bert, device,
    model_name="IndoBERT + Slang Normalization"
)

metrics_tweet, _ = evaluate_model(
    trained_tweet, test_enc_tweet, device,
    model_name="IndoBERTweet Baseline"
)

# 6. Comparison table (VERTIKAL = AKADEMIK BENAR)
comparison_df = pd.concat(
    [metrics_bert, metrics_tweet],
    ignore_index=True
)

print("\n=== KOMPARASI PERFORMA MODEL ===")
print(comparison_df)

comparison_df.to_csv(
    "results/metrics/comparison_2026.csv",
    index=False
)
