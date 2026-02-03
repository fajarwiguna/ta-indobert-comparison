import torch
import pandas as pd
from config import *

from src.data_loader import merge_datasets, split_data_multi_scheme
from src.preprocessor import tokenize_data
from src.model_builder import build_indobert_modified, build_indobertweet_baseline
from src.trainer import train_model
from src.evaluator import evaluate_model, plot_loss_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load dataset
full_df = merge_datasets()

all_results = []
all_histories = {}  # simpan loss curve semua skema

best_f1 = 0
best_scheme = None

for scheme in ["60:40", "70:30", "80:20"]:

    print(f"\n===== SKEMA SPLIT {scheme} =====")

    # 2. Split Data (DATA PREPARATION)
    train_df, val_df, test_df = split_data_multi_scheme(
        full_df, scheme=scheme
    )

    # 3. Tokenization
    train_enc_bert = tokenize_data(train_df, INDOBERT_MODEL, use_slang=True)
    val_enc_bert   = tokenize_data(val_df, INDOBERT_MODEL, use_slang=True)
    test_enc_bert  = tokenize_data(test_df, INDOBERT_MODEL, use_slang=True)

    train_enc_tweet = tokenize_data(train_df, INDOBERTWEET_MODEL, use_slang=False)
    val_enc_tweet   = tokenize_data(val_df, INDOBERTWEET_MODEL, use_slang=False)
    test_enc_tweet  = tokenize_data(test_df, INDOBERTWEET_MODEL, use_slang=False)

    # 4. Training
    model_bert = build_indobert_modified()
    trained_bert, history_bert = train_model(
        model_bert,
        train_enc_bert,
        val_enc_bert,
        device
    )

    model_tweet = build_indobertweet_baseline()
    trained_tweet, history_tweet = train_model(
        model_tweet,
        train_enc_tweet,
        val_enc_tweet,
        device
    )

    # simpan semua history
    all_histories[scheme] = {
        "bert": history_bert,
        "tweet": history_tweet
    }

    # 5. Evaluation
    metrics_bert, _ = evaluate_model(
        trained_bert,
        test_enc_bert,
        device,
        model_name=f"IndoBERT + Slang ({scheme})"
    )

    metrics_tweet, _ = evaluate_model(
        trained_tweet,
        test_enc_tweet,
        device,
        model_name=f"IndoBERTweet ({scheme})"
    )

    all_results.extend([metrics_bert, metrics_tweet])

    # 6. Tentukan skema terbaik (berdasarkan F1 IndoBERTweet)
    current_f1 = float(metrics_tweet.loc[0, "Macro F1-Score"])

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_scheme = scheme

# 7. Plot LOSS CURVE SEMUA SKEMA
print("\n=== MENAMPILKAN LOSS CURVE SEMUA SKEMA ===")

for scheme, histories in all_histories.items():
    plot_loss_curve(
        histories["bert"],
        f"IndoBERT + Slang ({scheme})"
    )
    plot_loss_curve(
        histories["tweet"],
        f"IndoBERTweet ({scheme})"
    )

print(f"\nSkema terbaik berdasarkan F1-score: {best_scheme}")

# 8. Simpan hasil komparasi
comparison_df = pd.concat(all_results, ignore_index=True)

print("\n=== KOMPARASI PERFORMA SEMUA SKEMA ===")
print(comparison_df)

comparison_df.to_csv(
    "results/metrics/comparison_multi_split_2026.csv",
    index=False
)
