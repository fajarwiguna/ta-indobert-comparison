import torch
import pandas as pd
import os
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

for scheme in ["60:40", "70:30", "80:20"]:

    print(f"\n" + "="*30)
    print(f"===== SKEMA SPLIT {scheme} =====")
    print("="*30)

    # 2. Split Data (Hasil: Train, Val, Test)
    train_df, val_df, test_df = split_data_multi_scheme(
        full_df, scheme=scheme
    )

    # 3. Tokenization (IndoBERT & IndoBERTweet)
    train_enc_bert = tokenize_data(train_df, INDOBERT_MODEL, use_slang=True)
    val_enc_bert   = tokenize_data(val_df, INDOBERT_MODEL, use_slang=True)
    test_enc_bert  = tokenize_data(test_df, INDOBERT_MODEL, use_slang=True)

    train_enc_tweet = tokenize_data(train_df, INDOBERTWEET_MODEL, use_slang=False)
    val_enc_tweet   = tokenize_data(val_df, INDOBERTWEET_MODEL, use_slang=False)
    test_enc_tweet  = tokenize_data(test_df, INDOBERTWEET_MODEL, use_slang=False)

    # 4. Training IndoBERT + Slang
    print(f"\n>> Training IndoBERT + Slang ({scheme})...")
    model_bert = build_indobert_modified()
    trained_bert, history_bert = train_model(
        model_bert,
        train_enc_bert,
        val_enc_bert,
        test_enc_bert, # Tambahkan test_enc
        device,
        model_name="IndoBERT_Slang"
    )

    # 5. Training IndoBERTweet
    print(f"\n>> Training IndoBERTweet ({scheme})...")
    model_tweet = build_indobertweet_baseline()
    trained_tweet, history_tweet = train_model(
        model_tweet,
        train_enc_tweet,
        val_enc_tweet,
        test_enc_tweet, # Tambahkan test_enc
        device,
        model_name="IndoBERTweet"
    )

    # Simpan semua history untuk plotting
    all_histories[scheme] = {
        "bert": history_bert,
        "tweet": history_tweet
    }

    # 6. Evaluation (Menghasilkan metrik untuk tabel lengkap)
    # Evaluasi VALIDATION (Opsional untuk tabel komparasi internal)
    m_val_bert, _ = evaluate_model(trained_bert, val_enc_bert, device, 
                                 model_name="IndoBERT + Slang", skema=scheme, set_name="Validation")
    m_val_tweet, _ = evaluate_model(trained_tweet, val_enc_tweet, device, 
                                  model_name="IndoBERTweet", skema=scheme, set_name="Validation")

    # Evaluasi TEST (Ini yang paling utama untuk hasil penelitian)
    m_test_bert, _ = evaluate_model(trained_bert, test_enc_bert, device, 
                                  model_name="IndoBERT + Slang", skema=scheme, set_name="Test")
    m_test_tweet, _ = evaluate_model(trained_tweet, test_enc_tweet, device, 
                                   model_name="IndoBERTweet", skema=scheme, set_name="Test")

    # Kumpulkan semua hasil untuk diconcat nanti
    all_results.extend([m_val_bert, m_val_tweet, m_test_bert, m_test_tweet])

# 7. Plot LOSS CURVE
print("\n=== MENAMPILKAN LOSS CURVE SEMUA SKEMA ===")
for scheme, histories in all_histories.items():
    plot_loss_curve(histories["bert"], f"IndoBERT + Slang ({scheme})")
    plot_loss_curve(histories["tweet"], f"IndoBERTweet ({scheme})")

# 8. Simpan hasil komparasi akhir ke satu CSV
comparison_df = pd.concat(all_results, ignore_index=True)

# Urutkan agar rapi: Skema dulu, baru Set, lalu Model
comparison_df = comparison_df.sort_values(by=['Skema Split', 'Set', 'Model'])

print("\n=== KOMPARASI PERFORMA SEMUA SKEMA ===")
print(comparison_df.to_string(index=False))

os.makedirs('results/metrics', exist_ok=True)
comparison_df.to_csv("results/metrics/final_comparison_all_metrics.csv", index=False)
print(f"\nSemua hasil tersimpan di results/metrics/final_comparison_all_metrics.csv")