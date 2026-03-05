import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from config import *

# Pastikan folder results ada
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/visualizations', exist_ok=True)

def evaluate_model(model, encodings, device='cpu', model_name="model", skema="60:40", set_name="Test"):
    """
    Evaluasi model pada data test/val secara lengkap.
    Mendukung tracking skema split untuk keperluan tabel komparasi.
    """
    model.eval()
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], encodings['labels'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 1. Hitung Metrik Macro (Sesuai kebutuhan TA: Acc, Prec, Rec, F1)
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average='macro', zero_division=0
    )

    # 2. Buat DataFrame (Sudah termasuk kolom Skema dan Set)
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Skema Split': [skema],
        'Set': [set_name],
        'Accuracy': [round(accuracy, 4)],
        'Macro Precision': [round(precision, 4)],
        'Macro Recall': [round(recall, 4)],
        'Macro F1-Score': [round(f1, 4)]
    })

    print(f"\n>>> Evaluasi {model_name} | Skema: {skema} | Set: {set_name}")
    print(metrics_df.to_string(index=False))

    # 3. Penamaan File Unik agar tidak saling menimpa
    # Contoh: indobert_6040_test
    safe_skema = skema.replace(":", "")
    file_id = f"{model_name.lower().replace(' ', '_')}_{safe_skema}_{set_name.lower()}"

    # 4. Visualisasi Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Ofensif', 'Ofensif'],
                yticklabels=['Non-Ofensif', 'Ofensif'])
    plt.title(f'Confusion Matrix\n{model_name} ({set_name} {skema})')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    
    cm_path = f'results/visualizations/cm_{file_id}.png'
    plt.savefig(cm_path)
    plt.close()

    # 5. Simpan metrik mentah ke CSV per model/skema
    metrics_path = f'results/metrics/raw_metrics_{file_id}.csv'
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df, cm


def plot_loss_curve(history, model_name="model"):
    """
    Plot Training vs Validation Loss Curve.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o', linestyle='--')
    plt.plot(history['val_loss'], label='Val Loss', marker='s', linestyle='-')
    plt.title(f'Learning Curve: {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (CrossEntropy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Nama file curve yang unik
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    path = f'results/visualizations/loss_{safe_name}.png'
    plt.savefig(path)
    plt.close()