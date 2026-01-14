# src/evaluator.py
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

def evaluate_model(model, encodings, device='cpu', model_name="model"):
    """
    Evaluasi model pada data test/val
    Return: DataFrame metrik dan confusion matrix
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

    # Hitung metrik
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='macro')

    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [round(accuracy, 4)],
        'Precision': [round(precision, 4)],
        'Recall': [round(recall, 4)],
        'Macro F1-Score': [round(f1, 4)]
    })

    print(f"\n=== Hasil Evaluasi {model_name} ===")
    print(metrics_df.to_string(index=False))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Ofensif', 'Ofensif'],
                yticklabels=['Non-Ofensif', 'Ofensif'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    cm_path = f'results/visualizations/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix disimpan di: {cm_path}")

    # Simpan metrik ke CSV
    metrics_path = f'results/metrics/metrics_{model_name.lower().replace(" ", "_")}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrik disimpan di: {metrics_path}")

    return metrics_df, cm