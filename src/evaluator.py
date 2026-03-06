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

def evaluate_model(model, encodings, device='cpu', model_name="model", skema="", set_name="Test"):
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

    # Buat DataFrame dengan kolom tambahan agar sinkron dengan sorting di main.py
    metrics_df = pd.DataFrame({
        'Skema Split': [skema],
        'Set': [set_name],
        'Model': [model_name],
        'Accuracy': [round(accuracy, 4)],
        'Macro Precision': [round(precision, 4)],
        'Macro Recall': [round(recall, 4)],
        'Macro F1-Score': [round(f1, 4)]
    })

    print(f"\n=== Hasil Evaluasi {model_name} ({set_name} - {skema}) ===")
    print(metrics_df.to_string(index=False))

    # Penamaan file yang unik berdasarkan skema dan set
    file_suffix = f"{model_name.lower().replace(' ', '_')}_{skema.replace(':', '')}_{set_name.lower()}"

    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Ofensif', 'Ofensif'],
                yticklabels=['Non-Ofensif', 'Ofensif'])
    plt.title(f'CM - {model_name} ({skema} {set_name})')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.tight_layout()
    
    cm_path = f'results/visualizations/cm_{file_suffix}.png'
    plt.savefig(cm_path)
    plt.close()

    # Simpan metrik individu ke CSV
    metrics_path = f'results/metrics/metrics_{file_suffix}.csv'
    metrics_df.to_csv(metrics_path, index=False)

    return metrics_df, cm

def plot_loss_curve(history, model_name="model"):
    """
    Plot Training vs Validation Loss Curve
    """
    plt.figure(figsize=(6,4))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title(f'Loss Curve - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Bersihkan nama file dari karakter spesial
    clean_name = model_name.lower().replace(" ", "_").replace(":", "").replace("+", "plus").replace("(", "").replace(")", "")
    path = f'results/visualizations/loss_{clean_name}.png'
    plt.savefig(path)
    plt.close()
    print(f"Loss curve disimpan di: {path}")