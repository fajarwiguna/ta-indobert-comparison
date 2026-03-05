import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from config import BATCH_SIZE, EPOCHS, LR, PATIENCE

def freeze_bert_encoder(model):
    """
    Freeze encoder BERT agar adaptasi bersifat ringan 
    sesuai dengan tujuan penelitian transfer learning.
    """
    for name, param in model.named_parameters():
        if name.startswith("bert"):
            param.requires_grad = False

def train_model(model, train_enc, val_enc, test_enc, device, model_name="model"):
    """
    Fungsi training dengan evaluasi lengkap (Val & Test).
    Mendukung skema split: 60:20:20, 70:15:15, atau 80:10:10.
    """
    model.to(device)
    freeze_bert_encoder(model)

    # ===== 1. Persiapan Dataset & Loader =====
    train_dataset = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_enc["labels"])
    val_dataset = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_enc["labels"])
    test_dataset = TensorDataset(test_enc["input_ids"], test_enc["attention_mask"], test_enc["labels"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===== 2. Optimizer & Scheduler =====
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ===== 3. Class Weighting (Penanganan Imbalance Data) =====
    labels_np = train_enc["labels"].cpu().numpy()
    class_weights = torch.tensor(
        compute_class_weight(class_weight="balanced", classes=np.unique(labels_np), y=labels_np),
        dtype=torch.float
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights (Ofensif vs Non-Ofensif): {class_weights.detach().cpu().numpy()}")

    # ===== 4. History Tracking =====
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],  # Menyimpan akurasi & F1 per epoch
        "test_results": None # Hasil akhir di data test
    }

    best_f1 = 0.0
    patience_counter = 0
    checkpoint_path = f"best_{model_name}.pt"

    # ===== 5. Training Loop =====
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # ===== 6. Validation Phase =====
        model.eval()
        val_loss = 0.0
        val_preds, val_golds = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_golds.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        v_acc = accuracy_score(val_golds, val_preds)
        v_f1  = f1_score(val_golds, val_preds, average="macro")
        v_prec = precision_score(val_golds, val_preds, average="macro", zero_division=0)
        v_rec  = recall_score(val_golds, val_preds, average="macro", zero_division=0)

        history["val_loss"].append(avg_val_loss)
        history["val_metrics"].append({"acc": v_acc, "f1": v_f1, "prec": v_prec, "rec": v_rec})

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Metric -> Acc: {v_acc:.4f}, Prec: {v_prec:.4f}, Rec: {v_rec:.4f}, F1: {v_f1:.4f}")

        # Early Stopping & Checkpoint
        if v_f1 > best_f1:
            best_f1 = v_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Model {model_name} terbaik disimpan (F1: {v_f1:.4f})")
        else:
            patience_counter += 1
            print(f"EarlyStopping: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("!! Early stopping diaktifkan !!")
                break

    # ==========================================
    # 7. FINAL TEST EVALUATION (DATA TEST)
    # ==========================================
    print(f"\n=== Evaluasi Akhir Data TEST ({model_name}) ===")
    
    # Load model terbaik yang disimpan saat training
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    test_preds, test_golds = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            test_golds.extend(labels.cpu().numpy())

    # Hitung Metrik Final pada Data Test
    history["test_results"] = {
        "accuracy": accuracy_score(test_golds, test_preds),
        "precision": precision_score(test_golds, test_preds, average="macro", zero_division=0),
        "recall": recall_score(test_golds, test_preds, average="macro", zero_division=0),
        "f1_score": f1_score(test_golds, test_preds, average="macro")
    }

    print("HASIL TEST FINAL:")
    for m, v in history["test_results"].items():
        print(f"Test {m.capitalize()}: {v:.4f}")

    return model, history