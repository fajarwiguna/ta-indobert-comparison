import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from config import BATCH_SIZE, EPOCHS, LR, PATIENCE


def freeze_bert_encoder(model):
    """
    Freeze encoder BERT.
    Digunakan untuk IndoBERT agar adaptasi bersifat ringan
    dan konsisten dengan tujuan penelitian.
    """
    for name, param in model.named_parameters():
        if name.startswith("bert"):
            param.requires_grad = False


def train_model(model, train_enc, val_enc, device):
    model.to(device)

    # ===== Freeze Encoder (AMAN untuk IndoBERT & IndoBERTweet) =====
    freeze_bert_encoder(model)

    # ===== Dataset =====
    train_dataset = TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        train_enc["labels"]
    )

    val_dataset = TensorDataset(
        val_enc["input_ids"],
        val_enc["attention_mask"],
        val_enc["labels"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ===== Optimizer =====
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # ===== Scheduler =====
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ===== CLASS WEIGHTING =====
    labels_np = train_enc["labels"].cpu().numpy()

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels_np),
        y=labels_np
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("Class weights:", class_weights.detach().cpu().numpy())

    # ===== History untuk Plot =====
    history = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": [],
        "f1": []
    }

    # ===== Early Stopping =====
    best_f1 = 0.0
    patience_counter = 0

    # ===== Training Loop =====
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids, attention_mask, labels = [
                x.to(device) for x in batch
            ]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        print(f"Training Loss: {avg_train_loss:.4f}")

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        preds, golds = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [
                    x.to(device) for x in batch
                ]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                golds.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(golds, preds)
        f1 = f1_score(golds, preds, average="macro")

        history["val_loss"].append(avg_val_loss)
        history["accuracy"].append(acc)
        history["f1"].append(f1)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {acc:.4f} | Macro F1-score: {f1:.4f}")

        # ===== Early Stopping =====
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("✓ Model terbaik disimpan")
        else:
            patience_counter += 1
            print(f"EarlyStopping Counter: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping diaktifkan")
                break

    print("\nTraining selesai (class weighting + transfer learning).")

    return model, history
