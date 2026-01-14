import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, f1_score
from config import BATCH_SIZE, EPOCHS, LR, PATIENCE

def freeze_bert_encoder(model):
    """
    Freeze encoder IndoBERT
    Custom slang embedding + classifier tetap dilatih
    """
    for name, param in model.named_parameters():
        if name.startswith("bert"):
            param.requires_grad = False

def train_model(model, train_enc, val_enc, device):
    model.to(device)

    # Freeze IndoBERT encoder (transfer learning terkontrol)
    freeze_bert_encoder(model)

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [
                x.to(device) for x in batch
            ]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs["loss"]
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # ===== VALIDASI =====
        model.eval()
        val_loss = 0
        preds, golds = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [
                    x.to(device) for x in batch
                ]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs["loss"].item()
                logits = outputs["logits"]

                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                golds.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(golds, preds)
        f1 = f1_score(golds, preds, average="weighted")

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

        # ===== EARLY STOPPING =====
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

    print("\nTraining selesai (custom embedding + fine-tuning).")
    return model
