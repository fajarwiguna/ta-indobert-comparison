# src/model_builder.py
from transformers import AutoModelForSequenceClassification

def build_indobert_modified():
    """
    IndoBERT dengan adaptasi slang berbasis preprocessing.
    Arsitektur TIDAK diubah.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "indobenchmark/indobert-base-p1",
        num_labels=2
    )
    print("Loaded IndoBERT (Adaptasi Slang via Preprocessing)")
    return model


def build_indobertweet_baseline():
    """
    IndoBERTweet baseline murni tanpa adaptasi slang.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "indolem/indobertweet-base-uncased",
        num_labels=2
    )
    print("Loaded IndoBERTweet (Baseline Murni)")
    return model
