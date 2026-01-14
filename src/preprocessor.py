# src/preprocessor.py
import re
import torch
from transformers import AutoTokenizer
from src.data_loader import load_kamus_alay
from config import MAX_LENGTH

alay_dict = load_kamus_alay()

def normalize_alay(text):
    words = text.lower().split()
    normalized = [alay_dict.get(word, word) for word in words]
    return ' '.join(normalized)

def clean_text(text, use_slang=True):
    text = text.lower()
    if use_slang:
        text = normalize_alay(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def tokenize_data(df, model_name, use_slang=True):
    df = df.copy()
    df['text'] = df['text'].apply(lambda x: clean_text(x, use_slang))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    encodings['labels'] = torch.tensor(df['label'].values, dtype=torch.long)
    return encodings
