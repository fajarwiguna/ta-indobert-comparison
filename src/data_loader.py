# src/data_loader.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *

def load_kamus_alay():
    """Load kamus alay dengan handling encoding dan separator yang aman.
    File new_kamusalay.csv biasanya:
    - Separator: tab (\t)
    - Encoding: cp1252 (Windows) atau latin-1
    - Tanpa header
    - Kolom 0: kata alay/slang
    - Kolom 1: kata formal
    """
    if not os.path.exists(KAMUS_ALAY_CSV):
        print(f"Warning: File {KAMUS_ALAY_CSV} tidak ditemukan. Normalisasi alay akan dilewati.")
        return {}  # Return empty dict jika file tidak ada

    # Urutan encoding yang paling sering berhasil untuk file Indonesia lama
    encodings_to_try = ['cp1252', 'latin-1', 'iso-8859-1', 'utf-8']
    separators_to_try = ['\t', ',']  # Tab paling umum, koma sebagai fallback

    for enc in encodings_to_try:
        for sep in separators_to_try:
            try:
                df = pd.read_csv(KAMUS_ALAY_CSV, header=None, encoding=enc, sep=sep, on_bad_lines='skip')
                # Pastikan ada minimal 2 kolom
                if df.shape[1] >= 2:
                    alay_dict = dict(zip(df[0].astype(str).str.strip().str.lower(),
                                         df[1].astype(str).str.strip().str.lower()))
                    print(f"Berhasil load kamus alay: {len(alay_dict)} pasang kata")
                    print(f"   Encoding: {enc}, Separator: '{sep}'")
                    print(f"   Contoh: '{list(alay_dict.keys())[:3]}' → '{list(alay_dict.values())[:3]}'")
                    return alay_dict
            except Exception as e:
                continue  # Coba kombinasi berikutnya

    # Fallback terakhir: latin-1 dengan tab (hampir selalu berhasil baca karakter)
    print("Fallback ke encoding 'latin-1' dan separator tab")
    df = pd.read_csv(KAMUS_ALAY_CSV, header=None, encoding='latin-1', sep='\t', on_bad_lines='skip')
    if df.shape[1] >= 2:
        alay_dict = dict(zip(df[0].astype(str).str.strip().str.lower(),
                             df[1].astype(str).str.strip().str.lower()))
    else:
        alay_dict = {}
    print(f"Loaded dengan fallback: {len(alay_dict)} entries")
    return alay_dict


def merge_datasets():
    """Gabung semua dataset menjadi satu dengan label biner ofensif + handling encoding"""
    dfs = []
    
    def safe_read_csv(path, **kwargs):
        """Baca CSV dengan trial encoding"""
        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            return None
        
        encodings = ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, **kwargs)
                print(f"Berhasil load {os.path.basename(path)} dengan encoding: {enc}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Gagal {enc}: {e}")
                continue
        # Fallback akhir
        print(f"Fallback latin-1 untuk {os.path.basename(path)}")
        return pd.read_csv(path, encoding='latin-1', **kwargs)
    
    # 1. data.csv (multi-label Twitter hate speech)
    df1 = safe_read_csv(DATA_CSV)
    if df1 is not None:
        df1['text'] = df1['Tweet'].astype(str)
        df1['label'] = ((df1['HS'] == 1) | (df1['Abusive'] == 1)).astype(int)
        dfs.append(df1[['text', 'label']])
        print(f"Loaded data.csv: {len(df1)} samples")
    
    # 2. in_hf.csv (HF superset)
    df2 = safe_read_csv(IN_HF_CSV)
    if df2 is not None:
        df2['text'] = df2['text'].astype(str)
        df2['label'] = df2['labels'].astype(int)
        dfs.append(df2[['text', 'label']])
        print(f"Loaded in_hf.csv: {len(df2)} samples")
    
    # 3. abusive.csv (hanya abusive)
    df3 = safe_read_csv(ABUSIVE_CSV)
    if df3 is not None:
        df3['text'] = df3['ABUSIVE'].astype(str)
        df3['label'] = 1
        dfs.append(df3[['text', 'label']])
        print(f"Loaded abusive.csv: {len(df3)} samples")
    
    # 4. IndoToxic2024 (annotated)
    df4 = safe_read_csv(INDOTOXIC2024_CSV)
    if df4 is not None:
        df4['text'] = df4['text'].astype(str)

        offensive_cols = [
            'toxicity',
            'polarized',
            'profanity_obscenity',
            'threat_incitement_to_violence',
            'insults',
            'identity_attack',
            'sexually_explicit'
        ]

        missing_cols = [c for c in offensive_cols if c not in df4.columns]
        if missing_cols:
            raise ValueError(f"Kolom hilang di IndoToxic2024: {missing_cols}")

        # 🔧 FIX UTAMA
        for col in offensive_cols:
            df4[col] = pd.to_numeric(df4[col], errors='coerce').fillna(0)

        df4['label'] = (df4[offensive_cols].sum(axis=1) > 0).astype(int)

        dfs.append(df4[['text', 'label']])
        print(f"Loaded IndoToxic2024: {len(df4)} samples")

        
    # 5. synthetic_chat_id.csv (LLM-generated synthetic data)
    df5 = safe_read_csv(SYNTHETIC_CHAT_CSV)
    if df5 is not None:
        df5['text'] = df5['text'].astype(str)
        df5['label'] = df5['label'].astype(int)
        dfs.append(df5[['text', 'label']])
        print(f"Loaded synthetic_chat_id.csv: {len(df5)} samples")

    
    if not dfs:
        raise ValueError("Tidak ada dataset yang berhasil dimuat!")
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"\nTotal merged dataset: {len(full_df)} samples")
    print("Distribusi label:")
    print(full_df['label'].value_counts(normalize=True))
    
    return full_df


def split_data(df):
    train_df, temp_df = train_test_split(df, test_size=TEST_SIZE + VAL_SIZE, random_state=RANDOM_STATE, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=VAL_SIZE / (TEST_SIZE + VAL_SIZE), random_state=RANDOM_STATE, stratify=temp_df['label'])
    return train_df, val_df, test_df
