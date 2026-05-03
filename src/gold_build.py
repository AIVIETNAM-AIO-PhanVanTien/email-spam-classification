import os
import json
import pickle
import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Label encoding
def encode_labels(df: pd.DataFrame, label_col: str = 'label'):
    """
    Encode label ham/spam → 0/1.
    Trả về df đã có cột label_encoded và LabelEncoder đã fit.
    """
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df[label_col])
    # In ra để xác nhận mapping (ham=0, spam=1 hay ngược lại)
    mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    print(f"  Label mapping: {mapping}")
    return df, le


# Train/Test split
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified split để đảm bảo tỉ lệ spam/ham cân bằng ở cả 2 tập.
    Seed cố định để tất cả notebooks dùng cùng một split.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_encoded']
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    return train_df, test_df


# TF-IDF VECTORIZATION
def build_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame,
                max_features: int = 10000, ngram_range: tuple = (1, 2)):
    """
    Fit TF-IDF chỉ trên train set, transform cả train lẫn test.
    Tránh data leakage: vectorizer không được thấy test set khi fit.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,    # unigram + bigram
        sublinear_tf=True,          # dùng log(tf) thay vì raw tf — giảm ảnh hưởng từ lặp nhiều
        strip_accents='unicode',
        min_df=2,                   # bỏ từ chỉ xuất hiện trong 1 email (nhiễu)
    )

    X_train_tfidf = vectorizer.fit_transform(train_df['processed_text'])
    X_test_tfidf  = vectorizer.transform(test_df['processed_text'])

    return vectorizer, X_train_tfidf, X_test_tfidf



# SAVE ARTIFACTS
def save_artifacts(
    train_df, test_df,
    vectorizer, le,
    X_train_tfidf, X_test_tfidf,
    output_dir: str,
    model_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Parquet — full dataframe (text + statistical features + label)
    train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test.parquet'),   index=False)
    print(f"  Saved train.parquet ({len(train_df)} rows)")
    print(f"  Saved test.parquet  ({len(test_df)} rows)")

    # TF-IDF sparse matrices
    sparse.save_npz(os.path.join(output_dir, 'X_train_tfidf.npz'), X_train_tfidf)
    sparse.save_npz(os.path.join(output_dir, 'X_test_tfidf.npz'),  X_test_tfidf)
    print(f"  Saved X_train_tfidf.npz {X_train_tfidf.shape}")
    print(f"  Saved X_test_tfidf.npz  {X_test_tfidf.shape}")

    # TF-IDF vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved tfidf_vectorizer.pkl")

    # Label encoder
    le_path = os.path.join(model_dir, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"  Saved label_encoder.pkl")

    # Feature names (dùng để debug / interpret model)
    feature_names = {
        'tfidf_features': vectorizer.get_feature_names_out().tolist(),
    }
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Metadata
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    class_dist    = train_df['label'].value_counts().to_dict()

    metadata = {
        'created_at':         datetime.datetime.now().isoformat(),
        'train_size':         len(train_df),
        'test_size':          len(test_df),
        'tfidf_vocab_size':   len(vectorizer.vocabulary_),
        'tfidf_max_features': vectorizer.max_features,
        'tfidf_ngram_range':  list(vectorizer.ngram_range),
        'label_mapping':      label_mapping,
        'train_class_distribution': class_dist,
    }
    with open(os.path.join(output_dir, 'gold_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved gold_metadata.json")


# MAIN PIPELINE
def build_gold(
    input_path:   str,
    output_dir:   str,
    model_dir:    str,
    test_size:    float = 0.2,
    random_state: int   = 42,
    tfidf_max_features: int   = 10000,
    tfidf_ngram_range:  tuple = (1, 2),
):

    # Load silver
    print("\n[1/4] Loading silver data...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows | Columns: {df.columns.tolist()}")


    # Encode labels
    print("\n[2/4] Encoding labels...")
    df, le = encode_labels(df)

    # Split
    print("\n[3/4] Splitting train/test...")
    train_df, test_df = split_data(df, test_size=test_size, random_state=random_state)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"  Train spam ratio: {train_df['label_encoded'].mean():.2%}")
    print(f"  Test  spam ratio: {test_df['label_encoded'].mean():.2%}")

    # TF-IDF
    print("\n[4/4] Building TF-IDF...")
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf(
        train_df, test_df,
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range
    )
    print(f"  Vocab size: {len(vectorizer.vocabulary_)}")

    # Save
    print("\n[+] Saving artifacts...")
    save_artifacts(
        train_df, test_df,
        vectorizer, le,
        X_train_tfidf, X_test_tfidf,
        output_dir, model_dir
    )

    print("Gold Layer complete")



# ENTRY POINT
if __name__ == "__main__":
    build_gold(
        input_path          = '../data/silver/emails_silver.parquet',
        output_dir          = '../data/gold/',
        model_dir           = '../models/',
        test_size           = 0.2,
        random_state        = 42,
        tfidf_max_features  = 10000,
        tfidf_ngram_range   = (1, 2),
    )