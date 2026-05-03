import pandas as pd
from text_preprocessing import clean_email_text, remove_stopwords_and_lemmatize

def silver_transformation(input_path, output_path):
    df = pd.read_parquet(input_path)

    # Loại bỏ trùng lặp và null
    df = df.drop_duplicates(subset = ['text']).dropna(subset = ['text'])

    df = df.reset_index(drop=True)

    # Làm sạch văn bản
    df['cleaned_text'] = df['text'].apply(clean_email_text)

    # NLP Processing
    df['processed_text'] = df['cleaned_text'].apply(remove_stopwords_and_lemmatize)

    df.to_parquet(output_path, index = False)
    print(f"Sliver Layer complete. Saved to: {output_path}")

if __name__ == "__main__":
    silver_transformation('../data/bronze/emails_bronze.parquet', '../data/silver/emails_silver.parquet')