import pandas as pd
from text_preprocessing import clean_email_text, remove_stopwords_and_lemmatize
from pathlib import Path


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
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    bronze_data = BASE_DIR / "data" / "bronze" / "emails_bronze.parquet"
    
    silver_output_file = BASE_DIR / "data" / "silver" / "emails_silver.parquet"
    
    silver_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    silver_transformation(bronze_data, silver_output_file)