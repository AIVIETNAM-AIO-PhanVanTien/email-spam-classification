import pandas as pd
import os
import datetime

def ingest_bronze_data(file_path: str, output_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df['ingestion_timestamp'] = datetime.datetime.now()
    df['source_file'] = os.path.basename(file_path)

    output_file = os.path.join(output_path, 'emails_bronze.parquet')

    df.to_parquet(output_file, index=False)

    print(f"Bronze Layer complete. Saved to: {output_file}")

if __name__ == "__main__":
    raw_data = '../data/raw/spam_Emails_data.csv'
    bronze_output = '../data/bronze/'
    ingest_bronze_data(raw_data, bronze_output)





