import pandas as pd

from src.database import engine

CSV_PATH = "data/raw/train.csv"
TABLE_NAME = "loan_data"

CHUNK_SIZE = 10000

for i, chunk in enumerate(
    pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)
):
    chunk.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists="replace" if i == 0 else "append",
        index=False
    )

    print(f"Imported chunk {i + 1}: {len(chunk)} rows")

print("CSV import completed successfully!")