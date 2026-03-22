from src.data_prep import load_data, clean_data, save_processed

RAW_PATH = "data/raw/spotify_data_clean.csv"
OUT_PATH = "data/processed/spotify_data_processed.csv"

df = load_data(RAW_PATH)
df_clean = clean_data(df)
save_processed(df_clean, OUT_PATH)

print("Done.")
print(df_clean.shape)
print(df_clean.head())