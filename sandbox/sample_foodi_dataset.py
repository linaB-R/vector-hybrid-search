import pandas as pd

# S3 path to the FooDI-ML dataset
url = "s3://glovo-products-dataset-d1c9720d/glovo-foodi-ml-dataset.csv"

# Spanish-speaking country codes from the dataset
es_cc = {"ES", "AR", "PE", "DO", "CR", "UY", "EC", "HN", "GT", "CL", "PA", "PR"}

# Read CSV in chunks from S3 and filter for Spanish-speaking countries
chunks = pd.read_csv(url, storage_options={"anon": True}, chunksize=200_000)
bag = []
total = 0

print("Sampling 50 entries from Spanish-speaking countries...")

for c in chunks:
    sub = c[c["country_code"].isin(es_cc)]
    bag.append(sub)
    total += len(sub)
    print(f"Collected {total} entries so far...")
    if total >= 50:
        break

# Concatenate and take exactly 50 rows
df_sample = pd.concat(bag, ignore_index=True).head(50)

# Save to local CSV
output_file = "sample_foodi_es_50.csv"
df_sample.to_csv(output_file, index=False)

print(f"Saved {len(df_sample)} entries to {output_file}")
print(f"Dataset shape: {df_sample.shape}")
print(f"Countries included: {sorted(df_sample['country_code'].unique())}")
