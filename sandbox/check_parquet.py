import pandas as pd
import os


########################################################
# check parquet file by manually
########################################################

df = pd.read_parquet("data/20250123_data.parquet")

# check the column names
print(df.columns)

# configre pandas to show all columns
pd.set_option('display.max_columns', None)

# check the column names
print(df.head())

# check the column names
print(df.tail())


