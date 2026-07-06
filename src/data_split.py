import pandas as pd
from sklearn.model_selection import train_test_split

# Read merged dataset
df = pd.read_csv("data/processed/merged_dataset.csv")

# Split data
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

# Save files
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Data splitting completed successfully.")