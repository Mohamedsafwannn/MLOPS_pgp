import os
import pandas as pd


# -----------------------------
# File Paths
# -----------------------------
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"


# -----------------------------
# Load all required datasets
# -----------------------------
def load_data():
    orders = pd.read_csv(os.path.join(RAW_DATA_PATH, "olist_orders_dataset.csv"))

    order_items = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_order_items_dataset.csv")
    )

    customers = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_customers_dataset.csv")
    )

    products = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_products_dataset.csv")
    )

    sellers = pd.read_csv(
        os.path.join(RAW_DATA_PATH, "olist_sellers_dataset.csv")
    )

    return orders, order_items, customers, products, sellers


# -----------------------------
# Merge datasets
# -----------------------------
def merge_data(orders, order_items, customers, products, sellers):

    merged_df = pd.merge(
        orders,
        order_items,
        on="order_id",
        how="left"
    )

    merged_df = pd.merge(
        merged_df,
        customers,
        on="customer_id",
        how="left"
    )

    merged_df = pd.merge(
        merged_df,
        products,
        on="product_id",
        how="left"
    )

    merged_df = pd.merge(
        merged_df,
        sellers,
        on="seller_id",
        how="left"
    )

    return merged_df


# -----------------------------
# Save merged dataset
# -----------------------------
def save_data(df):

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    output_path = os.path.join(
        PROCESSED_DATA_PATH,
        "merged_dataset.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"Merged dataset saved successfully at:\n{output_path}")
    print(f"Shape: {df.shape}")


# -----------------------------
# Main Function
# -----------------------------
def main():

    print("Loading datasets...")

    orders, order_items, customers, products, sellers = load_data()

    print("Datasets loaded successfully.")

    print("Merging datasets...")

    merged_df = merge_data(
        orders,
        order_items,
        customers,
        products,
        sellers
    )

    print("Datasets merged successfully.")

    save_data(merged_df)


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    main()