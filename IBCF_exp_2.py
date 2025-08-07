import pandas as pd
import os
import csv
from datetime import datetime
from IBCF_recsys import (
    build_user_item_matrix,
    split_ratings_by_user,
    build_item_similarity_matrix,
    evaluate_mae,
    save_similarity_matrix,
)


# === Setup ===
csv_filename = 'results_item_based_sensitivity_k.csv'
dataset_name = 'MovieLens100k'
method = 'item-based'
X = 0.8  # train ratio
k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]

# === Load dataset ===
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(
    '../datasets/MovieLens/ml-100k/u.data', 
    sep='\t', 
    names=column_names,
    encoding='latin-1'
)

train_df, test_df = split_ratings_by_user(ratings, test_ratio=1 - X)

user_item_matrix = build_user_item_matrix(train_df)

similarity_matrix = build_item_similarity_matrix(user_item_matrix)
save_similarity_matrix(similarity_matrix, path=f"item_similarity_matrix_X{X}.pkl")

def append_result_to_csv(dataset, method, k, X_train_ratio, mae, csv_file):
    now = datetime.now().isoformat()
    row = [dataset, method, k, X_train_ratio, mae, now]
    header = ['dataset', 'method', 'k', 'X_train_ratio', 'mae', 'timestamp']
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

# === Loop over k values ===
for k in k_values:
    print(f"\n=== Running experiment with k = {k} ===")
    mae = evaluate_mae(test_df, user_item_matrix, similarity_matrix, k)
    print(f"MAE for k = {k}: {mae:.4f}")
    append_result_to_csv(dataset_name, method, k, X, mae, csv_filename)







