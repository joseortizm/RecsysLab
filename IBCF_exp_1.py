from IBCF_recsys import build_user_item_matrix, split_ratings_by_user, build_item_similarity_matrix, evaluate_mae, save_similarity_matrix
import pandas as pd
import csv
from datetime import datetime
import os 

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

# === Setup ===
csv_filename = 'results_item_based_sensitivity_X.csv'
test_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
k = 30
dataset_name = 'MovieLens100k'
method = 'item-based'

# === Load dataset ===
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv(
    '../datasets/MovieLens/ml-100k/u.data', 
    sep='\t', 
    names=column_names,
    encoding='latin-1'
)

# ===  Parameter x  ===
for test_ratio in test_ratios:
    test_ratio = round(test_ratio, 2)
    X = round(1 - test_ratio, 2)
    print(f"\n=== Running experiment with Test Ratio = {test_ratio} (i.e. Train Ratio X = {X}) ===")
    
    train_df, test_df = split_ratings_by_user(ratings, test_ratio=test_ratio)
    user_item_matrix = build_user_item_matrix(train_df)
    similarity_matrix = build_item_similarity_matrix(user_item_matrix)
    # We save the similarity matrix as a .pkl file to avoid recalculating it in future runs.
    # This is useful because computing it is expensive and it remains unchanged if we use the same training set.
    save_similarity_matrix(similarity_matrix, path=f"item_similarity_matrix_X{X}.pkl")
    mae = evaluate_mae(test_df, user_item_matrix, similarity_matrix, k)

    print(f"MAE for X = {X:.2f}, k = {k}: {mae:.4f}")
    append_result_to_csv(dataset_name, method, k, X, mae, csv_filename)










