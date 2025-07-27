# https://files.grouplens.org/papers/www10_sarwar.pdf 
# https://files.grouplens.org/datasets/movielens/ml-100k.zip

import pandas as pd
from numpy.linalg import norm
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from datetime import datetime
from tqdm import tqdm
import time

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

# file u.data: this is a tab separated list of user id | item id | rating | timestamp.
ratings = pd.read_csv(
    '../datasets/MovieLens/ml-100k/u.data', 
    sep='\t', 
    names=column_names,
    encoding='latin-1'
)

#print(ratings.head())

# create user_item matrix 
#user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index='user_id', columns='item_id', values='rating')


#print("Shape: ", user_item_matrix.shape) # Shape: (943, 1682)
#print(user_item_matrix.head())

# 3.1 Item Similarity Computation

# cosine similarity only with users who rated both items.
## 3.1.1 Cosine-based Similarity
def cosine_sim(i: int, j: int, matrix: pd.DataFrame) -> float:
    """
    Compute the cosine similarity between item i and item j
    using only users who rated both items.

    Mathematically, this is based on the following formula:

        sim(i, j) = (vec_i ⋅ vec_j) / (||vec_i|| * ||vec_j||)

    In expanded form (as in the original paper by Sarwar et al., 2001):

        sim(i, j) = sum_{u ∈ U_{ij}} r_{u,i} * r_{u,j} /
                    (sqrt(sum_{u ∈ U_{ij}} r_{u,i}^2) * sqrt(sum_{u ∈ U_{ij}} r_{u,j}^2))

    where:
        - vec_i is the column vector of ratings for item i, that is, all the ratings given by different users to item i
        - vec_j is the column vector of ratings for item j, that is, all the ratings given by different users to item j
        - r_{u,i} is the rating user u gave to item i
        - U_{ij} is the set of users who rated both items i and j
    """

    # Extract the ratings for items i and j across all users
    ratings_i = matrix[i]
    ratings_j = matrix[j]
    
    # Find users who rated both items
    common = ratings_i.notna() & ratings_j.notna()
    # u1 -> True
    # u2 -> False
    # ... -> ...
    # u4 -> True
    # ... -> ... 
    
    # If no users rated both, similarity is 0
    if common.sum() == 0:
        return 0.0
    
    # Filter only the common ratings (pandas operation)
    vec_i = ratings_i[common]
    # u1 -> 5.0
    # u4 -> 3.0
    # ... -> ..
    vec_j = ratings_j[common]
    
    # Compute cosine similarity: dot product divided by product of norms
    numerator = np.dot(vec_i, vec_j)
    denominator = norm(vec_i) * norm(vec_j)
    
    return numerator / denominator if denominator != 0 else 0.0


#sim_50_181 = cosine_sim(50, 181, user_item_matrix)
#print(f"Similarity between item 50 and 181 is {sim_50_181:.4f}")

# Figure 2: Items and Similarity computation
'''
item_ids = user_item_matrix.columns
n_items = len(item_ids)

# init matrix
similarity_matrix = pd.DataFrame(
    data=np.zeros((n_items, n_items)),
    index=item_ids,
    columns=item_ids
)

# Similarity computation for (i, j)
for idx_i, i in enumerate(item_ids):
    # (symmetry) you will compare item i with items j where j ≥ i
    for idx_j in range(idx_i, n_items):
        j = item_ids[idx_j]
        sim = cosine_sim(i, j, user_item_matrix)
        
        # store the similarity in two cells of the matrix
        similarity_matrix.at[i, j] = sim
        similarity_matrix.at[j, i] = sim

    if idx_i % 100 == 0:
        print(f"Processed {idx_i}/{n_items} items...")
        # Processed 0/1682 items...
        # Processed 100/1682 items...
        # Processed 200/1682 items...

#print(similarity_matrix.iloc[:5, :5])
'''

def build_item_similarity_matrix(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    item_ids = user_item_matrix.columns
    n_items = len(item_ids)

    similarity_matrix = pd.DataFrame(
        data=np.zeros((n_items, n_items)),
        index=item_ids,
        columns=item_ids
    )

    start_time = time.time()
    for idx_i, i in enumerate(item_ids):
        for idx_j in range(idx_i, n_items):  # Only j ≥ i due to symmetry
            j = item_ids[idx_j]
            sim = cosine_sim(i, j, user_item_matrix)
            similarity_matrix.at[i, j] = sim
            similarity_matrix.at[j, i] = sim

        if idx_i % 100 == 0:
            elapsed = time.time() - start_time
            #print(f"Processed {idx_i}/{n_items} items...")
            print(f"[{idx_i}/{n_items}] Elapsed time: {elapsed:.2f} seconds")
    
    return similarity_matrix

# Extra:

def save_similarity_matrix(matrix: pd.DataFrame, path: str = "item_similarity_matrix.pkl") -> None:
    """
    Save the similarity matrix to a pickle file.
    
    Args:
        matrix (pd.DataFrame): The similarity matrix to save.
        path (str): File path to save the matrix (default: item_similarity_matrix.pkl).
    """
    matrix.to_pickle(path)
    print(f"Similarity matrix saved to {path}")


def load_similarity_matrix(path: str = "item_similarity_matrix.pkl") -> pd.DataFrame:
    """
    Load a similarity matrix from a pickle file.
    
    Args:
        path (str): File path to load the matrix from (default: item_similarity_matrix.pkl).
        
    Returns:
        pd.DataFrame: The loaded similarity matrix.
    """
    matrix = pd.read_pickle(path)
    print(f"Similarity matrix loaded from {path}")
    return matrix

# 3.2 Prediction Computation

# The most important step in a collaborative filtering system is to generate the output interface in terms of prediction.
# Once we isolate the set of most similar items based on the similarity measures
def get_top_k_similar_items(item_id: int, k: int, sim_matrix: pd.DataFrame) -> pd.Series:
    """
    Return the top‑k most similar items to `item_id`
    (excluding itself), sorted by similarity descending.
    """
    return (sim_matrix[item_id]
            .drop(labels=[item_id])   # remove self‑similarity
            .sort_values(ascending=False)
            .head(k))

# the next step is to look into the target users ratings and use a technique to obtain predictions.
# Here were consider two such techniques: 3.2.1 Weighted Sum and 3.2.2 Regression

# Selected method: Weighted Sum 

# Weighted Sum (paper version):
"""
P_{u,i} \;=\;
\frac{\displaystyle\sum_{\text{all similar items } N}
        S_{i,N}\, R_{u,N}}
     {\displaystyle\sum_{\text{all similar items } N}
        \lvert S_{i,N}\rvert}

where:
* \(\hat{r}_{u,i}\)  es la predicción del usuario \(u\) sobre el ítem \(i\).
* \(N(i)\)  son los \(k\) ítems más similares a \(i\) que el usuario \(u\) ha calificado.
* \(\operatorname{sim}(i,j)\)  es la similitud ítem‑ítem.
* \(r_{u,j}\)  es la calificación real del usuario \(u\) al ítem \(j\).

"""

# Weighted Sum (alternative notation):
"""
\hat{r}_{u,i} \;=\;
\frac{\displaystyle\sum_{j \in N(i)}
        \operatorname{sim}(i,j)\; r_{u,j}}
     {\displaystyle\sum_{j \in N(i)}
        \bigl|\operatorname{sim}(i,j)\bigr|}

where:
* \(P_{u,i}\)  ≡ \(\hat{r}_{u,i}\).
* \(S_{i,N}\)  es la similitud entre el ítem \(i\) y su vecino \(N\).
* \(R_{u,N}\)  es la calificación del usuario \(u\) al vecino \(N\).
* La suma se toma solo sobre los ítems \(N\) que son vecinos de \(i\)
  **y** que han sido calificados por \(u\).
"""

def predict_rating(user_id: int, item_id: int, ratings_matrix: pd.DataFrame, sim_matrix: pd.DataFrame, k: int = 30) -> float:
    """
    Predict rating that `user_id` would give to `item_id`
    using the weighted‑sum item‑based CF formula.
    """
    # Items already rated by this user
    user_ratings = ratings_matrix.loc[user_id].dropna()

    # The original paper (Sarwar et al., 2001) assumes that every user has rated
    # and “We only considered users that had rated 20 or more movies…”
    
    # As such, the prediction formula in Section 3.2 does not handle the case
    # where a user has no ratings at all (cold-start user).
    # In real-world implementations, however, we may encounter users without any
    # historical data. In those cases**, we can fall back to the global average rating,
    # the item’s average, or some default value, depending on the strategy chosen.

    # If user has no ratings we can’t predict -> return global mean
    # TODO: check if every user has rated in dataset
    if user_ratings.empty:
        return ratings_matrix.stack().mean()

   
    neighbours = get_top_k_similar_items(item_id, k, sim_matrix)
    
    # Neighbour items the user has rated
    neighbours = neighbours[neighbours.index.isin(user_ratings.index)]

    # **If no overlap, again fall back to user’s mean or global mean
    if neighbours.empty:
        return user_ratings.mean()

    # 3.2.1 Weighted‑sum
    numer = (neighbours * user_ratings[neighbours.index]).sum()
    denom = neighbours.abs().sum()

    return numer / denom if denom != 0 else user_ratings.mean()

# Experimental evaluation

## Data set

# ...(we only considered users that had rated 20 or more movies). We divided the database into a training set and test set. 
# A value of X= 0.8 would indicate 80% of the data was used as training set and 20% of the data was used as test set. 

def split_ratings_by_user(df: pd.DataFrame, test_ratio=0.2, min_ratings=20):
    train_list = []
    test_list = []

    for user_id, user_data in df.groupby('user_id'):
        if len(user_data) < min_ratings:
            continue  # omitimos usuarios con pocos datos

        train_u, test_u = train_test_split(
            user_data,
            test_size=test_ratio,
            random_state=42
        )

        train_list.append(train_u)
        test_list.append(test_u)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df

# ----------------------------------------------------------
# Sparsity level of the user-item rating matrix
#
#   Formula (from Sarwar et al., 2001):
#
#       sparsity = 1 - (nonzero_entries / total_entries)
#
#   Where:
#       - nonzero_entries: number of actual ratings in the dataset
#       - total_entries  : total number of possible ratings
#                        = number_of_users × number_of_items
#
#   Example for MovieLens 100k:
#       - Number of users  = 943
#       - Number of items  = 1682
#       - Number of ratings = 100,000
#
#       total_entries = 943 × 1682 = 1,586,126
#       sparsity = 1 - (100,000 / 1,586,126) ≈ 0.9369
#
#   → Interpretation:
#     About 93.69% of the user-item matrix is empty (no rating).
# ----------------------------------------------------------

def evaluate_mae(test_df: pd.DataFrame, ratings_matrix: pd.DataFrame, sim_matrix: pd.DataFrame, k: int = 30) -> float:
    """ 
    MAE = (1 / N) * sum_{i=1}^{N} | p_i - q_i |

     where:
       N     = total number of predictions
       p_i   = predicted rating for test point i
       q_i   = actual rating for test point i
    """

    errors = []

    for _, row in test_df.iterrows():
        user = row['user_id']
        item = row['item_id']
        true_rating = row['rating']

        try:
            pred_rating = predict_rating(user, item, ratings_matrix, sim_matrix, k)
            errors.append(abs(pred_rating - true_rating))
        except:
            continue 

    return sum(errors) / len(errors)


# Step 1: Split dataset by user (80% train, 20% test)
train_df, test_df = split_ratings_by_user(ratings, test_ratio=0.2)

# Step 2: Build the user-item matrix from training data
user_item_matrix = build_user_item_matrix(train_df)
print("Total number of items in the original ratings DataFrame:", ratings['item_id'].nunique()) #1682
print("Number of items with at least one rating (in user-item matrix):", user_item_matrix.shape[1]) # 1650

# Step 3: Compute the item-item similarity matrix
similarity_matrix = build_item_similarity_matrix(user_item_matrix)
save_similarity_matrix(similarity_matrix)

# Step 4: Evaluate prediction accuracy using MAE
k = 30
X_train_ratio = 0.8  # Because test_ratio = 0.2
dataset_name = "MovieLens100k"

mae = evaluate_mae(test_df, user_item_matrix, similarity_matrix, k=k)
print(f"Mean Absolute Error (k=30): {mae:.4f}")

# Step 5: Save the result in a CSV file
with open("results.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    
    # Write header if the file is empty
    if file.tell() == 0:
        writer.writerow(["dataset", "method", "k", "X_train_ratio", "mae", "timestamp"])
    
    writer.writerow([
        dataset_name,
        "item-based",
        k,
        X_train_ratio,
        round(mae, 4),
        datetime.now().isoformat()
    ])












