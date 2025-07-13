# https://files.grouplens.org/papers/www10_sarwar.pdf 
# https://files.grouplens.org/datasets/movielens/ml-100k.zip

import pandas as pd
from numpy.linalg import norm
import numpy as np

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
user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')

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
    for idx_j in range(idx_i, n_items):  # 
        j = item_ids[idx_j]
        sim = cosine_sim(i, j, user_item_matrix)
        
        # 
        similarity_matrix.at[i, j] = sim
        similarity_matrix.at[j, i] = sim
    # 

    if idx_i % 100 == 0:
        print(f"Processed {idx_i}/{n_items} items...")
        # Processed 0/1682 items...
        # Processed 100/1682 items...
        # Processed 200/1682 items...

print(similarity_matrix.iloc[:5, :5])








