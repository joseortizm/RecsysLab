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
        - r_{u,i} is the rating user u gave to item i
        - U_{ij} is the set of users who rated both items i and j
    """

    # Extract the ratings for items i and j across all users
    ratings_i = matrix[i]
    ratings_j = matrix[j]
    
    # Find users who rated both items
    common = ratings_i.notna() & ratings_j.notna()
    
    # If no users rated both, similarity is 0
    if common.sum() == 0:
        return 0.0
    
    # Filter only the common ratings
    vec_i = ratings_i[common]
    vec_j = ratings_j[common]
    
    # Compute cosine similarity: dot product divided by product of norms
    numerator = np.dot(vec_i, vec_j)
    denominator = norm(vec_i) * norm(vec_j)
    
    return numerator / denominator if denominator != 0 else 0.0


#sim_50_181 = cosine_sim(50, 181, user_item_matrix)
#print(f"Similarity between item 50 and 181 is {sim_50_181:.4f}")

# TODO: Figure 2: Items and Similarity computation









