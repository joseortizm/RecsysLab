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

# Paper version
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

# Alternative notation
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





