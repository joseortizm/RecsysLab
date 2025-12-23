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

print(ratings.head())