#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.DataFrame({
    'user_id': ['A', 'A', 'B', 'B', 'B', 'C', 'C'],
    'product_id': ['P1', 'P2', 'P2', 'P3', 'P4', 'P1', 'P3']
})

user_item_matrix = df.pivot_table(index='user_id', columns='product_id', aggfunc=lambda x: 1, fill_value=0)

cosine_sim = cosine_similarity(user_item_matrix.T)

cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print(cosine_sim_df.shape)
print(cosine_sim_df.head())

print(user_item_matrix.shape)
print(user_item_matrix.head())

print(set(df['product_id']) == set(user_item_matrix.columns))

print(user_item_matrix.isnull().sum().sum() == 0)
print(user_item_matrix.dtypes.unique() == np.dtype(int))

def recommend_products(user_id):
    user_products = df[df['user_id'] == user_id]['product_id'].unique()
    if not set(user_products).issubset(user_item_matrix.columns):
        return []
    sim_scores = cosine_sim_df[user_products].sum(axis=1).sort_values(ascending=False)
    sim_scores = sim_scores[~np.isin(sim_scores.index, user_products)]
    recommendations = list(sim_scores.index[:3])
    return recommendations

# Test the model
print(recommend_products('A'))


# In[40]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.DataFrame({
    'user_id': ['A', 'A', 'B', 'B', 'B', 'C', 'C'],
    'product_id': ['P1', 'P2', 'P2', 'P3', 'P4', 'P1', 'P3']
})
print("LISTED THE USER ID AND PRODUCTS ABOVE")


# In[28]:


# Construct user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', aggfunc=lambda x: 1, fill_value=0)


# In[29]:


# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(user_item_matrix.T)


# In[30]:


# Convert cosine similarity matrix to DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)


# In[45]:


# Test if cosine similarity matrix is correct
print("DATASET MATRIX IS AS FOLLOWS:")

print(cosine_sim_df.shape)
print(cosine_sim_df.head())


# In[46]:


# Test if user-item matrix is correct
print("NUMBER OF TIMES USER HAS ACCESSED/SEARCHED FOR THE PRODUCTS")
print(user_item_matrix.shape)
print(user_item_matrix.head())


# In[34]:


# Check if product IDs match between df and user_item_matrix
print(set(df['product_id']) == set(user_item_matrix.columns))


# In[35]:


# Check if there are any missing values or incorrect data types in user_item_matrix
print(user_item_matrix.isnull().sum().sum() == 0)
print(user_item_matrix.dtypes.unique() == np.dtype(int))


# In[36]:


# Define function to recommend products
def recommend_products(user_id):
    user_products = df[df['user_id'] == user_id]['product_id'].unique()
    if not set(user_products).issubset(user_item_matrix.columns):
        return []
    sim_scores = cosine_sim_df[user_products].sum(axis=1).sort_values(ascending=False)
    sim_scores = sim_scores[~np.isin(sim_scores.index, user_products)]
    recommendations = list(sim_scores.index[:3])
    return recommendations


# In[37]:


# Test the model
print(recommend_products('A'))

