import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

print("Madhusri S-24BAD065")
# Load dataset
ratings = pd.read_csv(
    r"C:\Users\THC\Downloads\archive (21)\ratings.dat",
    sep="::",
    engine="python",
    names=['user_id','movie_id','rating','timestamp']
)

movies = pd.read_csv(
    r"C:\Users\THC\Downloads\archive (21)\movies.dat",
    sep="::",
    engine="python",
    names=['movie_id','title','genres'],
    encoding='latin-1'
)

# Create Item-User Matrix
item_user_matrix = ratings.pivot(index='movie_id', columns='user_id', values='rating')

# Fill missing values
item_user_filled = item_user_matrix.fillna(0)

# Compute item similarity
item_similarity = cosine_similarity(item_user_filled)
item_similarity_df = pd.DataFrame(item_similarity, 
                                  index=item_user_matrix.index, 
                                  columns=item_user_matrix.index)

# Function to get similar items
def get_similar_items(movie_id, n=5):
    return item_similarity_df[movie_id].sort_values(ascending=False)[1:n+1]

# Predict rating using weighted average (FIXED)
def predict_rating(user_id, movie_id):
    if movie_id not in item_user_matrix.index:
        return 0
    
    user_ratings = item_user_matrix[user_id]
    similar_items = item_similarity_df[movie_id]
    
    numerator = 0
    denominator = 0
    
    for item, rating in user_ratings.items():
        if not np.isnan(rating) and item != movie_id:
            sim = similar_items[item]
            numerator += sim * rating
            denominator += abs(sim)
    
    if denominator == 0:
        return user_ratings.mean()
    
    return numerator / denominator

# Recommend items
def recommend_items(user_id, n=5):
    user_ratings = item_user_matrix[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index
    
    predictions = {}
    for item in unrated_items:
        predictions[item] = predict_rating(user_id, item)
    
    recommended = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    
    result = [(movies[movies['movie_id']==item]['title'].values[0], round(score,2)) 
              for item, score in recommended]
    
    return result

# Get recommendations
user_id = 10
recommendations = recommend_items(user_id, n=5)

print(f"Top 5 Item-Based Recommendations for User {user_id}:")
for title, score in recommendations:
    print(f"{title} → Predicted Rating: {score}")

# Evaluation (FIXED RMSE)
y_true = []
y_pred = []

sample_data = ratings.sample(1000)

for row in sample_data.itertuples():
    pred = predict_rating(row.user_id, row.movie_id)
    if pred != 0:
        y_true.append(row.rating)
        y_pred.append(pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", round(rmse,3))

# Precision@K
def precision_at_k(user_id, k=5, threshold=3.5):
    recommended = recommend_items(user_id, n=k)
    
    relevant = 0
    for title, score in recommended:
        if score >= threshold:
            relevant += 1
    
    return relevant / k

precision = precision_at_k(user_id, k=5)
print("Precision@5:", round(precision,3))

# Visualization - Item Similarity Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(item_similarity_df.iloc[:20,:20], cmap='coolwarm')
plt.title("Item Similarity Matrix")
plt.show()

# Visualization - Top Similar Items
movie_id = 50
similar_items = get_similar_items(movie_id)

titles = [movies[movies['movie_id']==mid]['title'].values[0] for mid in similar_items.index]

plt.figure(figsize=(8,5))
plt.bar(titles, similar_items.values)
plt.xticks(rotation=45, ha='right')
plt.title(f"Top Similar Items to Movie ID {movie_id}")
plt.ylabel("Similarity Score")
plt.show()