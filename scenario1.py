import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Create user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Fill missing values
user_item_filled = user_item_matrix.fillna(0)

# Compute similarity
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity, 
                                  index=user_item_matrix.index, 
                                  columns=user_item_matrix.index)

# Get similar users
def get_similar_users(user_id, n=5):
    return user_similarity_df[user_id].sort_values(ascending=False)[1:n+1]

# Predict rating
def predict_rating(user_id, movie_id):
    similar_users = get_similar_users(user_id)
    numerator, denominator = 0, 0
    
    for sim_user, sim_score in similar_users.items():
        rating = user_item_matrix.loc[sim_user, movie_id]
        if not np.isnan(rating):
            numerator += sim_score * rating
            denominator += sim_score

    return numerator / denominator if denominator != 0 else 0

# Recommend movies
def recommend_movies(user_id, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    
    predictions = {}
    for movie in unrated_movies:
        predictions[movie] = predict_rating(user_id, movie)

    top_n = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

    results = []
    for movie_id, score in top_n:
        title = movies[movies['movie_id']==movie_id]['title'].values
        if len(title) > 0:
            results.append((title[0], round(score,2)))

    return results

# Get recommendations
user_id = 10
top_movies = recommend_movies(user_id)

print("\nTop Recommended Movies:\n")
for title, score in top_movies:
    print(f"{title} → Predicted Rating: {score}")

# -------------------------------
# Evaluation
# -------------------------------
y_true, y_pred = [], []

for row in ratings.sample(1000, random_state=42).itertuples():
    pred = predict_rating(row.user_id, row.movie_id)
    if pred != 0:
        y_true.append(row.rating)
        y_pred.append(pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\nRMSE:", round(rmse,3))
print("MAE:", round(mae,3))

# -------------------------------
# 🔥 Visualization 1: User-Item Heatmap
# -------------------------------
plt.figure(figsize=(12,6))
sns.heatmap(user_item_filled.iloc[:50,:50], cmap='coolwarm')
plt.title("User-Item Matrix Heatmap")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

# -------------------------------
# 🔥 Visualization 2: Similarity Matrix
# -------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(user_similarity_df.iloc[:20,:20], cmap='viridis')
plt.title("User Similarity Matrix")
plt.xlabel("Users")
plt.ylabel("Users")
plt.show()

# -------------------------------
# 🔥 Visualization 3: Top Recommended Movies (Bar Chart)
# -------------------------------
titles = [movie[0] for movie in top_movies]
scores = [movie[1] for movie in top_movies]

plt.figure(figsize=(10,5))
plt.barh(titles, scores)
plt.xlabel("Predicted Rating")
plt.title(f"Top Recommended Movies for User {user_id}")
plt.gca().invert_yaxis()   # highest at top
plt.show()