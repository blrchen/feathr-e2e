from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
data = pd.read_csv('ratings_small.csv')

# Preprocessing
users = data['userId'].unique()
movies = data['movieId'].unique()
user_to_index = {user: idx for idx, user in enumerate(users)}
movie_to_index = {movie: idx for idx, movie in enumerate(movies)}
data['userId'] = data['userId'].apply(lambda x: user_to_index[x])
data['movieId'] = data['movieId'].apply(lambda x: movie_to_index[x])

# Split into training and validation sets
train_data = data.sample(frac=0.8, random_state=1)
val_data = data.drop(train_data.index)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RecommenderSystem(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        self.user_biases = nn.Embedding(n_users, 1)
        self.movie_biases = nn.Embedding(n_movies, 1)
        
    def forward(self, user, movie):
        user_embedding = self.user_factors(user)
        movie_embedding = self.movie_factors(movie)
        dot_product = (user_embedding * movie_embedding).sum(dim=-1)
        user_bias = self.user_biases(user).squeeze()
        movie_bias = self.movie_biases(movie).squeeze()
        return dot_product + user_bias + movie_bias

def train_model(model, train_data, val_data, epochs=10, learning_rate=0.01, weight_decay=0.001):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        count = 0
        for _, row in train_data.iterrows():
            count += 1
            if count % 10000 == 0:
                print(f"{datetime.now()} Epoch {epoch+1}/{epochs}: {count}/{len(train_data)}")
            user, movie, rating = int(row['userId']), int(row['movieId']), float(row['rating'])
            optimizer.zero_grad()
            prediction = model(torch.tensor(user).to(device), torch.tensor(movie).to(device))
            loss = loss_function(prediction, torch.tensor(rating).to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for _, row in val_data.iterrows():
                user, movie, rating = int(row['userId']), int(row['movieId']), float(row['rating'])
                prediction = model(torch.tensor(user).to(device), torch.tensor(movie).to(device))
                loss = loss_function(prediction, torch.tensor(rating).to(device))
                val_loss += loss.item()

        print(f"{datetime.now()} Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_data)}, Val Loss: {val_loss/len(val_data)}")
        torch.save(model.state_dict(), f"model_{epoch+1}.pth")

def recommend_movies(model, movie_id, n_recommendations=10):
    model.to(device)

    model.eval()
    with torch.no_grad():
        target_movie_factors = model.movie_factors(torch.tensor(movie_to_index[movie_id]))
        similarities = torch.matmul(model.movie_factors.weight, target_movie_factors)
        top_n_similar = torch.topk(similarities, n_recommendations+1)
        recommended_movies = [movies[idx] for idx in top_n_similar.indices[1:].numpy()]
    return recommended_movies

# Initialize the model
n_users = len(users)
n_movies = len(movies)
model = RecommenderSystem(n_users, n_movies)

# Train the model
print(f"{datetime.now()} Starting training...")
train_model(model, train_data, val_data)

# Recommend movies based on a specific movie
movie_id = 1  # Toy Story
recommended_movies = recommend_movies(model, movie_id)
print("Recommended movies based on Toy Story:", recommended_movies)

