import pandas as pd  
import numpy as np  
import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
  
# Define dataset class  
class MovieLensDataset(Dataset):  
    def __init__(self, df):  
        self.user_ids = df['userId'].values  
        self.movie_ids = df['movieId'].values  
        self.ratings = df['rating'].values  
          
    def __len__(self):  
        return len(self.user_ids)  
      
    def __getitem__(self, idx):  
        return (self.user_ids[idx], self.movie_ids[idx], self.ratings[idx])  
      
# Define model class  
class Recommender(nn.Module):  
    def __init__(self, n_users, n_movies, emb_size=50, hidden_size=100):  
        super().__init__()  
        self.user_emb = nn.Embedding(n_users, emb_size)  
        self.movie_emb = nn.Embedding(n_movies, emb_size)  
        self.fc1 = nn.Linear(2*emb_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, 1)  
          
    def forward(self, user_ids, movie_ids):  
        user_vec = self.user_emb(user_ids)  
        movie_vec = self.movie_emb(movie_ids)  
        x = torch.cat([user_vec, movie_vec], dim=1)  
        x = nn.functional.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x.view(-1)  
      
# Define training function  
def train(model, dataloader, optimizer, criterion, device):  
    model.train()  
    train_loss = 0.0  
    for user_ids, movie_ids, ratings in dataloader:  
        user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)  
        optimizer.zero_grad()  
        outputs = model(user_ids, movie_ids)  
        loss = criterion(outputs, ratings)  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item() * len(user_ids)  
    return train_loss / len(dataloader.dataset)  
  
# Define validation function  
def validate(model, dataloader, criterion, device):  
    model.eval()  
    valid_loss = 0.0  
    with torch.no_grad():  
        for user_ids, movie_ids, ratings in dataloader:  
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)  
            outputs = model(user_ids, movie_ids)  
            loss = criterion(outputs, ratings)  
            valid_loss += loss.item() * len(user_ids)  
    return valid_loss / len(dataloader.dataset)  
  
# Save the trained model to a file  
def save_model(model, file_path):  
    torch.save(model.state_dict(), file_path)  
  
# Load the trained model from a file  
def load_model(model, file_path):  
    model.load_state_dict(torch.load(file_path))  
  
# Load CSV data  
data = pd.read_csv('ratings.csv')  
  
# Map user and movie IDs to contiguous integers  
user_ids = data['userId'].unique().tolist()  
user2idx = {o:i for i,o in enumerate(user_ids)}  
data['userId'] = data['userId'].apply(lambda x: user2idx[x])  
  
movie_ids = data['movieId'].unique().tolist()  
movie2idx = {o:i for i,o in enumerate(movie_ids)}  
data['movieId'] = data['movieId'].apply(lambda x: movie2idx[x])  
  
# Split data into train and validation sets  
msk = np.random.rand(len(data)) < 0.8  
train_data = data[msk]  
valid_data = data[~msk]  
  
# Set up device  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
# Set up dataset and dataloader  
train_dataset = MovieLensDataset(train_data)  
valid_dataset = MovieLensDataset(valid_data)  
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  
valid_dataloader = DataLoader(valid_dataset, batch_size=64)  
  
# Set up model, optimizer, and loss function  
model = Recommender(len(user_ids), len(movie_ids)).to(device)  
optimizer = torch.optim.Adam(model.parameters())  
criterion = nn.MSELoss()  
  
# Train model  
n_epochs = 20  
for epoch in range(n_epochs):  
    train_loss = train(model, train_dataloader, optimizer, criterion, device)  
    valid_loss = validate(model, valid_dataloader, criterion, device)  
    print(f'Epoch {epoch+1}, train_loss = {train_loss:.4f}, valid_loss = {valid_loss:.4f}')  

save_model(model, "recommendation_model.pth")  
