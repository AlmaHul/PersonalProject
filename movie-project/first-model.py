import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

# Läs in datan
data = pd.read_csv("movies_clean.csv")

# Målvariabel (gilla = 1, annars 0)
data["like"] = (data["rating"] >= 4).astype(int)

# Features
feature_cols = [
    "age", "gender", "occupation",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
X = data[feature_cols].copy()
y = data["like"]

# Koda kategoriska variabler
X = pd.get_dummies(X, columns=["gender", "occupation"], drop_first=True)


X = X.astype("float32")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Konvertera till PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Bygg ett enkelt nätverk
class MovieNN(nn.Module):
    def __init__(self, input_dim):
        super(MovieNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# Initiera modellen
input_dim = X_train.shape[1]
model = MovieNN(input_dim)

# Loss & optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Träna modellen
epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# Utvärdering
model.eval()



model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = (y_pred >= 0.5).float()

# Konvertera till numpy för sklearn
y_true = y_test_tensor.numpy()
y_pred_class_np = y_pred_class.numpy()

# Accuracy + rapport
acc = (y_pred_class.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
print(f"\n✅ Test Accuracy: {acc:.2f}\n")
print("📊 Klassificeringsrapport:")
print(classification_report(y_true, y_pred_class_np, digits=2))


# Räkna antal "gilla" och "inte gilla"
print(y.value_counts(normalize=False))
print("\nAndel i procent:")
print(y.value_counts(normalize=True) * 100)

