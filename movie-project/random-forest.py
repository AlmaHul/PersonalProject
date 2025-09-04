import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


data = pd.read_csv("movies_clean.csv")

# Skapa målvariabel
data["like"] = (data["rating"] >= 4).astype(int)

# Välj features
feature_cols = [
    "age", "gender", "occupation",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

X = data[feature_cols].copy()
y = data["like"]

# Konvertera kategoriska kolumner till numeriska
X = pd.get_dummies(X, columns=["gender", "occupation"], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bygg och träna en Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Utvärdering
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
