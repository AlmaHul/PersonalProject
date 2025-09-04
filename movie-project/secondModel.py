import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# Konfig
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_FPATH = "text_embeddings.pt"
GENRE_TENSOR_FPATH = "X_genre_tensor.pt"
MODEL_FPATH = "text_genre_model.pt"
CLEAN_CSV = "movies_clean.csv"
INDEX_CSV = "movies_with_index.csv"

# Sätt detta till True om du modellen ska tränas vid import
TRAIN_ON_IMPORT = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hjälp-funktioner
def extract_year_from_title(title: str):
    if not isinstance(title, str):
        return None
    m = re.search(r"\((\d{4})\)", title)
    return int(m.group(1)) if m else None

# Svensk -> dataset-genre-nycklar
GENRE_COLS = [
    "Action","Adventure","Animation","Children's","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
]
SV_GENRE_MAP = {
    "action": "Action",
    "äventyr": "Adventure",
    "animation": "Animation",
    "barn": "Children's",
    "komedi": "Comedy",
    "crime": "Crime",
    "dokumentär": "Documentary",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "film-noir": "Film-Noir",
    "skräck": "Horror",
    "musikal": "Musical",
    "mysterium": "Mystery",
    "romantisk": "Romance",
    "romantik": "Romance",
    "sci-fi": "Sci-Fi",
    "thriller": "Thriller",
    "krig": "War",
    "western": "Western",
}

# Målgruppskolumner
AUDIENCE_COLS = ["Teen", "Adult", "Family", "Women"]
SV_AUDIENCE_HINTS = {
    "Teen": ["tonår", "ungdom", "ung", "tonåringar", "ungdomar"],
    "Women": ["kvinnor", "tjejer", "unga kvinnor"],
    "Family": ["familj", "barnvänlig", "hela familjen"],
    "Adult": ["vuxna"],
}


# Ladda data
data = pd.read_csv("movies_clean.csv")

# säkerställ rating & like
if "rating" not in data.columns:
    raise ValueError("CSV måste innehålla kolumnen 'rating'.")
data["like"] = (data["rating"] >= 4).astype(int)

# year: extrahera från titel
if "year" not in data.columns:
    data["year"] = data["title"].apply(extract_year_from_title)

# textkälla för embeddings
text_col = "description" if "description" in data.columns else "title"
texts = data[text_col].fillna(data["title"]).astype(str).tolist()


# Skapa/Ladda text-embeddings
if os.path.exists(EMBEDDINGS_FPATH):
    text_embeddings = torch.load(EMBEDDINGS_FPATH).to(torch.float32)
else:
    sbert = SentenceTransformer(EMBEDDING_MODEL, device=device)
    batch, embs = 64, []
    for i in range(0, len(texts), batch):
        embs.append(sbert.encode(texts[i:i+batch], convert_to_tensor=True, show_progress_bar=True))
    text_embeddings = torch.cat(embs, dim=0).to(torch.float32)
    torch.save(text_embeddings.cpu(), EMBEDDINGS_FPATH)
text_embeddings = text_embeddings.to(device)


# Genre-features
missing_genres = [g for g in GENRE_COLS if g not in data.columns]
if missing_genres:
    # Om genrekolumner saknas, fyll med nollor
    for g in missing_genres:
        data[g] = 0

X_genre = data[GENRE_COLS].values.astype("float32")
scaler = StandardScaler()
X_genre = scaler.fit_transform(X_genre)
X_genre_tensor = torch.tensor(X_genre, dtype=torch.float32).to(device)
torch.save(X_genre_tensor.cpu(), GENRE_TENSOR_FPATH)


# Målgrupp-feature

have_audience = "target_audience" in data.columns
if have_audience:
    # Gör one-hot på kända kategorier
    onehot = pd.get_dummies(data["target_audience"])
    for col in AUDIENCE_COLS:
        if col not in onehot.columns:
            onehot[col] = 0
    X_audience = onehot[AUDIENCE_COLS].values.astype("float32")
    X_audience_tensor = torch.tensor(X_audience, dtype=torch.float32).to(device)
else:
    X_audience_tensor = None


# Bygg NN-träningsmatris
if have_audience:
    X_tensor = torch.cat([text_embeddings, X_genre_tensor, X_audience_tensor], dim=1)
else:
    X_tensor = torch.cat([text_embeddings, X_genre_tensor], dim=1)

y_tensor = torch.tensor(data["like"].values, dtype=torch.float32).view(-1, 1).to(device)


# NN-arkitektur
class TextGenreNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)


# Träning
def train_and_save(epochs=20, lr=1e-3):
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor.cpu()
    )
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    model = TextGenreNN(X_train.shape[1]).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {ep+1}/{epochs} - Loss: {loss_sum/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_FPATH)
    data.to_csv(INDEX_CSV, index=False)
    return model

if TRAIN_ON_IMPORT:
    _ = train_and_save()



input_dim = X_tensor.shape[1]
model_nn = TextGenreNN(input_dim).to(device)
if os.path.exists(MODEL_FPATH):
    model_nn.load_state_dict(torch.load(MODEL_FPATH, map_location=device))
    model_nn.eval()
else:
    print("Varning: Hittar inte tränad modell. NN-prediktioner kommer vara noll.")
    model_nn.eval()


# Förbered unika filmer (undvik dubbletter)
movies_df = data.copy()
unique_movies_df = movies_df.drop_duplicates(subset="title").reset_index(drop=True)
unique_indices = unique_movies_df.index.values
unique_embeddings = text_embeddings[unique_indices]  # matcha embeddings mot unika rader


# Rekommendationsfunktion
_sentence_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

def _genre_score_from_text(user_text_lower: str, df: pd.DataFrame):
    score = np.zeros(len(df))
    # direkt genreord enligt dataset-kolumner
    for sv_word, eng_col in SV_GENRE_MAP.items():
        if sv_word in user_text_lower and eng_col in df.columns:
            score += df[eng_col].values
    if score.max() > 0:
        score = score / score.max()
    return score

def _audience_score_from_text(user_text_lower: str, df: pd.DataFrame):
    if not have_audience:
        return np.zeros(len(df))
    score = np.zeros(len(df))
    for aud_col, hints in SV_AUDIENCE_HINTS.items():
        if any(h in user_text_lower for h in hints) and aud_col in df.columns:
            score += df[aud_col].values
    if score.max() > 0:
        score = score / score.max()
    return score

def _location_score(user_text_lower: str, df: pd.DataFrame):
    score = np.zeros(len(df))
    if "location" in df.columns:
        loc_series = df["location"].fillna("").str.lower()
        user_words = re.findall(r"[a-ö0-9]+", user_text_lower)
        for i, loc in enumerate(loc_series):
            for w in user_words:
                if len(w) >= 3 and w in loc:
                    score[i] += 1
        if score.max() > 0:
            score = score / score.max()
    else:
        # försök matcha platsord mot description/title text
        text_series = (df["description"] if "description" in df.columns else df["title"]).fillna("").str.lower()
        if any(k in user_text_lower for k in ["new york", "london", "paris", "tokyo", "los angeles", "la", "stockholm"]):
            target_words = [w for w in ["new york","london","paris","tokyo","los angeles","stockholm","la"] if w in user_text_lower]
            for i, t in enumerate(text_series):
                for w in target_words:
                    if w in t:
                        score[i] += 1
            if score.max() > 0:
                score = score / score.max()
    return score

def _year_score(user_text_lower: str, df: pd.DataFrame):
    score = np.zeros(len(df))
    if "year" in df.columns:
        years = df["year"].fillna(2000).astype(int).values
        # exakt årtal i texten
        matches = re.findall(r"\b(19|20)\d{2}\b", user_text_lower)
        if matches:
            # ta första kompletta årtalet
            y_match = re.search(r"\b((?:19|20)\d{2})\b", user_text_lower)
            if y_match:
                y_target = int(y_match.group(1))
                score = np.array([max(0.0, 1.0 - abs(int(y) - y_target) / 30.0) for y in years])
        else:
            # period-heuristik
            if "80-tal" in user_text_lower:
                score = np.array([1.0 if 1980 <= y < 1990 else 0.0 for y in years])
            elif "90-tal" in user_text_lower:
                score = np.array([1.0 if 1990 <= y < 2000 else 0.0 for y in years])
            elif "00-tal" in user_text_lower or "2000-talets början" in user_text_lower:
                score = np.array([1.0 if 2000 <= y < 2010 else 0.0 for y in years])
            elif "10-tal" in user_text_lower:
                score = np.array([1.0 if 2010 <= y < 2020 else 0.0 for y in years])
        if score.max() > 0:
            score = score / score.max()
    return score

def recommend_movies(user_text: str, top_n: int = 5) -> pd.DataFrame:
    """
    Returnerar en DataFrame med topp-N filmer baserat på:
    - plot/description
    - genrer
    - målgrupp
    - plats
    - år
    - rating
    """
    user_text_lower = user_text.lower().strip()

    # semantisk likhet mot plot/description
    user_emb = _sentence_model.encode([user_text], convert_to_tensor=True).cpu()
    sims = cosine_similarity(user_emb, unique_embeddings.cpu())[0]  # shape: (num_unique,)

    # signaler från texten
    genre_score = _genre_score_from_text(user_text_lower, unique_movies_df)
    audience_score = _audience_score_from_text(user_text_lower, unique_movies_df)
    location_score = _location_score(user_text_lower, unique_movies_df)
    year_score = _year_score(user_text_lower, unique_movies_df)

    # NN-prediktion (”gilla”-sannolikhet)
    if os.path.exists(MODEL_FPATH):
        if have_audience:
            unique_X = torch.cat(
                [
                    unique_embeddings,
                    torch.tensor(unique_movies_df[GENRE_COLS].values, dtype=torch.float32),
                    torch.tensor(
                        pd.get_dummies(unique_movies_df.get("target_audience", pd.Series([""]*len(unique_movies_df)))).reindex(columns=AUDIENCE_COLS, fill_value=0).values,
                        dtype=torch.float32
                    ),
                ],
                dim=1
            ).to(device)
        else:
            unique_X = torch.cat(
                [
                    unique_embeddings,
                    torch.tensor(unique_movies_df[GENRE_COLS].values, dtype=torch.float32),
                ],
                dim=1
            ).to(device)

        with torch.no_grad():
            pred_logits = model_nn(unique_X)
            pred_prob = torch.sigmoid(pred_logits).cpu().numpy().flatten()
    else:
        pred_prob = np.zeros(len(unique_movies_df), dtype=np.float32)

    # rating – endast om användaren ber om ”bra”, ”bästa”, ”top”
    use_rating = any(key in user_text_lower for key in ["bra film", "bra filmer", "bästa", "top", "hög rating", "höga betyg"])
    rating_norm = unique_movies_df["rating"].values / max(1.0, unique_movies_df["rating"].max())

    # kombinera
    combined = (
        sims * 0.40 +
        pred_prob * 0.25 +
        genre_score * 0.15 +
        audience_score * 0.10 +
        location_score * 0.05 +
        year_score * 0.05
    )

    if use_rating:
        combined = combined * 0.7 + rating_norm * 0.3  # injicera rating bara när efterfrågat

    # topp-N
    top_idx = combined.argsort()[-top_n:][::-1]

    # kolumner ut
    out_cols = ["title", "rating"]
    for c in ["year", "location"]:
        if c in unique_movies_df.columns:
            out_cols.append(c)
    out_cols += [g for g in GENRE_COLS if g in unique_movies_df.columns]
    if have_audience:
        for a in AUDIENCE_COLS:
            if a in unique_movies_df.columns:
                out_cols.append(a)

    return unique_movies_df.iloc[top_idx][out_cols].reset_index(drop=True)

# Testa
if __name__ == "__main__":
    user_input = "Jag vill se en skräckfilm"
    recommendations = recommend_movies(user_input, top_n=5)
    print("\n🎬 Top 5 rekommenderade filmer:")
    print(recommendations)
