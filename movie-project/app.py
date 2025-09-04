from flask import Flask, request, jsonify
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Ladda modeller & data ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

movies_df = pd.read_csv("movies_with_index.csv")
text_embeddings = torch.load("text_embeddings.pt").to(torch.float32).cpu()

# unika filmer (så vi slipper dubbletter)
unique_movies_df = movies_df.drop_duplicates(subset="title").reset_index(drop=True)
unique_indices = unique_movies_df.index.values
unique_embeddings = text_embeddings[unique_indices]

sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# === Flask app ===
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_text = data.get("text", "")
    top_n = int(data.get("top_n", 5))
    genre = data.get("genre", None)

    # filtrera på genre om angivet
    filtered_df = unique_movies_df.copy()
    filtered_indices = filtered_df.index.values
    if genre and genre in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[genre] == 1]
        filtered_indices = filtered_df.index.values

    if filtered_df.empty:
        return jsonify({"recommendations": []})

    # skapa embedding för användarens input
    user_emb = sentence_model.encode([user_text], convert_to_tensor=True).cpu()

    # cosine similarity
    sims = cosine_similarity(user_emb, unique_embeddings[filtered_indices].cpu())[0]
    top_idx_local = sims.argsort()[-top_n:][::-1]

    # hämta filmer
    top_movies = filtered_df.iloc[top_idx_local]

    return jsonify({"recommendations": top_movies["title"].tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
