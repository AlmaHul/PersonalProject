from flask import Flask, request, jsonify
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
from secondModel import recommend_movies


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/test")
def index():
    return "Filmrekommendations-API körs ✅"

@app.route("/", methods=["POST"])
def recommend():
    payload = request.get_json(force=True) or {}
    user_text = payload.get("text", "")
    top_n = int(payload.get("top_n", 5))
    df = recommend_movies(user_text, top_n=top_n)
    return jsonify({"recommendations": df.to_dict(orient="records")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
