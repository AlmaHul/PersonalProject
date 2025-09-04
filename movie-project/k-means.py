import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = pd.read_csv("movies_clean.csv")

genre_cols = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
X = data[genre_cols]

# K-Means clustering
kmeans = KMeans(n_clusters=8, random_state=42)
data["cluster"] = kmeans.fit_predict(X)

# PCA för 2D-visualisering
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
data["pca1"] = reduced[:, 0]
data["pca2"] = reduced[:, 1]

# Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data["pca1"], data["pca2"], c=data["cluster"], cmap="tab10", alpha=0.6)
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Filmkluster baserat på genrer (K-Means + PCA)")
plt.show()

# Visa exempel på filmer per kluster
for cluster_id in range(3):
    print(f"\n🎬 Kluster {cluster_id} - exempel på filmer:")
    print(data[data["cluster"] == cluster_id]["title"].head(10).to_string(index=False))



# Summera genrer per kluster
cluster_summary = data.groupby("cluster")[genre_cols].mean()

# Visa de 3 vanligaste genrerna per kluster
for cluster_id in cluster_summary.index:
    top_genres = cluster_summary.loc[cluster_id].sort_values(ascending=False).head(3)
    print(f"\n📊 Kluster {cluster_id}: vanligaste genrer")
    for genre, score in top_genres.items():
        print(f"  - {genre}: {score:.2f}")
