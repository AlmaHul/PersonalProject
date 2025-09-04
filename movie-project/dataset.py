import pandas as pd


# Läs in u.movie-project

ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv("u.data", sep="\t", names=ratings_cols, encoding="latin-1")


# Läs in u.item

item_cols = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies = pd.read_csv("u.item", sep="|", names=item_cols, encoding="latin-1")


# Läs in u.user

user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
users = pd.read_csv("u.user", sep="|", names=user_cols, encoding="latin-1")


# Slå ihop dataframes

ratings_movies = pd.merge(ratings, movies, on="movie_id")
full_data = pd.merge(ratings_movies, users, on="user_id")


# Ta bort onödiga kolumner
full_data = full_data.drop(columns=["timestamp", "video_release_date"])

# Ta bort rader med saknade värden
full_data = full_data.dropna()


# Spara till CSV

full_data.to_csv("movies_clean.csv", index=False)

print("Rensad movie-project sparad som 'movies_clean.csv'")
print(full_data.head())
