from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def content_filter(data, top_n = 10, track_index):
    numerical_features = [
        "popularity", "duration_ms", "explicit", "danceability", "energy", "key",
        "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature"
    ]

    feature_matrix = data[numerical_features]
    similarity_matrix = cosine_similarity(feature_matrix)

    similar_tracks = similarity_matrix[track_index]
    recommended_indices = np.argsort(similar_tracks)[::-1][:top_n + 1]
    recommended_indices = [idx for idx in recommended_indices if idx != track_index]


if __name__ == "__main__":
    try:
        content_filter()
    except Exception as e:
        print(f"An error occured while content filtering {e}")
