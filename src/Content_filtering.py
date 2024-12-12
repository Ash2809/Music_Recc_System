from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import faiss

def content_filter(data, track_index, top_n=10):
    numerical_features = [
        "popularity", "duration_ms", "explicit", "danceability", "energy", "key",
        "loudness", "mode", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature"
    ]

    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(data[numerical_features].values).astype('float32')

    index = faiss.IndexFlatL2(feature_matrix.shape[1])  
    index.add(feature_matrix)

    if track_index < 0 or track_index >= len(data):
        raise IndexError("track_index is out of range.")
    
    track_vector = feature_matrix[track_index].reshape(1, -1)
    distances, indices = index.search(track_vector, top_n + 1)

    recommended_indices = indices[0][1:] 
    recommended_distances = distances[0][1:]

    similarity_scores = [1 / (1 + dist) for dist in recommended_distances]

    recommended_songs = data.iloc[recommended_indices].copy()
    recommended_songs['similarity'] = similarity_scores

    return recommended_songs

if __name__ == "__main__":
    try:
        # Load dataset
        data = pd.read_csv(r"C:\Projects\Music_Recc_System\data\processed.csv")
        track_index = 0  # Example: Index of the track to find similar tracks for
        recommendations = content_filter(data, track_index, top_n=10)

        # Display recommendations
        print("Recommended Tracks:")
        print(recommendations[['track_name', 'artists', 'similarity']])  # Adjust columns as needed
    except Exception as e:
        print(f"An error occurred while content filtering: {e}")
