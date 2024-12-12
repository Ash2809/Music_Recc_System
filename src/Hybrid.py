from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import faiss
from src.Collaborative_filtering import collaborative_filter_genre
from src.Content_filtering import content_filter

def Hybrid_filter(dataset, track_index, top_n=10, weight=0.2):
    content_recs = content_filter(dataset, track_index, top_n=top_n)
    content_scores = {i: score for i, score in zip(content_recs.index, content_recs["similarity"])}

    collab_recs = collaborative_filter_genre(dataset, n=top_n)
    collab_scores = {i: score for i, score in zip(collab_recs.index, collab_recs["collab_score"])}

    max_content = max(content_scores.values()) if content_scores else 1
    normalized_content = {k: v / max_content for k, v in content_scores.items()}

    max_collab = max(collab_scores.values()) if collab_scores else 1
    normalized_collab = {k: v / max_collab for k, v in collab_scores.items()}

    hybrid_scores = {}
    all_ids = set(normalized_content.keys()).union(set(normalized_collab.keys()))

    for idx in all_ids:
        content_score = normalized_content.get(idx, 0)
        collab_score = normalized_collab.get(idx, 0)
        hybrid_scores[idx] = weight * content_score + (1 - weight) * collab_score

    sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

    sorted_recommendations = []
    for idx, score in sorted_hybrid[:top_n]:
        track_info = dataset.iloc[idx]
        sorted_recommendations.append({
            'track_name': track_info['track_name'],
            'artists': track_info['artists'],
            'hybrid_score': score
        })
    
    return pd.DataFrame(sorted_recommendations)

if __name__ == "__main__":
    try:
        dataset = pd.read_csv(r"C:\Projects\Music_Recc_System\data\processed.csv")
        track_index = 5  
        recommendations = Hybrid_filter(dataset, track_index, top_n=10)

        print("Hybrid Recommendations:")
        print(recommendations)
    except Exception as e:
        print(f"An error occurred while hybrid filtering: {e}")
