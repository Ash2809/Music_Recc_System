import pandas as pd

def collaborative_filter_genre(dataset, n):
    genre_group = dataset.groupby("track_genre", group_keys=False)
    top_songs = genre_group.apply(lambda x: x.nlargest(n, "popularity"))
    top_songs = top_songs.reset_index(drop=True)
    top_songs["collab_score"] = top_songs.groupby("track_genre").cumcount(ascending=False) + 1
    return top_songs

if __name__ == "__main__":
    try:
        dataset = pd.read_csv(r"C:\Projects\Music_Recc_System\data\processed.csv")
        n = 10
        top_songs = collaborative_filter_genre(dataset, n)
        print(top_songs)
    except Exception as e:
        print(f"An error occurred while collaborative filtering: {e}")
