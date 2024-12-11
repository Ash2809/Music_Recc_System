def collaborative_filter_genre(dataset,n):
    genre_group=dataset.groupby("track_genre")
    top_songs=genre_group.apply(lambda x:x.nlargest(n,"popularity"))
    top_songs["collab_score"]=top_songs.groupby("track_genre").cumcount(ascending=False)+1
    return top_songs
if __name__ =="__main__":
    try:
        collaborative_filter_genre()
    except Exception as e:
        print(f"an error occured while collab_filtering {e}")
    