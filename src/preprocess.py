import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocessor(path,output_path) -> pd.DataFrame:
    dataset = pd.read_csv(path)

    dataset=dataset.drop(columns=["Unnamed: 0", "track_id", "album_name"], axis = 1)
    dataset.dropna(inplace=True)
    dataset["explicit"]=dataset["explicit"].astype(int)

    le=LabelEncoder()
    # dataset["track_genre"]=le.fit_transform(dataset["track_genre"])
    # dataset["artists"]=le.fit_transform(dataset["artists"])
    
    scaler = StandardScaler()
    numeric_features = ["danceability", "energy", "tempo", "valence", "loudness", "speechiness",
                        "acousticness", "instrumentalness", "liveness", "popularity", "duration_ms"]
    
    dataset[numeric_features] = scaler.fit_transform(dataset[numeric_features])

    dataset.to_csv(output_path, header = True, index = False)
    print(f"Preprocessed data saved to {output_path}")

    return dataset

if __name__ == "__main__":
    path = r"C:\Projects\Music_Recc_System\data\dataset.csv"
    output_path = r"C:\Projects\Music_Recc_System\data\processed.csv"
    try:
        dataset = preprocessor(path,output_path)
        print("Data Preprocessing Completed.")
    except Exception as e:
        print("An error Occured while preprocessing {e}")