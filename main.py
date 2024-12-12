import pandas as pd
import numpy as np
from src.Hybrid import Hybrid_filter
from difflib import get_close_matches 

def get_track_index(dataset, track_name):
    track_match = dataset[dataset['track_name'].str.lower() == track_name.lower()]
    if not track_match.empty:
        return track_match.index[0]
    
    close_matches = get_close_matches(track_name, dataset['track_name'], n=1, cutoff=0.6)
    if close_matches:
        print(f"Did you mean '{close_matches[0]}'?")
        track_match = dataset[dataset['track_name'] == close_matches[0]]
        return track_match.index[0] if not track_match.empty else None
    
    return None

def run_pipeline(dataset, track_index, top_n=10, weight=0.7):
    recc_songs = Hybrid_filter(dataset, track_index, top_n=top_n, weight=weight)
    return recc_songs

if __name__ == "__main__":
    try:
        dataset = pd.read_csv(r"C:\Projects\Music_Recc_System\data\processed.csv")
        
        user_input = input("Enter the name of the track you like: ")
        track_index = get_track_index(dataset, user_input)

        if track_index is None:
            print("Sorry, the track was not found in the dataset.")
        else:
            recommendations = run_pipeline(dataset, track_index)
            print("\nRecommended Tracks:")
            print(recommendations)
    except Exception as e:
        print(f"An error occurred: {e}")
