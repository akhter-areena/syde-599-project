import json, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import dok_matrix
from sklearn.model_selection import train_test_split
import argparse
import os

def parseTrackURI(uri):
    return uri.split(":")[2]

def processPlaylistForClustering(full_playlists, playlists_to_store, tracks):
    """
    Create sparse matrix mapping playlists to track
    lists that are consumable by most clustering algos
    """

    # List of all track IDs in db
    trackIDs = list(tracks["tid"])
    
    # Map track id to matrix index
    IDtoIDX = {k:v for k,v in zip(trackIDs,range(0,len(trackIDs)))}
    
    fullPlaylistIDs = list(full_playlists["pid"])
    toStorePlaylistIDs = list(playlists_to_store["pid"])
    
    print("Create sparse matrix mapping playlists to tracks")
    playlistSongSparse = dok_matrix((len(fullPlaylistIDs), len(trackIDs)), dtype=np.float32)

    for i in tqdm(range(len(toStorePlaylistIDs))):
        # Get playlist and track ids from DF
        currPlaylistID = toStorePlaylistIDs[i]
        trackID = playlists_to_store.loc[currPlaylistID]["tracks"]
        playlistIDX = currPlaylistID
        
        # Get matrix index for track id
        trackIDX = [IDtoIDX.get(i) for i in trackID]
        
        # Set index to 1 if playlist has song
        playlistSongSparse[playlistIDX, trackIDX] = 1 

    return playlistSongSparse.tocsr(), IDtoIDX

def createDFs(idx, numFiles, path, files, testPercent):
    """
    Creates playlist and track DataFrames from
    json files
    """
    # Get correct number of files to work with
    files = files[idx:idx+numFiles]

    tracksSeen = set()
    playlistsLst = []
    trackLst = []

    print("Creating track and playlist DFs")
    for i, FILE in enumerate(tqdm(files)):
        # get full path to file
        name = path + FILE 
        with open(name) as f:
            data = json.load(f)
            playlists = data["playlists"]

            # for each playlist
            for playlist in playlists:
                for track in playlist["tracks"]:
                    if track["track_uri"] not in tracksSeen:
                        tracksSeen.add(track["track_uri"])
                        trackLst.append(track)
                playlist["tracks"] = [parseTrackURI(x["track_uri"]) for x in playlist["tracks"]]
                playlistsLst.append(playlist)
    
    playlistDF = pd.DataFrame(playlistsLst)
    playlistDF.set_index("pid")

    # Split up playlist so that we can test accuracy on unseen playlists
    playlistDF_train, playlistDF_test = train_test_split(playlistDF, test_size=testPercent)

    # We still want to store ALL tracks though
    tracksDF = pd.DataFrame(trackLst)
    # Split id from spotifyURI for brevity
    tracksDF["tid"] = tracksDF.apply(lambda row: parseTrackURI(row["track_uri"]), axis=1)

    # Print out some of the created DFs for testing purposes
    print(f"Train playlist dataframe with length {len(playlistDF_train)}")    
    print(playlistDF_train.head(10))

    print(f"Test playlist dataframe with length {len(playlistDF_test)}")    
    print(playlistDF_test.head(10))

    print(f"Track playlist dataframe with length {len(tracksDF)}")    
    print(tracksDF.head(10))

    playlistClusteredDF, IDtoIDXMap = processPlaylistForClustering(
                                                    full_playlists=playlistDF,
                                                    playlists_to_store=playlistDF_train,
                                                    tracks=tracksDF)

    # Add sparseID for easy coercision to sparse matrix for training data
    tracksDF["sparse_id"] = tracksDF.apply(lambda row: IDtoIDXMap[row["tid"]], axis=1)
    tracksDF = tracksDF.set_index("tid")
    
    # Write DFs to CSVs
    print(f"Pickling {len(playlistDF)} original all playlists")
    playlistDF.to_pickle("lib/playlists_all.pkl")

    print(f"Pickling {len(playlistDF_train)} training playlists")
    playlistDF_train.to_pickle("lib/playlists_train.pkl")

    print(f"Pickling {len(playlistDF_test)} test playlists")
    playlistDF_test.to_pickle("lib/playlists_test.pkl")

    print(f"Pickling {len(tracksDF)} tracks")
    tracksDF.to_pickle("lib/tracks.pkl")

    print(f"Pickling clustered playlists")
    pickle.dump(playlistClusteredDF, open(f"lib/playlistSparse.pkl", "wb"))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numFiles', required=True, type=int)
    parser.add_argument('--testPercent', required=True, type=float)
    args = parser.parse_args()

    if args.numFiles > 0:
        print(f"Reading in {args.numFiles} files of data.")
        # Extract of files to read in
        def sortFile(f):
            f = f.split('.')[2].split('-')[0]
            return int(f)
        files = os.listdir("spotify_million_playlist_dataset/data")
        files.sort(key=sortFile)
        
        # Create DFs and write them as pickle files
        createDFs(
            idx=0, 
            numFiles=args.numFiles,
            path="spotify_million_playlist_dataset/data/",
            files=files,
            testPercent=args.testPercent
        )

main()