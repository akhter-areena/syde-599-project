import logging
from tqdm import tqdm
from gensim.models import Word2Vec
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE


PATH_TO_PLAYLIST_DATA = r"lib"
NUM_COMPONENTS = 3

print("GET DATA....")

tracks = pd.read_pickle(f"{PATH_TO_PLAYLIST_DATA}/tracks.pkl")
playlists_test = pd.read_pickle(f"{PATH_TO_PLAYLIST_DATA}/playlists_test.pkl")
playlists_train = pd.read_pickle(
    f"{PATH_TO_PLAYLIST_DATA}/playlists_train.pkl")

print("SEE IF PLAYLIST IS TOO SMALL....")


# See if playlists too small
for i in tqdm(list(playlists_train["tracks"].values)):
    if len(i) < 2:
        print(f'length of playlist train small! Delete! Length is {len(i)}')

for i in tqdm(list(playlists_train["tracks"].values)):
    if len(i) < 2:
        print(f'length of playlist test small! Delete! Length is {len(i)}')


songs = tracks[['sparse_id', 'track_name']]

print("POPULATE TRAIN SONG ID MATRIX...")

all_sparse_ids = []

# Populate train matrix consisting of playlists where each playlist consists of song ids
train_playlist_songs = []

for i in tqdm(playlists_train['tracks'].values[:25]):
    train_playlist_songs.append(list(songs[songs.index.isin(i)]["sparse_id"]))
    unique_ids = songs[songs.index.isin(i)]["sparse_id"].unique()

    for id in unique_ids:
        all_sparse_ids.append(id)

print("POPULATE TEST SONG ID MATRIX...")

# Populate test matrix consisting of playlists where each playlist consists of song ids
test_playlist_songs = []

for i in tqdm(playlists_test['tracks'].values[:5]):
    test_playlist_songs.append(list(songs[songs.index.isin(i)]["sparse_id"]))

    unique_ids = songs[songs.index.isin(i)]["sparse_id"].unique()

    for id in unique_ids:
        all_sparse_ids.append(id)

print("GET SPARSE MATRIX MAPPINGS...")

# Get all sparse ids
np_sparse = np.array(all_sparse_ids)

# All of the sparse ids
unique_np_sparse = np.unique(np_sparse)

# Create mapping of index of sparse id to actual sparse id. This index is our matrix index.
mappings = {id: index for index, id in enumerate(unique_np_sparse)}


print("IMPORT MODEL...")

# import model
model = Word2Vec.load("lib/song_name2vec.model")

# Train matrix
print("GET FINAL TRAIN MATRIX OF ALL PLAYLISTS AND ALL SONGS....")

num_components = NUM_COMPONENTS

train_playlistSong = np.full(
    (len(train_playlist_songs), len(unique_np_sparse), num_components), -5)
new_sparse_matrix = np.full(
    (len(train_playlist_songs), len(unique_np_sparse)), 0)

for i in tqdm(range(len(train_playlist_songs))):
    # Get playlist and sparse ids from DF
    playlistIDX = i
    curr_playlist_sparse_ids = train_playlist_songs[i]

    curr_playlist_song_ids = [mappings[id] for id in curr_playlist_sparse_ids]

    # Get embeddings
    embeddings = model.wv[curr_playlist_sparse_ids]

    # Get embeddings
    embeddings = model.wv[curr_playlist_sparse_ids]

    tsne = TSNE(n_components=NUM_COMPONENTS, perplexity=5)
    small_embeddings = tsne.fit_transform(embeddings)

    # Set index to embedding if playlist has song
    train_playlistSong[playlistIDX, curr_playlist_song_ids] = small_embeddings
    new_sparse_matrix[playlistIDX, curr_playlist_song_ids] = 1


# Test matrix

print("GET FINAL TEST MATRIX OF ALL PLAYLISTS AND ALL SONGS....")

test_playlistSong = np.full(
    (len(test_playlist_songs), len(songs), num_components), -5)

for i in tqdm(range(len(test_playlist_songs))):
    # Get playlist and sparse ids from DF
    playlistIDX = i
    curr_playlist_sparse_ids = test_playlist_songs[i]

    curr_playlist_song_ids = [mappings[id] for id in curr_playlist_sparse_ids]

    # Get embeddings
    embeddings = model.wv[curr_playlist_sparse_ids]

    tsne = TSNE(n_components=NUM_COMPONENTS, perplexity=5)
    small_embeddings = tsne.fit_transform(embeddings)

    # Set index to embedding if playlist has song
    test_playlistSong[playlistIDX, curr_playlist_song_ids] = small_embeddings
    new_sparse_matrix[playlistIDX, curr_playlist_song_ids] = 1

# Pickle file the outputs

print("SAVE FINAL SONG AND PLAYLIST MATRICES TO FILE")

np.save("lib/song_names_train", train_playlistSong, allow_pickle=True)
np.save("lib/song_names_test", test_playlistSong, allow_pickle=True)
np.save("lib/new_sparse_matrix", new_sparse_matrix, allow_pickle=True)