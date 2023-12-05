import logging
from tqdm import tqdm
from gensim.models import Word2Vec
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from gensim.models.callbacks import CallbackAny2Vec

import multiprocessing
from time import time


PATH_TO_PLAYLIST_DATA = r"/lib"

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


print("REWORK DATA....")

artists = tracks[['artist_name', 'sparse_id']]

# Get unique id corresponding to each artist
artists_to_id = pd.DataFrame(
    artists['artist_name'].unique(), columns=["artist_name"])
artists_to_id.head()

artists_to_id['artist_index'] = artists_to_id.index

print("GET ARTIST IDS AND SPARSE IDS FOR EACH TRAIN PLAYLIST....")

# Populate another matirx consisting of sparse ids
all_sparse_ids = []

# Populate matrix consisting of playlists where each playlist consists of artist ids
train_playlist_artists = []
train_playlist_sparse_ids = []

for i in tqdm(playlists_train['tracks'].values[:25]):
    artist_names = artists[artists.index.isin(i)]['artist_name']
    artist_ids = list(pd.merge(artist_names, artists_to_id,
                      how="left")['artist_index'])
    train_playlist_artists.append(artist_ids)
    train_playlist_sparse_ids.append(
        list(artists[artists.index.isin(i)]['sparse_id']))

    unique_ids = artists[artists.index.isin(i)]['sparse_id'].unique()

    for id in unique_ids:
        all_sparse_ids.append(id)


print("GET ARTIST IDS AND SPARSE IDS FOR EACH TEST PLAYLIST....")

# Populate matrix consisting of playlists where each playlist consists of song ids
test_playlist_artists = []
test_playlist_sparse_ids = []

for i in tqdm(playlists_test['tracks'].values[:5]):
    artist_names = artists[artists.index.isin(i)]['artist_name']
    artist_ids = list(pd.merge(artist_names, artists_to_id,
                      how="left")['artist_index'])
    test_playlist_artists.append(artist_ids)
    test_playlist_sparse_ids.append(
        list(artists[artists.index.isin(i)]['sparse_id']))

    unique_ids = artists[artists.index.isin(i)]['sparse_id'].unique()

    for id in unique_ids:
        all_sparse_ids.append(id)

# Get all sparse ids
np_sparse = np.array(all_sparse_ids)

# All of the sparse ids
unique_np_sparse = np.unique(np_sparse)

# Create mapping of index of sparse id to actual sparse id. This index is our matrix index.
mappings = {id: index for index, id in enumerate(unique_np_sparse)}


print("GET EMBEDDINGS....")

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

concat_playlist_artists = test_playlist_artists + train_playlist_artists


class Callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.training_loss = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 1:
            current_loss = loss
        else:
            current_loss = loss - self.loss_previous_step
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.training_loss.append(current_loss)
        self.epoch += 1
        self.loss_previous_step = loss


model = Word2Vec(
    vector_size=256,
    window=10,
    min_count=1,
    sg=0,
    negative=20,
    workers=multiprocessing.cpu_count()-1)

print(model)

logging.disable(logging.NOTSET)  # enable logging
t = time()

model.build_vocab(concat_playlist_artists)

print(model)

callback = Callback()
t = time()

model.train(concat_playlist_artists,
            total_examples=model.corpus_count,
            epochs=1,
            compute_loss=True,
            callbacks=[callback])


print("POPULATE FINAL TRAIN PLAYLIST MATRIX....")

# 256 is number of embeddings
train_playlistArtist = np.full((len(train_playlist_artists), len(
    unique_np_sparse), 256), -5)

# Iterate through playlists
for i in tqdm(range(len(train_playlist_artists))):

    playlistIDX = i
    # Index of songs in playlists
    curr_playlist_sparse_ids = train_playlist_sparse_ids[i]
    curr_playlist_song_ids = [mappings[id] for id in curr_playlist_sparse_ids]

    # Artist ids of songs in playlist
    curr_playlist_artist_ids = train_playlist_artists[i]

    # Get embeddings
    embeddings = model.wv[curr_playlist_artist_ids]

    # Set index to embedding if playlist has song
    train_playlistArtist[playlistIDX,
                         curr_playlist_song_ids] = embeddings


# Test matrix

print("POPULATE FINAL TEST PLAYLIST MATRIX....")

test_playlistArtist = np.full((len(test_playlist_artists), len(
    unique_np_sparse), 256), -5)

for i in tqdm(range(len(test_playlist_artists))):
    playlistIDX = i
    # Index of songs in playlists
    curr_playlist_sparse_ids = test_playlist_sparse_ids[i]
    curr_playlist_song_ids = [mappings[id] for id in curr_playlist_sparse_ids]

    # Artist ids of songs in playlist
    curr_playlist_artist_ids = test_playlist_artists[i]

    # Get embeddings
    embeddings = model.wv[curr_playlist_artist_ids]

    # Set index to embedding if playlist has song
    test_playlistArtist[playlistIDX,
                        curr_playlist_song_ids] = embeddings


print("PRINT SHAPE....")

print(test_playlistArtist.shape)
print(train_playlistArtist.shape)

# Pickle file the outputs

print("SAVE FINAL ARTIST AND PLAYLIST MATRICES TO FILE......")

np.save("artist_names_train", train_playlistArtist, allow_pickle=True)
np.save("artist_names_test", test_playlistArtist, allow_pickle=True)