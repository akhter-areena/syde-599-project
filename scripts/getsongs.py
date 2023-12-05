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
for i in tqdm(list(playlists_train["tracks"].values[:60])):
    if len(i) < 2:
        print(f'length of playlist train small! Delete! Length is {len(i)}')

for i in tqdm(list(playlists_train["tracks"].values[:5])):
    if len(i) < 2:
        print(f'length of playlist test small! Delete! Length is {len(i)}')


songs = tracks[['sparse_id', 'track_name']]

print("POPULATE TRAIN SONG ID MATRIX...")

all_sparse_ids = []

# Populate train matrix consisting of playlists where each playlist consists of song ids
train_playlist_songs = []

for i in tqdm(playlists_train['tracks'].values[:1000]):
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

print("GET EMBEDDINGS....")

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

concat_playlist_songs = test_playlist_songs + train_playlist_songs


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

model.build_vocab(concat_playlist_songs)

print(model)

callback = Callback()
t = time()

model.train(concat_playlist_songs,
            total_examples=model.corpus_count,
            epochs=100,
            compute_loss=True,
            callbacks=[callback])

print("SAVE MODEL....")

# Save model
model.save("song_name2vec.model")


# Train matrix
print("GET FINAL TRAIN MATRIX OF ALL PLAYLISTS AND ALL SONGS....")


train_playlistSong = np.full(
    (len(train_playlist_songs), len(unique_np_sparse), 256), -5)
train_sparse_matrix = np.full(
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

    # Set index to embedding if playlist has song
    train_playlistSong[playlistIDX, curr_playlist_song_ids] = embeddings
    train_sparse_matrix[playlistIDX, curr_playlist_song_ids] = 1


# Test matrix

print("GET FINAL TEST MATRIX OF ALL PLAYLISTS AND ALL SONGS....")

test_playlistSong = np.full(
    (len(test_playlist_songs), len(unique_np_sparse), 256), -5)

test_sparse_matrix = np.full(
    (len(test_playlist_songs), len(unique_np_sparse)), 0)


for i in tqdm(range(len(test_playlist_songs))):
    # Get playlist and sparse ids from DF
    playlistIDX = i
    curr_playlist_sparse_ids = test_playlist_songs[i]

    curr_playlist_song_ids = [mappings[id] for id in curr_playlist_sparse_ids]

    # Get embeddings
    embeddings = model.wv[curr_playlist_sparse_ids]

    # Set index to embedding if playlist has song
    test_playlistSong[playlistIDX, curr_playlist_song_ids] = embeddings
    test_sparse_matrix[playlistIDX, curr_playlist_song_ids] = 1

# Pickle file the outputs

print("PRINT SHAPE....")

print(test_sparse_matrix.shape)
print(test_playlistSong.shape)
print(train_playlistSong.shape)
print(train_sparse_matrix.shape)

print("SAVE FINAL SONG AND PLAYLIST MATRICES TO FILE")

np.save("song_names_train", train_playlistSong, allow_pickle=True)
np.save("song_names_test", test_playlistSong, allow_pickle=True)
np.save("train_sparse_matrix", train_sparse_matrix, allow_pickle=True)
np.save("test_sparse_matrix", test_sparse_matrix, allow_pickle=True)