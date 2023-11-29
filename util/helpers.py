import math
import random
import numpy as np
from scipy.sparse import dok_matrix

def playlistToSparseMatrixEntry(playlist, songs):
    """
    Converts a playlist with list of songs
    into a sparse matrix with just one row
    """
    # print(songs.iloc[1:5])
    playlistMtrx = dok_matrix((1, len(songs)))
    tracks = [songs.loc[str(x)]["sparse_id"] for x in list(playlist["tracks"])]
    playlistMtrx[0, tracks] = 1
    return playlistMtrx.tocsr()


def getPlaylistTracks(playlist, songs):
    return [songs.loc[x] for x in playlist["tracks"]]


def getTrackandArtist(trackURI, songs):
    song = songs.loc[trackURI]
    return (song["track_name"], song["artist_name"])


def getTracksFromSparseID(ids, songs, track_mappings):
    np_ids = np.array(ids)
    sparse_ids = track_mappings[track_mappings['My Matrix Id'].isin(np_ids)]['Sparse Id']
    track_ids = songs[songs['sparse_id'].isin(sparse_ids)].index

    return track_ids


def getGroundTruthFromPlaylistID(
    playlist_id,
    sparse_matrix,
    # mappings,
    # playlists,
    num_samples=20
):
    """
        playlist_id:    should correspond with a playlist's sparse_id
        sparse_matrix:  the sparseMatrix for the entire song dataset
        playlists:      dataframe containing playlist information
        num_samples:    number of samples of TRUE and FALSE song values to check
    """
    # 1. get actual playlist ID
    # playlist_id = mappings[mappings.index == idx].loc[idx]['Playlist Df Id']
    # print(f"given playlist id: {idx}, mapped playlist id: {playlist_id}")

    # 2. get length of playlist
    # playlist_length = playlists[playlists['pid'] == playlist_id].loc[playlist_id]['num_tracks']
    # num_samples = min(playlist_length, num_samples)
    # print(f"num_samples is: {num_samples}")

    array_songs_in_playlist_idx = sparse_matrix[playlist_id]

    # 3. obtain num_samples songs that ARE in the playlist (sparse_ids)
    # 4. obtain num_samples songs that ARE NOT in the playlist (sparse_ids)
    sparse_ids_in_play, sparse_ids_not_in_play = [], []
    for index, num in enumerate(array_songs_in_playlist_idx): #FIX THIS
        if num != 0.0:
            sparse_ids_in_play.append(index)
        else:
            sparse_ids_not_in_play.append(index)
    
    num_samples = min(num_samples, len(sparse_ids_in_play))
    ids_in_playlist = list(np.random.choice(sparse_ids_in_play, size=num_samples))
    ids_not_in_playlist = list(np.random.choice(sparse_ids_not_in_play, size=num_samples))

    ids_in_playlist.extend(ids_not_in_playlist)
    # 5. return sparse_ids
    return ids_in_playlist, num_samples


def get_output_size(input_size, padding, stride, kernel):   
    return math.floor((input_size + 2*padding - kernel)/stride) + 1


def obscurePlaylist(orig_tracks, percentToObscure): 
    """
    Obscure a portion of a playlist's songs for testing
    """
    k = int(len(orig_tracks) * percentToObscure)
    indices = random.sample(range(len(orig_tracks)), k)
    obscured = [orig_tracks[i] for i in indices]
    tracks = [i for i in orig_tracks + obscured if i not in orig_tracks or i not in obscured]
    return tracks, obscured