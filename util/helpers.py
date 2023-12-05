import math
import random
import torch
import numpy as np
from scipy.sparse import dok_matrix

PADDED_IDX = 0
CLS_IDX = 1

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

def get_input_and_target(batch_input, vocabulary_length, masking_rate=0.3):
    playlist_header = 1
    input_length = batch_input.size(1)
    
    # Get the playlist song list that aren't padded.
    playlist_songs = ~(batch_input == PADDED_IDX)
    num_songs_in_playlists = []
    
    # Create the masked input as all Padded values
    masked_input = torch.full((5, input_length), PADDED_IDX)
    
    # Create the target as the lenght of the total songs of all Padded.
    # TODO Check if we need to set PADDED and CLS values as 1 in target
    target = torch.full((5, vocabulary_length), float(PADDED_IDX))

    # Loop through each playlist and create the masked input and target.
    for i in range(playlist_songs.shape[0]):
        # Collect the songs in the input
        num_songs_in_playlists.append(torch.sum(playlist_songs[i]).item())
        num_songs_in_playlist = num_songs_in_playlists[i] - playlist_header
        
        # Set number of songs to mask
        num_mask = math.floor(num_songs_in_playlist*masking_rate)
        num_keep = num_songs_in_playlist-num_mask
        
        # Get a random order of indicies starting after the playlist header.
        random_indices = torch.add(torch.randperm(num_songs_in_playlist), playlist_header)
        
        # If the number of songs to mask is greater than 0 mask songs.
        if num_mask > 0:
            masked_input[i][:playlist_header] = batch_input[i][:playlist_header]
            # Enter the songs to keep based on the random masking.
            masked_input[i][playlist_header:num_keep+playlist_header] = batch_input[i][random_indices[:num_keep]]
            
            # TODO figure out what Pad and CLS should be
        target[i][batch_input[i][random_indices[:num_keep]]] = float(0.0)
        # print(batch_input[i][random_indices[num_keep:num_songs_in_playlist]])
        target[i][batch_input[i][random_indices[num_keep:num_songs_in_playlist]]] = float(1.0)
        
        target_weights = target.clone()
        for i in range(batch_input.size(0)):
            num_ones = torch.sum(target_weights[i] == 1).item()
            while num_ones > 0:
                # TODO should we set to 2 to avoid CLS and Pad?
                idx = random.randint(2, vocabulary_length-1)
                if target_weights[i, idx] == 0.0:
                    target_weights[i, idx] = 1.0
                    num_ones -= 1
        
        return masked_input, target, target_weights