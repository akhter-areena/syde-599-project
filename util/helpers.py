import random
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


def getTracksFromSparseID(ids, songs):
    return [songs[songs['sparse_id'] == i].index[0] for i in ids]


def getGroundTruthFromPlaylistID(playlist_id, sparse_matrix, playlists, num_samples=20):
    """
        playlist_id:    should correspond with a playlist's sparse_id
        sparse_matrix:  the sparseMatrix for the entire song dataset
        playlists:      dataframe containing playlist information
        num_samples:    number of samples of TRUE and FALSE song values to check
    """
    # 1. get length of playlist
    playlist_length = playlists[playlists['pid'] == playlist_id].loc[0]['num_tracks']
    num_samples = min(playlist_length, num_samples)

    # 2. obtain num_samples songs that ARE in the playlist (sparse_ids)
    # 3. obtain num_samples songs that ARE NOT in the playlist (sparse_ids)
    
    # 4. return sparse_ids

    pass


def obscurePlaylist(orig_tracks, percentToObscure): 
    """
    Obscure a portion of a playlist's songs for testing
    """
    k = int(len(orig_tracks) * percentToObscure)
    indices = random.sample(range(len(orig_tracks)), k)
    obscured = [orig_tracks[i] for i in indices]
    tracks = [i for i in orig_tracks + obscured if i not in orig_tracks or i not in obscured]
    return tracks, obscured