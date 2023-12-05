import os
import pickle

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import Dataset


class PlaylistDataset(Dataset):
    def __init__(self, playlist_info_sparse_ids, songs_list):
        self.playlist_info_sparse_ids = playlist_info_sparse_ids
        self.songs_list = songs_list
        
    def __len__(self):
        # should return the number of samples aka the number of 
        return len(self.playlist_info_sparse_ids)
    
    def __getitem__(self, playlist_idx):
        playlist_song_sparse_ids = self.playlist_info_sparse_ids[playlist_idx]
        songs_vocab_ids = []
        for i in playlist_song_sparse_ids:
            # grab the id of the song in the internal vocabulary,
            #  and add 2 to it since the embedding lookup table has the padding and CLS at indices 0 and 1
            songs_vocab_ids.append(self.songs_list[self.songs_list['sparse_ids'] == i].index[0] + 2)

        full_padded_input = [1]
        full_padded_input.extend(songs_vocab_ids)

        len_curr = len(full_padded_input)
        if len_curr > 100:
            full_padded_input = full_padded_input[:100]
        elif len_curr < 100:
            padding = torch.zeros([1,100-len_curr], dtype=int)
            padding = padding.squeeze(0).tolist()
            full_padded_input.extend(padding)
        
        full_padded_input_tensor = torch.tensor(full_padded_input)

        return {"songs": full_padded_input_tensor}


class TransformerClassifier():
    # TODO: add more to init?
    def __init__(self, reTrain=False, name="TransformerNet.pkl"):
        self.name = "Transformer"
        self.pathName = name
        self.test_mapped_to_PID = pd.read_pickle("lib/test_playlist_mappings.pkl")
        self.songs_lookup = pd.read_pickle("lib/transformer/songs.pkl") 
        
        self.initModel(reTrain)
    
    # DONE
    def initModel(self, reTrain):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            print("Retraining model.")
            self.model = TransformerNet(
                num_tracks=self.num_tracks,
            )
            print("about to get dataloaders")
            train_dataloader, test_dataloader = self.getDataloaders()
            print("done getting dataloaders")
            self.trainModel(train_dataloader, test_dataloader)
        else:
            print(f"Evaluating model that is stored at {self.pathName}")
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))

    # DONE
    def getDataloaders(self):
        song_names_train = np.load("lib/transformer/song_names_train_1.npy", allow_pickle=True)
        song_names_test = np.load("lib/transformer/song_names_test.npy", allow_pickle=True)
        artist_names_train = np.load("lib/transformer/song_names_train_1.npy", allow_pickle=True)  #TODO change to artists!! this is a placeholder!!
        artist_names_test = np.load("lib/transformer/song_names_test.npy", allow_pickle=True) #TODO change to artists!! this is a placeholder!!

        train_dataset = PlaylistDataset(song_names_train, self.songs_lookup)
        test_dataset = PlaylistDataset(song_names_test, self.songs_lookup)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 5)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 5)
        return train_dataloader, test_dataloader
    
    # DONE
    def saveModel(self):
        pickle.dump(self.model, open(f"lib/{self.pathName}", "wb"))

    def train(model, train_loader, loss_fn, optimizer, device="cpu", epoch=-1):
        """
        Trains a model for one epoch (one pass through the entire training data).

        :param model: PyTorch model
        :param train_loader: PyTorch Dataloader for training data
        :param loss_fn: PyTorch loss function
        :param optimizer: PyTorch optimizer, initialized with model parameters
        :kwarg epoch: Integer epoch to use when printing loss and accuracy
        :returns: Accuracy score
        """
        total_loss = 0
        all_predictions = []
        all_targets = []
        loss_history = []
        
        sequence_len = 40 #TODO This shoudld be passed in.
        mask_perc = 30 #TODO This should be passed in.

        model = model.to(device)
        model.train()  # Set model in training mode
        
        # NOTE Assume inputs are padded
        for i, (inputs, targets, mask) in enumerate(train_loader):
            
            optimizer.zero_grad()
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
            outputs = model(inputs, mask)
            loss = loss_fn(outputs, targets, mask)
            loss.backward()
            optimizer.step()

            # Track some values to compute statistics
            total_loss += loss.item()

            # Save loss every 100 batches
            if (i % 100 == 0) and (i > 0):
                running_loss = total_loss / (i + 1)
                loss_history.append(running_loss)
                # print(f"Epoch {epoch + 1}, batch {i + 1}: loss = {running_loss:.2f}")

        final_loss = total_loss / len(train_loader)
        # Print average loss and accuracy
        print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}")
        return final_loss

    # DONE
    def getPredictionsFromTracks(
        self, 
        model_output, 
        playlist_tracks, 
        num_predictions, 
        all_tracks = pd.read_pickle("lib/tracks.pkl")
    ):
        # 1: obtain an ID for each entry in the model output
        # 2: sort the model output in descending order
        sorted_song_IDs_asc = np.argsort(model_output)

        # 3: obtain songs based on IDX --> NOTE this used to happen in a utility function but i cannot be bothered
        predicted_tracks = []
        for song_id in sorted_song_IDs_asc:
            sparse_id = self.songs[self.songs.index == song_id]['sparse_ids'][song_id]
            predicted_tracks.append(all_tracks[all_tracks['sparse_id'] == sparse_id].index[0])

        # 4: loop through until we have found 500 unique new songs
        curr_count = 0
        tracks_to_return = []
        print("right before loop")
        while curr_count < num_predictions:
            for track in reversed(predicted_tracks):
                if track not in playlist_tracks:
                    tracks_to_return.append(track)
                curr_count += 1

        return tracks_to_return

    # DONE
    def predict(self, 
        X,
        num_predictions,
        song_names_test=np.load("lib/transformer/song_names_test.npy"),
    ):
        """
        Method used to collect predictions from model output and 
            X:                  the playlist we are doing predictions on
            numPredictions:     we are doing 500 for this challenge
            embeddings_file:    contains the embeddings that can be used to 
        NOTE: removed the ability to pass track set in and instead will always use all tracks. can add back if necessary
        """
        pid, pTracks = X["pid"], X["tracks"]

        # this should be the same as the previous implementation because the train 
        # and test playlists are being grabbed in the same order as before
        # NOTE: this is because the sparse matrix for some reason did not populate with the order of the PIDs, 
        # so we would be passing in something like 27105, 11028, etc, when they live in the playlist test reference object as 0,1,2 etc.
        # @areena see our imessage photos for reference
        pid_vocab = self.test_mapped_to_PID[self.test_mapped_to_PID['Playlist Df Id'] == pid]['Matrix id']
    
        # get playlist embeddings for this playlist
        playlist_info = song_names_test[int(pid_vocab)]
        playlist_song_sparse_ids = playlist_info[pid_vocab]
        songs_vocab_ids = []
        for i in playlist_song_sparse_ids:
            # grab the id of the song in the internal vocabulary,
            #  and add 2 to it since the embedding lookup table has the padding and CLS at indices 0 and 1
            songs_vocab_ids.append(self.songs_list[self.songs_list['sparse_ids'] == i].index[0] + 2)
        full_padded_input = [1]
        full_padded_input.extend(songs_vocab_ids)
        len_curr = len(full_padded_input)
        if len_curr > 100:
            full_padded_input = full_padded_input[:100]
        elif len_curr < 100:
            padding = torch.zeros([1,100-len_curr], dtype=int)
            padding = padding.squeeze(0).tolist()
            full_padded_input.extend(padding)
        input_for_model = self.model.input_embeddings(full_padded_input)

        # get model prediction
        self.model.eval()
        model_output = self.model(input_for_model)
        # sort the song outputs
        predicted_songs = self.getPredictionsFromTracks(model_output, pTracks, num_predictions)

        return predicted_songs


class SongAndPlaylistEmbeddings(nn.Embedding):
    """
        Usage: embeddings_lookup = SongAndPlaylistEmbeddings(num_positions=V+1, embedding_dim=256, padding_idx=0)
        - where V is the vocabulary size (ie. the total number of songs), and the +1 is the padding
    """

    def __init__(self,num_positions: int, embedding_dim: int, padding_idx: int) -> None:
        super().__init__(num_positions, embedding_dim, padding_idx=padding_idx)
        self.song_embeddings = pd.read_pickle("../lib/transformer/songs.pkl")
        self.artist_embeddings = pd.read_pickle("../lib/transformer/songs.pkl") #TODO change to actual artists!! this is a placeholder!!
        self.weight = self._lookup_embeddings()

    def _lookup_embeddings(self) -> nn.Parameter:
        song_embeds = torch.tensor(np.array(self.song_embeddings['song_embeddings'].tolist()))
        artist_embeds = torch.tensor(np.array(self.artist_embeddings['song_embeddings'].tolist())) #TODO change to artist_embeddings!!!! placeholder!!

        # element-wise addition of embeddings (should work if they are the same dimension)
        concat_tokens = song_embeds + artist_embeds

        # add CLS first
        CLS_token = torch.mean(concat_tokens, dim=0, keepdim=True)
        concat_tokens_2 = torch.cat([CLS_token, concat_tokens], dim=0)
        # now add padding
        padding_token = torch.zeros([1,256])
        final_embeds = torch.cat([padding_token, concat_tokens_2], dim=0)

        out = nn.Parameter(torch.tensor(final_embeds, dtype=torch.float32), requires_grad=True)

        return out


class TransformerNet(nn.Module):
    """
    Transformer used for computing predictions given different input features
    """
    # TODO: Dana find out n_head, n_layers, 
    # Maxmimum playlist length  = ?
    # number of attention heads = 4 as placeholder, possible 
    def _init_(self, d_model=256, n_head=4, n_layers=3, num_songs = 1000):
        super(TransformerNet, self).__init__()
        # d_model = # of hidden dimensions
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.num_songs = num_songs

        # TODO: Retrieve embedding function from Maria
        # Number of embeddings = ?, 10 as a placeholder
        # num_embeddings = size of vocab (all songs)
        self.input_embeddings = SongAndPlaylistEmbeddings(num_embeddings=num_songs, embedding_dim=d_model, padding_idx = 0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=2*self.d_model,
            dropout=0.1,
            activation="gelu",
            # batch_first (bool) If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
            # TODO: DOUBLE CHECK FORMAT OF INPUT AND OUTPUT, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        ###
        
        # TODO: Ensure correct sizing on lin layer
        self.output_layer = nn.Linear(self.d_model, self.num_songs)
        
    
    def forward(self, input_ids, padding_mask): #TODO
        
        # Inputs --> Embeddings
        transformer_inputs = self.input_embeddings(input_ids)
        print("Input to transformer encoder size: ", transformer_inputs.shape)
        # Embeddings --> Transformer
        outputs = self.transformer_encoder(transformer_inputs, src_key_padding_mask=padding_mask)
        print("Shape of outputs after transformer encoder: ", outputs.shape)
        # Check size
        print
        # TODO: Confirm with Areena that this is what she meant by CLS token
        outputs = outputs[:,0,:].squeeze()
        print("Expected shape after fitler and squeeze: (Batch, embedding size = 256)")
        print("Output after CLS filter and squeeze: ", outputs.shape)

        # TODO: Confirm this is the expected shape
        # (Batch, embedding size = 256)
        
        # Transformer --> Linear output layer
        # TODO: Confirm linear goes after squeeze
        outputs = self.output_layer(outputs)
        print("Expected shape: (B, num_songs)")
        print("Actual shape after linear layer, before signmoid: ", outputs.shape)

        # Now apply sigmoid
        # Dimensions should be (B, num_songs)
        song_predictions = nn.Sigmoid(outputs)

        # Dimensions of song_predictions shoudl be (1, num_songs)
        print("Song prediction shape (after sigmoid): ", song_predictions.shape)

        return song_predictions
        