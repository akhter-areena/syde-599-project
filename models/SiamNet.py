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

from util.helpers import getTracksFromSparseID, getGroundTruthFromPlaylistID


# Note would need to change if added more features
class PlaylistDataset(Dataset):
    def __init__(self, song_names, artist_names):
        self.song_names = song_names
        self.artist_names = artist_names

    def __len__(self):
        return len(self.song_names)

    def __getitem__(self, playlist_idx):
      song_names_torch = torch.tensor(self.song_names)
      aritst_names_torch = torch.tensor(self.artist_names)

      return {"song_names": song_names_torch[playlist_idx], 'artist_names': aritst_names_torch[playlist_idx]}


class SiamNetClassifier():
    def __init__(self, playlists, sparsePlaylists, tracks, reTrain=False, name="SiamNetClassifier.pkl"):
        self.name = "SiamNet"
        self.pathName = name
        self.playlistData = sparsePlaylists
        self.playlists = playlists 
        self.tracks = tracks
        self.num_tracks = len(tracks)
        # NOTE: make this not hardcoded
        self.train_mapped_to_PID = pd.read_pickle("lib/train_playlist_mappings.pkl")
        self.test_mapped_to_PID = pd.read_pickle("lib/test_playlist_mappings.pkl")
       
        self.initModel(reTrain)


    def initModel(self, reTrain):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            print("Retraining model.")
            self.model = SiamNet(
                num_tracks=self.num_tracks,
            )
            print("about to get dataloaders")
            train_dataloader, test_dataloader = self.getDataloaders()
            print("done getting dataloaders")
            self.trainModel(train_dataloader, test_dataloader)
        else:
            print(f"Evaluating model that is stored at {self.pathName}")
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))


    def getDataloaders(self):
        song_names_train = np.load("lib/song_names_train.npy")
        song_names_test = np.load("lib/song_names_test.npy")
        artist_names_train = np.load("lib/artist_names_train.npy")
        artist_names_test = np.load("lib/artist_names_test.npy")
        # artist_names_train = np.load("lib/song_names_train.npy")
        # artist_names_test = np.load("lib/song_names_test.npy")

        train_dataset = PlaylistDataset(song_names_train, artist_names_train)
        test_dataset = PlaylistDataset(song_names_test, artist_names_test)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 5)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 5)
        return train_dataloader, test_dataloader

    def trainModel(self, train_dataloader, test_dataloader, num_epochs=50, device='cpu'): #TODO change num_epochs back
        train_losses = []
        test_losses = []
        train_metrics = []
        test_metrics = []

        print("in trainModel")
        for epoch in range(num_epochs):
            acc, final_loss = self._train(epoch, train_dataloader)
            acc, final_loss = self._test(epoch, test_dataloader)

            # train_losses.append(train_loss)
            # test_losses.append(test_loss)
            # train_metrics.append(train_acc)
            # test_metrics.append(test_acc)
        
        return train_losses, train_metrics, test_losses, test_metrics
        #NOTE returning these in case we wish to plot the training curves during development

    def _train(self, epoch, train_loader, device="cpu"):
        # print("in train")
        total_loss = 0
        all_predictions = []
        all_targets = []
        loss_history = []

        # NOTE: these aren't handled very tastefully, maybe parameterize??
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 1e-5
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        # print("about to do model.to(device)")
        self.model = self.model.to(device)
        # print("about to do model.train()")
        self.model.train()
        
        ''' TRAIN '''
        # print("about to enumerate dataloader")
        for i, inputs in enumerate(train_loader):
            # print(f"train: {i}")
            # print("begin inner loop")
            optimizer.zero_grad()
            # print("about to do inputs.to(device)")
            inputs_x1, inputs_x2 = inputs['song_names'], inputs['artist_names']
            inputs_x1, inputs_x2 = inputs_x1.to(device), inputs_x2.to(device)
            # print("about to do forward pass")
            outputs = self.model(inputs_x1, inputs_x2)
            # print("done forward pass")

            # we wish to fit to some of the ground truth, but not every single
            #   entry since the whole idea is that we can be learning new 
            #   relationships between songs that may not be in the original
            # dataloader returns the playlist embeddings in order as they appear
            #   in the sparse matrix, so we can leverage that in order to obtain
            #   their ground truth labels
            ids, num_samples = getGroundTruthFromPlaylistID(i, self.playlistData)
            # create labels
            targets = torch.zeros(2 * num_samples)
            targets[:num_samples] = 1.0
            outputs_for_loss = torch.index_select(outputs, dim=1, index=torch.tensor(ids))
            # print(outputs_for_loss.shape)
            # print(targets.shape)
            targets = targets.unsqueeze(0)
            # print(targets.shape)
            
            loss = loss_fn(outputs_for_loss, targets.to(device))
            loss.backward()
            optimizer.step()

            # Track some values to compute statistics
            total_loss += loss.item()

            all_targets.extend(targets)
            all_predictions.extend(outputs_for_loss.detach())

            # Save loss every 100 batches
            if (i % 100 == 0) and (i > 0):
                running_loss = total_loss / (i + 1)
                loss_history.append(running_loss)
            
            # #TODO: remove
            # if i == 0:
            #     break

        # NOTE: really all this is telling us is how many of the 40 sampled songs are labelled correctly
        acc = mean_squared_error(all_targets, all_predictions)
        final_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}, average train accuracy = {acc * 100:.3f}%")
        return acc, final_loss

    def _test(self, epoch, test_loader, device="cpu"):

        total_loss = 0
        all_predictions = []
        all_targets = []

        loss_fn = nn.MSELoss()

        self.model = self.model.to(device)
        self.model.eval()  # Set model in evaluation mode
        for i, inputs in enumerate(test_loader):
            # print(f"test: {i}")
            with torch.no_grad():
                inputs_x1, inputs_x2 = inputs['song_names'], inputs['artist_names']
                inputs_x1, inputs_x2 = inputs_x1.to(device), inputs_x2.to(device)
                
                outputs = self.model(inputs_x1, inputs_x2)

                ids, num_samples = getGroundTruthFromPlaylistID(i, self.playlistData)
                # create labels
                targets = torch.zeros(2 * num_samples)
                targets[:num_samples] = 1.0
                outputs_for_loss = torch.index_select(outputs, dim=1, index=torch.tensor(ids))
                targets = targets.unsqueeze(0)
                
                loss = loss_fn(outputs_for_loss, targets.to(device))

                # Track some values to compute statistics
                total_loss += loss.item()

                all_targets.extend(targets)
                all_predictions.extend(outputs_for_loss.detach())

            # #TODO: remove
            # if i == 0:
            #     break

        # NOTE: really all this is telling us is how many of the 40 sampled songs are labelled correctly
        acc = mean_squared_error(all_targets, all_predictions)
        final_loss = total_loss / len(test_loader)
        print(f"Epoch {epoch + 1} done. Average test loss = {final_loss:.2f}, average test accuracy = {acc * 100:.3f}%")
        return acc, final_loss

    def getPredictionsFromTracks(self, model_output, playlist_tracks, num_predictions):
        # 1: obtain an ID for each entry in the model output
        # 2: sort the model output in descending order
        sorted_song_IDs_asc = np.argsort(model_output)
        sorted_song_IDs = sorted_song_IDs_asc[::-1]

        # 3: obtain songs based on IDX
        predicted_tracks = getTracksFromSparseID(ids=sorted_song_IDs, songs=self.tracks)
        
        # 4: loop through until we have found 500 unique new songs
        curr_count = 0
        tracks_to_return = []
        while curr_count < num_predictions:
            for track in predicted_tracks:
                if track not in playlist_tracks:
                    tracks_to_return.append(track)
                curr_count += 1

        return tracks_to_return

    def predict(self, 
        X,
        num_predictions,
        embeddings_song=np.load("lib/song_names_test.npy"),
        embeddings_artist=np.load("lib/artist_names_test.npy")
    ):
        """
        Method used to collect predictions from model output and 
            X:                  the playlist we are doing predictions on
            numPredictions:     we are doing 500 for this challenge
            embeddings_file:    contains the embeddings that can be used to 
        NOTE: removed the ability to pass track set in and instead will always use all tracks. can add back if necessary
        """
        pid, pTracks = X["pid"], X["tracks"] #NOTE: we need to make sure that we update the playlists_test w the appropriate test playlists

        playlist_name_embeddings, playlist_song_embeddings = embeddings_song[pid], embeddings_artist[pid]

        model_output = self.model(playlist_name_embeddings, playlist_song_embeddings)
        # sort the song outputs
        predicted_songs = self.getPredictionsFromTracks(model_output, pTracks, num_predictions)

        return predicted_songs



class SiamNet(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_tracks):
        super(SiamNet, self).__init__()
        self.conv1 = nn.Conv1d(1857, 2048, 3, bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(2048, 2048, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(2048, 2048, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        # self.conv4 = nn.Conv1d(512, 1024, 3, bias=False)
        # self.BN4 = nn.BatchNorm1d(256)

        self.MLP =  nn.Sequential(
            nn.Linear(20480, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1857),
            nn.Sigmoid(),
        )

    def pass_through_CNN(self, x):
        x = x.type(torch.float32)
        residual_before = F.relu(self.conv1(x))
        # residual_before = F.relu(self.conv2(x)) #used to create residual connection

        x = F.relu(self.conv3_1(residual_before))
        residual_after = F.relu(self.conv3_2(x)) #used to create residual connection
        # concatenate to residual connection
        x = residual_after + residual_before
        # x = F.relu(self.BN4(self.conv4(x)))
        x = F.max_pool1d(x,2)
        
        return x

    def forward(self, x_songnames, x_artistnames):

        x1 = self.pass_through_CNN(x_songnames)
        x2 = self.pass_through_CNN(x_artistnames)

        # concatenate features
        x = x1 + x2

        x = x.reshape(-1, 20480)
        song_predictions = self.MLP(x)
        return song_predictions