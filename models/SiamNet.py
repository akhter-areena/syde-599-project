'''
    - song names
    - artist names
    - duration
'''
import os
import pickle

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn

from util.helpers import getTracksFromSparseID

# TODO: finish this implementation
class SiamNetClassifier():
    def __init__(self, playlists, sparsePlaylists, tracks, reTrain=False, name="SiamNetClassifier.pkl"):
        super(SiamNetClassifier, self).__init__()
        self.name = "SiamNet"
        self.pathName = name
        self.playlistData = sparsePlaylists
        self.playlists = playlists 
        self.tracks = tracks
        self.num_tracks = len(tracks)
        self.initModel(reTrain)

    def initModel(self, reTrain):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            print("Retraining model.")
            self.model = SiamNet(
                num_tracks=self.num_tracks,
            )
            self.trainModel(self.playlistData)
        else:
            print(f"Evaluating model that is stored at {self.pathName}")
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))

    def trainModel(self, num_epochs=50, device='cpu'):
        # TODO: sampling of 20 songs in playlist, 20 songs not for computing loss

        train_losses = []
        test_losses = []
        train_metrics = []
        test_metrics = []

        for epoch in range(num_epochs):
            train_acc, train_loss = self._train(epoch)
            test_acc, test_loss = self._test(epoch)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_metrics.append(train_acc)
            test_metrics.append(test_acc)
        
        return train_losses, train_metrics, test_losses, test_metrics
        #NOTE returning these in case we wish to plot the training curves during development

    def _train(self, epoch, device="cpu"):
        total_loss = 0
        all_predictions = []
        all_targets = []
        loss_history = []

        # NOTE: these aren't handled very tastefully, maybe parameterize??
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 1e-5
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()

        # TODO dataloader
        train_loader = None

        self.model = self.model.to(device)
        self.model.train()
        
        ''' TRAIN '''
        for i, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            outputs = self.model(inputs)

            # we wish to fit to some of the ground truth, but not every single
            #   entry since the whole idea is that we can be learning new 
            #   relationships between songs that may not be in the original
            # dataloader returns the playlist embeddings in order as they appear
            #   in the sparse matrix, so we can leverage that in order to obtain
            #   their ground truth labels
            # TODO: confirm with sara that she used the same train-test split as areena


            loss = loss_fn(outputs, targets.to(device))
            loss.backward()
            optimizer.step()

            # Track some values to compute statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            all_predictions.extend(preds.detach().cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            # Save loss every 100 batches
            if (i % 100 == 0) and (i > 0):
                running_loss = total_loss / (i + 1)
                loss_history.append(running_loss)

        # NOTE: really all this is telling us is how many of the 40 sampled songs are labelled correctly
        acc = accuracy_score(all_targets, all_predictions)
        final_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}, average train accuracy = {acc * 100:.3f}%")
        pass

    def _test(self, epoch, device="cpu"):

        total_loss = 0
        all_predictions = []
        all_targets = []

        loss_fn = nn.CrossEntropyLoss()

        # TODO: dataloader
        test_loader = None

        model = model.to(device)
        model.eval()  # Set model in evaluation mode
        for i, inputs in enumerate(test_loader):
            with torch.no_grad():
                outputs = model(inputs.to(device))
                loss = loss_fn(outputs, targets.to(device))

                # Track some values to compute statistics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=-1)
                all_predictions.extend(preds.detach().cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        acc = accuracy_score(all_targets, all_predictions)
        final_loss = total_loss / len(test_loader)
        print(f"Epoch {epoch + 1} done. Average test loss = {final_loss:.2f}, average test accuracy = {acc * 100:.3f}%")
        return acc, final_loss
        pass

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

    def predict(self, X, num_predictions):
        """
        Method used to collect predictions from model output and 
            X:              the playlist we are doing predictions on
            numPredictions: we are doing 500 for this challenge
        NOTE: removed the ability to pass track set in and instead will always use all tracks. can add back if necessary
        """
        playlist = X
        pTracks = playlist["tracks"]
        # TODO: (ASK SARA) obtain playlist input data somehow
        playlist_data = None # placeholder for now

        model_output = self.model(playlist_data)
        # sort the song outputs
        predicted_songs = self.getPredictionsFromTracks(model_output, pTracks, num_predictions)

        return predicted_songs



class SiamNet(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_tracks):
        super(SiamNetClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, bias=False) # TODO: (ASK SARA) change these values
        self.BN1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 3, bias=False)
        self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=1, bias=False)
        self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=1, bias=False)
        self.BN3_2 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 3, bias=False)
        self.BN4 = nn.BatchNorm1d(256)

        self.MLP =  nn.Sequential(
            nn.Linear(256*283, 2048),
            nn.ReLU(),

            nn.Linear(2048, 256),
            nn.ReLU(),

            nn.Linear(256, num_tracks),
        )

    def pass_through_CNN(self, x):
        x = F.relu(self.conv1(x))
        residual_before = F.relu(self.conv2(x)) #used to create residual connection

        x = F.relu(self.conv3_1(residual_before))
        residual_after = F.relu(self.conv3_2(x)) #used to create residual connection
        
        # concatenate to residual connection
        x = residual_after + residual_before
        x = F.relu(self.BN4(self.conv4(x)))
        x = F.max_pool1d(x, 2)
        
        return x

    def forward(self, x_songnames, x_artistnames, x_duration):

        x1 = self.pass_through_CNN(x_songnames)
        x2 = self.pass_through_CNN(x_artistnames)
        x3 = self.pass_through_CNN(x_duration)
        
        # concatenate features
        x = x1 + x2 + x3

        x = x.reshape(-1, 256*283)
        song_predictions = self.MLP(x)
        return song_predictions