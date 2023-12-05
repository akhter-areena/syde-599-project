import os
import pickle

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch import nn, Tensor
from torch.utils.data import Dataset

from util.helpers import getTracksFromSparseID, getGroundTruthFromPlaylistID, getGroundTruth


# Note would need to change if added more features
class PlaylistDataset(Dataset):
    def __init__(self, song_names, artist_names, targets):
        self.song_names = song_names
        self.artist_names = artist_names
        self.targets = targets

    def __len__(self):
        return len(self.song_names)

    def __getitem__(self, playlist_idx):
      song_names_torch = torch.tensor(self.song_names)
      aritst_names_torch = torch.tensor(self.artist_names)
      targets_torch = torch.tensor(self.targets)

      return {"song_names": song_names_torch[playlist_idx], 'artist_names': aritst_names_torch[playlist_idx], 'targets': targets_torch[playlist_idx]}


class SiamNetClassifier():
    def __init__(self, playlists, sparsePlaylists, tracks, reTrain=False, name="SiamNetClassifier.pkl", device="cpu"):
        self.name = "SiamNet"
        self.pathName = name
        self.playlistData = sparsePlaylists
        self.playlists = playlists 
        self.tracks = tracks
        self.num_tracks = len(tracks)
        print(f"tracks dimension: {self.tracks}")
        # NOTE: make this not hardcoded
        self.train_mapped_to_PID = pd.read_pickle("lib/train_playlist_mappings.pkl")
        self.test_mapped_to_PID = pd.read_pickle("lib/test_playlist_mappings.pkl")
       
        self.initModel(reTrain, device=device)


    def initModel(self, reTrain, device="cpu"):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            print("Retraining model.")
            train_dataloader, test_dataloader = self.getDataloaders()
            
            # Get shape of batches
            shape = None
            for i, batch in enumerate(train_dataloader):
                shape = batch['song_names'].shape
                print(f"Data shape: {shape}")
                if batch['song_names'].shape != batch['artist_names'].shape:
                    print(f"song dim: {batch['song_names'].shape}")
                    print(f"artist dim: {batch['artist_names'].shape}")
                    raise RuntimeError("Feature length are not the same")
                break
            self.model = SiamNet(
                num_channels=shape[-1],
                num_tracks=shape[-2]
            )
            
            print("about to get dataloaders")
            print("done getting dataloaders")
            self.trainModel(train_dataloader, test_dataloader, device=device)
        else:
            print(f"Evaluating model that is stored at {self.pathName}")
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))


    def getDataloaders(self):
        song_names_train = np.load("lib/song_names_train.npy")
        song_names_test = np.load("lib/song_names_test.npy")
        artist_names_train = np.load("lib/artist_names_train.npy")
        artist_names_test = np.load("lib/artist_names_test.npy")
        
        # NOTE This is not a great solution
        playlist_target_train = []
        for i in range(len(song_names_train)):
            playlist_target_train.append([])
            for song_embeddings in song_names_train[i]:
                if all(val == -5 for val in song_embeddings):
                    playlist_target_train[i].append(float(0))
                else:
                    playlist_target_train[i].append(float(1))
                    
        if len(playlist_target_train)!=25 and len(playlist_target_train[0])!= 1857:
            print(len(playlist_target_train))
            print(len(playlist_target_train[0]))
            raise RuntimeError("WTF")
                
        playlist_target_test = []
        for i in range(len(song_names_test)):
            playlist_target_test.append([])
            for song_embeddings in song_names_test[i]:
                if all(val == -5 for val in song_embeddings):
                    playlist_target_test[i].append(0.0)
                else:
                    playlist_target_test[i].append(1.0)
        
        # artist_names_train = np.load("lib/song_names_train.npy")
        # artist_names_test = np.load("lib/song_names_test.npy")

        train_dataset = PlaylistDataset(song_names_train, artist_names_train, playlist_target_train)
        test_dataset = PlaylistDataset(song_names_test, artist_names_test, playlist_target_test)
        print(f"Song names size: {len(song_names_test[0])}")
        print(f"Artist names size: {len(artist_names_test[0])}")

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 5, shuffle = False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
        return train_dataloader, test_dataloader

    def trainModel(self, train_dataloader, test_dataloader, num_epochs=10, device='cpu'):
        train_losses = []
        test_losses = []
        train_metrics = []
        test_metrics = []

        print("in trainModel")
        for epoch in range(num_epochs):
            acc, final_loss = self._train(epoch, train_dataloader, device=device)
            acc, final_loss = self._test(epoch, test_dataloader, device=device)

            # train_losses.append(train_loss)
            # test_losses.append(test_loss)
            # train_metrics.append(train_acc)
            # test_metrics.append(test_acc)
        
        self.saveModel()

        return train_losses, train_metrics, test_losses, test_metrics
        #NOTE returning these in case we wish to plot the training curves during development

    def saveModel(self):
        """
        """
        pickle.dump(self.model, open(f"lib/{self.pathName}", "wb"))

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
        loss_fn = nn.CrossEntropyLoss()

        #print(f"about to do model.to({device})")
        self.model = self.model.to(device)
        #print("about to do model.train()")
        self.model.train()
        print("completed training")
        
        ''' TRAIN '''
        print("about to enumerate dataloader")
        #print(type(train_loader))
        #print(len(train_loader))
        
        acc_list = []
        for i, inputs in enumerate(train_loader):
            # print(f"train: {i}")
            print("begin inner loop")
            optimizer.zero_grad()
            # print("about to do inputs.to(device)")
            inputs_x1, inputs_x2 = inputs['song_names'], inputs['artist_names']
            #print(inputs_x1.shape)
            #print(f"inputs_x1.to({device}), inputs_x2.to({device})")
            inputs_x1, inputs_x2 = inputs_x1.to(device), inputs_x2.to(device)
            # print("about to do forward pass")
            outputs = self.model(inputs_x1, inputs_x2)
            #print(f"output shape: {outputs.shape}")
            
            # print("done forward pass")

            # we wish to fit to some of the ground truth, but not every single
            #   entry since the whole idea is that we can be learning new 
            #   relationships between songs that may not be in the original
            # dataloader returns the playlist embeddings in order as they appear
            #   in the sparse matrix, so we can leverage that in order to obtain
            #   their ground truth labels
            # print(f"Input type: {inputs}")
            ids, num_samples = getGroundTruthFromPlaylistID(i, self.playlistData)
            #print(f"Length ids {len(ids)}")
            # create labels
            
            targets = torch.zeros(2 * num_samples)
            targets[:num_samples] = 1.0
            outputs_for_loss = torch.index_select(outputs, dim=1, index=torch.tensor(ids).to(device)).to(device)
            #print("output loss:", outputs_for_loss.shape)
            #print("target shape:", targets.shape)
            targets = targets.unsqueeze(0)
            targets = inputs["targets"].to(device)
            #print("target shape:", targets.shape)
            
            loss = loss_fn(outputs, targets)
            # loss = loss_fn(outputs, getGroundTruth(i, self.playlistData))
            print(f"Loss: {loss.shape}")
            loss.backward()
            optimizer.step()

            # Track some values to compute statistics
            total_loss += loss.item()
            #print(f"targets repeated {targets.repeat(5,1).shape}")

            
            # all_targets.extend(targets.repeat(5,1).tolist())
            # all_predictions.extend(outputs_for_loss.detach().tolist())

            # Save loss every 100 batches
            if (i % 100 == 0) and (i > 0):
                running_loss = total_loss / (i + 1)
                loss_history.append(running_loss)
                
            acc_list.append(mean_squared_error(targets.tolist(), outputs.detach().tolist()))
        
        acc = sum(acc_list)/len(acc_list)
        # NOTE: really all this is telling us is how many of the 40 sampled songs are labelled correctly
        # print(f"target: {len(all_targets)}, prediction: {len(all_predictions)}")
        # acc = mean_squared_error(all_targets, all_predictions)
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
        
        acc_list = []
        for i, inputs in enumerate(test_loader):
            #print(f"test: {i}")
            with torch.no_grad():
                inputs_x1, inputs_x2 = inputs['song_names'], inputs['artist_names']
                inputs_x1, inputs_x2 = inputs_x1.to(device), inputs_x2.to(device)
                
                outputs = self.model(inputs_x1, inputs_x2).to(device)

                ids, num_samples = getGroundTruthFromPlaylistID(i, self.playlistData)
                # create labels
                targets = torch.zeros(2 * num_samples).to(device)
                targets[:num_samples] = 1.0
                outputs_for_loss = torch.index_select(outputs, dim=1, index=torch.tensor(ids).to(device))
                targets = targets.unsqueeze(0)
                
                targets = inputs["targets"].to(device)
                
                loss = loss_fn(outputs, targets)

                # Track some values to compute statistics
                total_loss += loss.item()

                # all_targets.extend(targets)
                # all_predictions.extend(outputs_for_loss.detach())
                
                acc_list.append(mean_squared_error(targets.tolist(), outputs.detach().tolist()))

        acc = sum(acc_list)/len(acc_list)
        # NOTE: really all this is telling us is how many of the 40 sampled songs are labelled correctly
        final_loss = total_loss / len(test_loader)
        print(f"Epoch {epoch + 1} done. Average test loss = {final_loss:.2f}, average test accuracy = {acc * 100:.3f}%")
        return acc, final_loss

    def getPredictionsFromTracks(self, model_output, playlist_tracks, num_predictions, track_mappings):
        # 1: obtain an ID for each entry in the model output
        # 2: sort the model output in descending order
        sorted_song_IDs_asc = np.argsort(model_output)

        # 3: obtain songs based on IDX
        predicted_tracks = getTracksFromSparseID(ids=sorted_song_IDs_asc[0], songs=self.tracks, track_mappings=track_mappings)
        
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

    def predict(self, 
        X,
        num_predictions,
        embeddings_song=np.load("lib/song_names_test.npy"),
        embeddings_artist=np.load("lib/artist_names_test.npy"),
        track_mappings = pd.read_pickle("lib/mappings_pd.pkl")
    ):
        """
        Method used to collect predictions from model output and 
            X:                  the playlist we are doing predictions on
            numPredictions:     we are doing 500 for this challenge
            embeddings_file:    contains the embeddings that can be used to 
        NOTE: removed the ability to pass track set in and instead will always use all tracks. can add back if necessary
        """
        pid, pTracks = X["pid"], X["tracks"]

        print(pid)

        pid_internal = self.test_mapped_to_PID[self.test_mapped_to_PID['Playlist Df Id'] == pid]['Matrix id']

        playlist_name_embeddings, playlist_song_embeddings = embeddings_song[int(pid_internal)], embeddings_artist[int(pid_internal)]
        playlist_name_embeddings = torch.tensor(playlist_name_embeddings)
        playlist_song_embeddings = torch.tensor(playlist_song_embeddings)

        self.model.eval()
        model_output = self.model(playlist_name_embeddings, playlist_song_embeddings)
        model_output = model_output.detach()
        # sort the song outputs
        predicted_songs = self.getPredictionsFromTracks(model_output, pTracks, num_predictions, track_mappings)

        return predicted_songs


class SiamNet(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_channels, num_tracks):
        super(SiamNet, self).__init__()
        print(f"num channels: {num_channels}")
        print(f"num tracks: {num_tracks}")
        self.second_channels = num_channels*2
        self.max_pool_kernel = 2
        self.num_tracks = num_tracks
        
        # NOTE look into if padding required.
        self.conv1 = nn.Conv1d(num_channels, self.second_channels, 3, padding="same", bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(self.second_channels, self.second_channels, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(self.second_channels, self.second_channels, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        # self.conv4 = nn.Conv1d(512, 1024, 3, bias=False)
        # self.BN4 = nn.BatchNorm1d(256)
        
        self.bn1 = nn.BatchNorm1d(self.second_channels)
        self.bn2 = nn.BatchNorm1d(self.second_channels)
        self.bn3 = nn.BatchNorm1d(self.second_channels)

        # NOTE Should work without *5
        self.MLP =  nn.Sequential(
            nn.Linear(self.second_channels*(self.num_tracks//self.max_pool_kernel), num_tracks*2), #20480
            nn.ReLU(),

            nn.Linear(num_tracks*2, num_tracks), #1857 is the num tracks
            nn.Sigmoid(),
        )
        print("Siam Initalized")

    def pass_through_CNN(self, x):
        # import torch.nn.functional as F
        # NOTE repeats for MLP fitment due to batch convergence
        # x = x.repeat(5,1,1) #NOTE: delete this for training
        x = x.permute((0, 2, 1))
        #print(f"Pass through x: {x.shape}")
        x = x.type(torch.float32)
        # residual_before = self.bn1(F.relu(self.conv1(x)))
        
        residual_before = F.relu(self.conv1(x))
        #print(f"X = residual before (conv 1): {residual_before.shape}")

        # residual_before = F.relu(self.conv2(x)) #used to create residual connection

        # x = self.bn2(F.relu(self.conv3_1(residual_before)))
        x = F.relu(self.conv3_1(residual_before))
        #print(f"X 2nd conv: {x.shape}")
        
        
        # residual_after = self.bn3(F.relu(self.conv3_2(x))) #used to create residual connection
        residual_after = F.relu(self.conv3_2(x))
        #print(f"X = residual after: {residual_after.shape}")
        # concatenate to residual connection
        x = residual_after + residual_before
        #print(f"X shape after adding residuals: {x.shape}")
        # x = F.relu(self.BN4(self.conv4(x)))
        x = F.max_pool1d(x,self.max_pool_kernel)
        #print(f"X shape after max pool: {x.shape}")

        return x

    def forward(self, x_songnames, x_artistnames):
        x1 = self.pass_through_CNN(x_songnames)
        x2 = self.pass_through_CNN(x_artistnames)

        # concatenate features
        x = x1 + x2

        #print("Shape before MLP: ", x.shape)
        x = x.reshape(-1, self.second_channels*(self.num_tracks//self.max_pool_kernel))
        song_predictions = self.MLP(x)
        #print("Shape after MLP: ", song_predictions.shape)
        return song_predictions