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
    pass


class TransformerClassifier():
    # TODO: add more to init?
    def __init__(self):
        self.name = "Transformer"
        self.test_mapped_to_PID = pd.read_pickle("lib/test_playlist_mappings.pkl")     
    
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

    # TODO - Maria
    def getDataloaders():
        song_names_train = None #TODO
        song_names_test = None #TODO
        artist_names_train =None  #TODO
        artist_names_test = None #TODO

        train_dataset = PlaylistDataset(song_names_train, artist_names_train)
        test_dataset = PlaylistDataset(song_names_test, artist_names_test)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 5)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 5)
        return train_dataloader, test_dataloader
    
    # DONE
    def saveModel(self):
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
            print(inputs_x1.shape)
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

            all_targets.extend(targets.tolist())
            all_predictions.extend(outputs_for_loss.detach().tolist())

            # Save loss every 100 batches
            if (i % 100 == 0) and (i > 0):
                running_loss = total_loss / (i + 1)
                loss_history.append(running_loss)
            
        # NOTE: really all this is telling us is how many of the 40 sampled songs are labelled correctly
        acc = mean_squared_error(all_targets, all_predictions)
        final_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} done. Average train loss = {final_loss:.2f}, average train accuracy = {acc * 100:.3f}%")
        return acc, final_loss

    # TODO
    def getPredictionsFromTracks(self, model_output, playlist_tracks, num_predictions, track_mappings):
        # 1: obtain an ID for each entry in the model output
        # 2: sort the model output in descending order
        sorted_song_IDs_asc = np.argsort(model_output)

        # 3: obtain songs based on IDX
        predicted_tracks = None #TODO this will be from the dataframe containing lookup maps

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

    # TODO - Maria
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



class TransformerNet(nn.Module):
    """
    Transformer used for computing predictions given different input features
    """
    def _init_(self):
        super(TransformerNet, self).__init__()
        
        pass #TODO
    
    def forward(self): #TODO
        
        song_predictions = ...
        return song_predictions
        