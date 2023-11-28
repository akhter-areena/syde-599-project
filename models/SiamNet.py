''' MARIA PLANNING
TODO:
- create CNN architecture
- write loss / masking loop
- write predict function

TODO: set torch manual seed somewhere in the calling loop

- flowchart + notes for dana and zach
'''

'''
    - song names
    - artist names
    - duration
'''
import os
import pickle

from torch import nn
import torch.nn.functional as F

# TODO: finish this implementation
class SiamNetClassifier():
    def __init__(self, playlists, sparsePlaylists, tracks, reTrain=False, name="SiamNetClassifier.pkl"):
        super(SiamNetClassifier, self).__init__()
        self.name = "SiamNet"
        self.pathName = name
        self.playlistData = sparsePlaylists
        self.playlists = playlists 
        self.tracks = tracks
        self.num_tracks = len(tracks) #TODO: figure out this datatype
        self.initModel(reTrain)

    def initModel(self, reTrain):
        libContents = os.listdir("lib")
        if self.pathName not in libContents or reTrain:
            print("Retraining model.")
            self.model = NearestNeighbors(
                n_neighbors=60,
                metric="cosine")
            self.trainModel(self.playlistData)
        else:
            print(f"Evaluating model that is stored at {self.pathName}")
            self.model = pickle.load(open(f"lib/{self.pathName}", "rb"))



class SiamNet(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_tracks):
        super(SiamNetClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, bias=False)
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