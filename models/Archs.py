# Setting up different architectures to test
import pickle

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch import nn, Tensor
from torch.utils.data import Dataset

from util.helpers import getTracksFromSparseID, getGroundTruthFromPlaylistID

# This is the initial architecture developed by Maria and Sara
class Arch0(nn.Module):
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

# Task 1a: Adding more channels to the CNN
# Features: Channels starting at 64, increasing to 512, no batch norm, residual connections
# TODO: Run model
class Arch1a(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_channels, num_tracks):
        super(SiamNet, self).__init__()
        print(f"num channels: {num_channels}")
        print(f"num tracks: {num_tracks}")
        self.max_pool_kernel = 2
        self.num_tracks = num_tracks
        
        # NOTE look into if padding required.
        self.conv1 = nn.Conv1d(num_channels, 64, 3, padding="same", bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        # self.conv4 = nn.Conv1d(256, 512, 3, bias=False)
        # self.BN4 = nn.BatchNorm1d(256)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # NOTE Should work without *5
        self.MLP =  nn.Sequential(
            nn.Linear(256*(self.num_tracks//self.max_pool_kernel), num_tracks*2), #20480
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
        print(f"X shape after max pool: {x.shape}")

        return x

    def forward(self, x_songnames, x_artistnames):
        x1 = self.pass_through_CNN(x_songnames)
        x2 = self.pass_through_CNN(x_artistnames)

        # concatenate features
        x = x1 + x2

        #print("Shape before MLP: ", x.shape)
        x = x.reshape(-1, 256*(self.num_tracks//self.max_pool_kernel))
        song_predictions = self.MLP(x)
        #print("Shape after MLP: ", song_predictions.shape)
        return song_predictions

# Task 1b: Adding more channels to the CNN
# Features: Channels starting at 8, increasing to 256, no batch norm, residual connections
# TODO: Run this model

class Arch1b(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_channels, num_tracks):
        super(SiamNet, self).__init__()
        print(f"num channels: {num_channels}")
        print(f"num tracks: {num_tracks}")
        self.max_pool_kernel = 2
        self.num_tracks = num_tracks
        
        # NOTE look into if padding required.
        self.conv1 = nn.Conv1d(num_channels, 8, 3, padding="same", bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(8, 16, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(16, 32, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        # self.conv4 = nn.Conv1d(32, 64, 3, bias=False)
        # self.BN4 = nn.BatchNorm1d(256)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # NOTE Should work without *5
        self.MLP =  nn.Sequential(
            nn.Linear(32*(self.num_tracks//self.max_pool_kernel), num_tracks*2), #20480
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
        print(f"X shape after max pool: {x.shape}")

        return x

    def forward(self, x_songnames, x_artistnames):
        x1 = self.pass_through_CNN(x_songnames)
        x2 = self.pass_through_CNN(x_artistnames)

        # concatenate features
        x = x1 + x2

        #print("Shape before MLP: ", x.shape)
        x = x.reshape(-1, 32*(self.num_tracks//self.max_pool_kernel))
        song_predictions = self.MLP(x)
        #print("Shape after MLP: ", song_predictions.shape)
        return song_predictions


# Task 2a: Adding another CNN later
# TODO: Run this model
class Arch2a(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_channels, num_tracks):
        super(SiamNet, self).__init__()
        print(f"num channels: {num_channels}")
        print(f"num tracks: {num_tracks}")
        self.max_pool_kernel = 2
        self.num_tracks = num_tracks
        
        # NOTE look into if padding required.
        self.conv1 = nn.Conv1d(num_channels, 64, 3, padding="same", bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(256, 512, 3, bias=False)
        # self.BN4 = nn.BatchNorm1d(512)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # NOTE Should work without *5
        self.MLP =  nn.Sequential(
            nn.Linear(512*(self.num_tracks//self.max_pool_kernel), num_tracks*2), #20480
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
        print(f"X shape after adding residuals: {x.shape}")
        x = F.relu(self.conv4(x))
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
        x = x.reshape(-1, 512*(self.num_tracks//self.max_pool_kernel))
        song_predictions = self.MLP(x)
        #print("Shape after MLP: ", song_predictions.shape)
        return song_predictions

# Task 3a: Adding batch normalization to every layer

class Arch3a(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_channels, num_tracks):
        super(SiamNet, self).__init__()
        print(f"num channels: {num_channels}")
        print(f"num tracks: {num_tracks}")
        self.max_pool_kernel = 2
        self.num_tracks = num_tracks
        
        # NOTE look into if padding required.
        self.conv1 = nn.Conv1d(num_channels, 64, 3, padding="same", bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(256, 512, 3, bias=False)
        # self.BN4 = nn.BatchNorm1d(256)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

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
        
        residual_before = F.relu(self.bn1(self.conv1(x)))
        #print(f"X = residual before (conv 1): {residual_before.shape}")

        # residual_before = F.relu(self.conv2(x)) #used to create residual connection

        # x = self.bn2(F.relu(self.conv3_1(residual_before)))
        x = F.relu(self.bn2(self.conv3_1(residual_before)))
        #print(f"X 2nd conv: {x.shape}")
        
        
        # residual_after = self.bn3(F.relu(self.conv3_2(x))) #used to create residual connection
        residual_after = F.relu(self.bn3(self.conv3_2(x)))
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

# Task 3b: Adding batch normalization only to the output of the CNNs

class Arch3b(nn.Module):
    """
    SiameseNet used for computing predictions given different input features
    """
    def __init__(self, num_channels, num_tracks):
        super(SiamNet, self).__init__()
        print(f"num channels: {num_channels}")
        print(f"num tracks: {num_tracks}")
        self.max_pool_kernel = 2
        self.num_tracks = num_tracks
        
        # NOTE look into if padding required.
        self.conv1 = nn.Conv1d(num_channels, 64, 3, padding="same", bias=False)
        # self.BN1 = nn.BatchNorm1d(64)

        # self.conv2 = nn.Conv1d(256, 512, 3, bias=False)
        # # self.BN2 = nn.BatchNorm1d(128)

        self.conv3_1 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        # self.BN3_1 = nn.BatchNorm1d(128)
        self.conv3_2 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        # self.BN3_2 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(256, 512, 3, bias=False)
        self.BN4 = nn.BatchNorm1d(512)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # NOTE Should work without *5
        self.MLP =  nn.Sequential(
            nn.Linear(512*(self.num_tracks//self.max_pool_kernel), num_tracks*2), #20480
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
        
        residual_before = F.relu(self.bn1(self.conv1(x)))
        #print(f"X = residual before (conv 1): {residual_before.shape}")

        # residual_before = F.relu(self.conv2(x)) #used to create residual connection

        # x = self.bn2(F.relu(self.conv3_1(residual_before)))
        x = F.relu(self.bn2(self.conv3_1(residual_before)))
        #print(f"X 2nd conv: {x.shape}")
        
        
        # residual_after = self.bn3(F.relu(self.conv3_2(x))) #used to create residual connection
        residual_after = F.relu(self.bn3(self.conv3_2(x)))
        #print(f"X = residual after: {residual_after.shape}")
        # concatenate to residual connection
        x = residual_after + residual_before
        #print(f"X shape after adding residuals: {x.shape}")

        x = F.relu(self.conv4(x))
        x = self.BN4(x)
        x = F.max_pool1d(x,self.max_pool_kernel)
        #print(f"X shape after max pool: {x.shape}")

        return x

    def forward(self, x_songnames, x_artistnames):
        x1 = self.pass_through_CNN(x_songnames)
        x2 = self.pass_through_CNN(x_artistnames)

        # concatenate features
        x = x1 + x2

        #print("Shape before MLP: ", x.shape)
        x = x.reshape(-1, 512*(self.num_tracks//self.max_pool_kernel))
        song_predictions = self.MLP(x)
        #print("Shape after MLP: ", song_predictions.shape)
        return song_predictions
