import argparse, random
import pandas as pd

from models.NNeighClassifier import NNeighClassifier
from eval.Evaluate import Evaluate

class RecommendationEngine:
    """
    Args:
        numFiles (int): CLI variable that determines how many MPD files to read
        retrainNNC (bool): determines whether to retrain NNC or read from file

    Attributes:
        NNC (NNeighClassifier): NNeighbor Classifier used for predictions
        playlists (DataFrame): contains all playlists read into memory
        tracks (DataFrame): all tracks read into memory
        playlistSparse (scipy.CSR matrix) playlists formatted for predictions
    """
    def __init__(self, model_names):
        # Read in the relevant train, test data and features
        self.readData()

        # Initialize an empty model dictionary
        self.models = self.buildClassifiers(model_names)
    
    def readData(self):
        """
        Read song and playlist data from pickled dataframes. 
        """
        # Read data
        print("Reading data")
        self.playlists_all = pd.read_pickle("lib/playlists_all.pkl")
        self.playlists_train = pd.read_pickle("lib/playlists_train.pkl")
        self.playlists_test = pd.read_pickle("lib/playlists_test.pkl")
        self.tracks = pd.read_pickle("lib/tracks.pkl")
        self.playlistSparse = pd.read_pickle("lib/playlistSparse.pkl")
        print(f"Working with {len(self.playlists_train)} playlists " + 
            f"and {len(self.tracks)} tracks")

    def buildClassifiers(self, model_names):
        """
        Init classifiers and set initial classifier as main
        """
        # TODO: would eventually read in all model names and build dict of models
        return {
            'nnc': self.buildNNC()
        }

    def buildNNC(self): 
        """
        Init NNC classifier
        """
        return NNeighClassifier(
            sparsePlaylists=self.playlistSparse,
            tracks=self.tracks,
            playlists=self.playlists_all,
            reTrain=True) 
    
if __name__ == "__main__":
    # Prepare command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', help='Enter models you want delimited by ,', type=str)
    args = parser.parse_args()

    # Init class
    model_names = [str(model) for model in args.models.split(',')]
    print(f"Models we're comparing: {model_names}")

    # Initialize engine and train models 
    # These models get pickled and saved to the /lib folder 
    recommender = RecommendationEngine(model_names)

    # Evaluate the models 
    # # TODO: Eventually this should evaluate all models
    evaluator = Evaluate(
        tracks=recommender.tracks, 
        model=recommender.models['nnc'])
    print(evaluator.evaluate())