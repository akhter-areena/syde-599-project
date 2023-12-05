import argparse
import pandas as pd
import numpy as np
from pprint import pprint

from models.NNeighClassifier import NNeighClassifier
# from models.SiamNet import SiamNetClassifier
from models.Transformer import TransformerClassifier
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
    def __init__(self, models, train):
        # Read in the relevant train, test data and features
        self.readData()
        self.train = train

        # Initialize an empty model dictionary
        self.models = self.buildClassifiers(models)
    
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
        if model_names[0] == "trans":
            return {
                'trans': self.buildTransNet(train=self.train)
            }
        print(model_names)
        raise RuntimeError("Should not be here")
        # return {
        #     # 'nnc': self.buildNNC(
        #     #     train=self.train
        #     # ),
        #     'cnn': self.buildSiamNet(
        #         train=self.train
        #     ),
        #     'trans': self.buildTransNet(train=self.train)
        # }

    def buildNNC(self, train): 
        """
        Init NNC classifier
        """
        return NNeighClassifier(
            sparsePlaylists=self.playlistSparse,
            tracks=self.tracks,
            playlists=self.playlists_all,
            reTrain=train
        ) 
    
    def buildSiamNet(self, train): 
        """
        Init NNC classifier
        """
        return SiamNetClassifier(
            playlists=self.playlists_all,
            sparsePlaylists=self.playlistSparseForCNN,
            tracks=self.tracks,
            reTrain=train
        ) 
    
    def buildTransNet(self, train):
        """
        Init Transformer Classifier
        """
        return TransformerClassifier(reTrain=train)
    
def main():
    # Prepare command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', help='Enter models you want delimited by ,', type=str)
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-f', '--first_test', action='store_true')
    args = parser.parse_args()

    model_names = [str(model) for model in args.models.split(',')]
    print(f"Models we're comparing: {model_names}")

    # Initialize recommendation engine.

    # If train is true or the model doesn't exist, we retrain the model, and it gets pickled and saved to the /lib folder.
    # If train is false, we load in the saved model, and we continue to evaluation.
    print(f"Training is set to: {args.train}")
    recommender = RecommendationEngine(
        models=model_names, 
        train=args.train)

    # Evaluate the models 
    #TODO: same as above, make the model choose between passed-in options
    # evaluator = Evaluate(
    #     tracks=recommender.tracks, 
    #     model=recommender.models['nnc']
    # )

    #TODO: DANA THIS NEEDS TO WORK SO WE GET EVALUATION METRICS FOR THE CURRENT PICKLED MODEL.
    evaluator = Evaluate(
        tracks=recommender.tracks,
        model=recommender.models['trans']
    )
    
    if args.first_test:
        evaluator.obscure_and_save()

    pprint(evaluator.evaluate())

if __name__ == "__main__":
    main()