import random
import pandas as pd
import math

class Evaluate:
    def __init__(self, tracks, model) -> None:
        self.all_tracks = tracks
        self.model = model
        self.playlists_test = pd.read_pickle("lib/playlists_test.pkl").head(10)
        self.percentsToObscure = [0.1, 0.25, 0.5, 0.75, 0.95]

    def obscurePlaylist(self, playlist, percent): 
        """
        Obscure a portion of a playlist's songs for testing
        Return that playlist and its obscured tracks 
        """
        orig_tracks = playlist['tracks']
        k = int(len(orig_tracks) * percent)
        indices = random.sample(range(len(orig_tracks)), k)
        obscured = [orig_tracks[i] for i in indices]
        tracks = [i for i in orig_tracks + obscured if i not in orig_tracks or i not in obscured]
        return obscured, tracks

    '''
    This metric rewards total number of retrieved relevant tracks (regardless of order).
    '''
    def computeRPrecision(self, predictions, ground_truth):
        overlap = [value for value in predictions if value in ground_truth]
        return len(overlap)/len(ground_truth)

    '''
    This metric rewards total number of retrieved relevant tracks (regardless of order).
    '''
    def computeNDCG(self, predictions, ground_truth):
        # Check if first prediction is in ground truth to initialize DCG 
        dcg = 1 if predictions[0] in ground_truth else 0
        for i in range(2, len(predictions)):
            if predictions[i] in ground_truth:
                dcg +=  1/(math.log2(i))

        # Now calculate IDCG
        overlap = [value for value in predictions if value in ground_truth]
        idcg = 1
        for i in range(2, len(overlap)):
            idcg += 1/(math.log2(i))

        # NDCG is the DCG/IDCG
        return dcg/idcg

    '''
    This measures how many "clicks" (lists of 10 songs) that the user would have 
    to look through to find a "relevant" or ground truth song.
    '''
    def computeRecommendedSongsClicks(self, predictions, ground_truth):
        num_clicks = 0
        for i in range(0, len(predictions), 10):
            currTen = predictions[i:i+10]

            # If any of the current 10 songs overlap with the ground_truth, return
            overlap = [value for value in currTen if value in ground_truth]
            if len(overlap) > 0:
                return num_clicks
            
            num_clicks += 1
        
        # If we go through all the number of clicks, we return 51
        return 51

    def computeMetrics(self, playlist, percent):
        # Create a copy of the old playlist's metadata, and update its tracks 
        playlistSub = playlist.copy()
        obscured = set(playlist[f'obscuredTracks@_{percent}'])
        playlistSub['tracks'] = set(playlist[f'remainingTracks@_{percent}'])

        # Pass that "new playlist" into the model's predictor
        predictions = self.model.predict(
            playlistSub, 
            500, 
            self.all_tracks)

        rPrecision = self.computeRPrecision(predictions, obscured)
        ndcg = self.computeNDCG(predictions, obscured)
        recommendedSongsClicks = self.computeRecommendedSongsClicks(predictions, obscured)

        return rPrecision, ndcg, recommendedSongsClicks

    def evaluate(self):
        print(f"Evaluating model on {len(self.playlists_test)} test playlists.")
       
        # Intialize output structure
        metrics = {}
        for percent in self.percentsToObscure:
            metrics[f'obscured@_{percent}'] = {}

        # Compute metrics at each level of obscurity
        for percent in self.percentsToObscure:
            # Generate obscured playlist and remaining (GT) tracks
            self.playlists_test[[f'obscuredTracks@_{percent}', f'remainingTracks@_{percent}']] = self.playlists_test.apply(self.obscurePlaylist, args=(percent,), axis=1, result_type="expand")
            
            # Compute metrics for each playlist
            self.playlists_test[[f'rPrecision@_{percent}', f'NDCG@_{percent}', f'recommendedSongsClicks@_{percent}']] = self.playlists_test.apply(self.computeMetrics, args=(percent,), axis=1, result_type="expand")
        
            # Agreggate metrics across the entire test dataset 
            metrics[f'obscured@_{percent}']['rPrecision'] = self.playlists_test[f'rPrecision@_{percent}'].mean()
            metrics[f'obscured@_{percent}']['NDCG'] = self.playlists_test[f'NDCG@_{percent}'].mean()
            metrics[f'obscured@_{percent}']['recommendedSongsClicks'] = self.playlists_test[f'recommendedSongsClicks@_{percent}'].mean()

            print(self.playlists_test.head(10))

        return metrics