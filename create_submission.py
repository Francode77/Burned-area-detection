import numpy as np
import pandas as pd
import h5py 

from trimesh.voxel.runlength import dense_to_brle
from pathlib import Path
from collections import defaultdict

from typing import Any, Union, Dict, Literal
from numpy.typing import NDArray
import torch 
from make_predictions import MakePrediction

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# class RandomModel:
#     def __init__(self, shape):
#         self.shape = shape
#         return

#     def __call__(self, input):
#         # input is ignored, just generate some random predictions
#         return np.random.randint(0, 2, size=self.shape, dtype=bool)
    
class FixedModel:
    def __init__(self, shape) -> None:
        self.shape = shape        
        return
    
    def __call__(self, input_dict) -> Any:
        
        image=input_dict['post']
        _, prediction = MakePrediction.predict(image) 
        
        return prediction

def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue
            
            result[uuid]['post'] = values['post_fire'][...]
            # result[uuid]['pre'] = values['pre_fire'][...]

    return dict(result)

def compute_submission_mask(id: str, mask: NDArray):
    brle = dense_to_brle(mask.flatten())
    print (len(brle),np.arange(len(brle)))
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

def obtain_submission_df(submissions: Dict[str, NDArray]) -> pd.DataFrame:
    res = []
    for uuid, prediction in submissions.items():
      submission_mask = compute_submission_mask(uuid, prediction)
      res.append(pd.DataFrame(submission_mask))
    return pd.concat(res)

if __name__ == '__main__':
    validation_fold = retrieve_validation_fold('data/train_eval.hdf5')

    # use a list to accumulate results
    result = []
    # instantiate the model
    model = FixedModel(shape=(512, 512))
    for uuid in validation_fold:
        input_image = validation_fold[uuid]

        # perform the prediction
        predicted = model(input_image)
        # convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))
    print (type(result))
    
    #submission_df=obtain_submission_df(result)
    # concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv('predictions.csv', index=False)
    
    