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
from plotting import FieldPlotter 

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "resnet50"

class FixedModel:
    def __init__(self, shape, model_name) -> None:
        self.shape = shape     
        self.model_name = model_name
        return
    
    def __call__(self, input_dict, uuid) -> Any:
        image=input_dict['post']
        
        # Calculate the metric and the prediction mask
        metric, mask, prediction = MakePrediction.predict(image,self.model_name) 
        
        # Plot the result
        FieldPlotter.plot_submission(image, metric, mask, 2)
        
        return prediction

def retrieve_validation_fold(path: Union[str, Path]) -> Dict[str, NDArray]:
    result = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for uuid, values in fp.items():
            if values.attrs['fold'] != 0:
                continue 
            result[uuid]['post'] = values['post_fire'][...]
 
    return dict(result)

def compute_submission_mask(id: str, mask: NDArray):
    brle = dense_to_brle(mask.flatten()) 
    return {"id": id, "rle_mask": brle, "index": np.arange(len(brle))}

if __name__ == '__main__':

    validation_fold = retrieve_validation_fold('data/train_eval.hdf5')
    
    # use a list to accumulate results
    result = []     
    
    # Instantiate the model
    model = FixedModel(shape=(512, 512), model_name=MODEL_NAME)
 
    for uuid in validation_fold: 
        input_image = validation_fold[uuid]
        
        # Perform the prediction
        predicted = model(input_image,uuid)
        
        # Convert the prediction in RLE format
        encoded_prediction = compute_submission_mask(uuid, predicted)
        result.append(pd.DataFrame(encoded_prediction))
    
    # Concatenate all dataframes
    submission_df = pd.concat(result)
    submission_df.to_csv(f'predictions_{MODEL_NAME}.csv', index=False)
    
    