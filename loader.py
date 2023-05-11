from typing import List, Tuple

import h5py
import numpy as np


def create_folds(hdf5_file: str, folds: int) -> None: 
    # Read hdf5 file and create fold attribute with value
    with h5py.File(hdf5_file, "r+") as f:
        # count the number of folds
        total_folds=0
        max_idx=0
        for idx, (uuid, values) in enumerate(f.items()): 
            values.attrs.create("fold", idx//folds , dtype=np.int64) 
            if values.attrs["fold"] > total_folds:
                total_folds = values.attrs["fold"]
            max_idx= idx
    print (str(hdf5_file),'folds:' , total_folds , 'items:', max_idx)

def loader(
    hdf5_file: str, folds: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    post = []
    pre = []
    masks = []
    names = []

    # Read hdf5 file and filter by fold
    with h5py.File(hdf5_file, "r") as f:
        for uuid, values in f.items():
            if values.attrs["fold"] not in folds:
                continue
            if "pre_fire" not in values:
                continue

            post.append(values["post_fire"][...])
            pre.append(values["pre_fire"][...])
            masks.append(values["mask"][...])
            names.append(uuid)

    # Convert to numpy arrays
    post = np.stack(post, axis=0, dtype=np.int32)
    pre = np.stack(pre, axis=0, dtype=np.int32)
    masks = np.stack(masks, axis=0, dtype=np.int32)

    return post, pre, masks, names
