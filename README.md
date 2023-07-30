
We construct a method to train a model to detect areas of burned land from Sentinel 2 images. 
It can be run on a personal computer with a GPU. 

## Prerequisites

Pytorch needs to be installed with CUDA

## Installation
Create a conda environment with the libraries in `requirements.txt`
We have used python 3.10.11 and torch==2.0.1+cu117 

## Method

We can use the Sentinel 2 bands provided in the source files.

1. We mask areas with water using the NDWI index 
2. We calculate the ABAI index, which shows a good detection of burned land areas 
3. We detect areas where active fire occurs using the SAHM (Structural Analysis of Hydrologic Modeling) index and mask them out in the ABAI metric

4. We use image augmentation 
5. We use standard deviation to make the index more apparent across all images

6. We train a Resnet model on this metric by using Intersection over Union as score.


## Usage

run `preprocess_batch.py` 

 - loads the images from the .hd5f files 
 - calculates the metric with the functions in the Field class 
 - writes the metric into a .tiff file (source)
 - writes the truth into a .tiff file (target)


run `preview_batch.py`
Previews the images inside the .hdf5

run `train_model.py`
Training of the model with pytorch using unet architecture (resnet, xception_net, and deeplab_v3)

run `verify predictions.py`
Verifies the prediction with the truth

run `create_submission.py`
Prepares the .csv file to submit to the leaderboard

## Includes

`field.py`

Class with functions to 
    - calculate NDVI and other indexes
    - write the metric and mask 
    - create a water mask         
    - fire mask to exclude region with active fire 

`plotter.py`
Several functions to plot everything

`make_prediction.py`

 calculates the metric for an image and makes the prediction by using the selected model
    
     



## Results

We can visually verify how our model performs by looking at the plots.

For verfication we can run through the files with `verify_predictions.py`

![verification]('./assets/Figure 2023-07-30 183028.png')
On the left we see the original file in RGB
Next we see the result of our metric, yellow indicates burned land area
Next the resulting mask from our model
On the right the truth mask

## Limitations

Because the dataset has images where some burned areas have been contained in a previous pass by Sentinel satellites, they are also detected but not valid as truth. This imposed a problem that we coudln't overcome using this method.
A better approach would thus be to train the model on the provided bands without first using a metric.

Because the metric method proves reasonable results without this limitations, ie. if all burned areas would needed to be detected and not the ones from the labeled dataset, we were currently satisfied and didn't spend time and money on a more scientific approach. This would create a metric from the machine learning model's deep learning.
