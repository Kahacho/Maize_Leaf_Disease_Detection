# Maize Disease Classification

## Introduction
* Classifying maize disease using CNN and PyTorch.

## Running the script
* Install the dependencies by running the command: `pip install -r requirements.txt`
* To start the app, run the command `uvicorn main:app`

## Training the CNN model
* Download data from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GDON8Q

* Extract the archived datasets, and copy them to `\model_training\maize_dataset` directory. 
However, copy the images in `MSV_2` to `MSV_1`, and rename `MSV_1` to `MSV`.

* Run the `model_training.ipynb` script to see the training of a CNN model in action.