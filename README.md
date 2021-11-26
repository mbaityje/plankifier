# Zoofier
Code for plankton dataset creation and classification based on Keras.

---

## Quick Summary

Use `analyze_dataset.py` to analyze the datasets.

Use `train.py` to train models on plankton data.

Use `train_from_saved.py`, to train models on saved data. For instance, you can use this command to reproduce our results using the data provided with the same train,test and valid split.

Use `predict.py`, to classify images using a single model.

Use `predict_ensemble.py`, to classify images that does not have labels using ensemble of models. 

Since keras gives lots of annoying messages, when launching in interactive shell, you might want to send redirect the stderr by appending `2>/dev/null` to your execution.


---

## Repo Structure

The repo contains the following directories:

- `src`: contains auxiliary code.

- `out`: the output is stored here (no output is uploaded to GitHub, so it must be created).

- `data`: the input data is stored here.

- `images`: contains a handful of images, so that a user can see them without downloading the repo.

- `trained-models`: contains only best trained models for the end users.


---

## Datasets

The datasets can be downloaded from https://doi.org/10.25678/0004DY.  

The datasets consists of two folders 1) ``zooplankton_0p5x`` and 2) ``Processed_Data``. If you want to train the classifiers from scratch then use the ``zooplankton_0p5x`` folder else if you want to use the same train, test and val split as we used for our publication then use the ``Processed_Data`` folder.

## Data organization

1) The directory `zooplankton_0p5x` contains one subdirectory per class, with the name of that class. In the directory related to each class, we have:

- A directory called `training_data`. This folder contains all the images related to the class.
- A tab-separated-value file, `features.tsv`, containing the morphological and color features. In addition to those, the first column (`url`) defines which image each line refers to, the second column (`timestamp`) describes when the image was acquired, and the third one is the file size.

2) The directory ``Processed_Data`` contains just a single pickle file. The pickle file can be loaded using pandas library, as shown here: 

```python
Data = pd.read_pickle(DataPath+'/Data.pickle')

# train data
trainFilenames=Data[0] 
X_train_image=Data[1]
y_train=Data[2]

# test data
testFilenames=Data[3]
X_test_image=Data[4]
y_test=Data[5]

#validation data
valFilenames=Data[6]
X_val_image=Data[7]
y_val=Data[8]

#features data
trainXfeat=Data[9]
testXfeat=Data[10]
valXfeat=Data[11]
```



### Analyzing the data

In order to get some rough information on the dataset, run 

```python
python analyze_dataset.py -datapath data/zooplankton_0p5x/
```

A summary of the dataset will be created (`./out//dataset_info//zooplankton.txt`, the output location and name can be personalized through the flags `-outpath` and `-name`), along with a couple of figures in that same directory, containing class abundances and image size distributions.

The same script can be used to make the joint analysis of two separate datasets:

```python
python analyze_dataset.py -datapath ./data/zooplankton_0p5x_1/ ./data/zooplankton_0p5x_2/
```

---

## Training models

In order to train a fresh model, use `train.py`. 

```python
python train.py
```
There are lots of input commands that can be given to the script. To query them, use the `-h` flag (`python train.py -h`). 

### Description of `train.py`

The script can be used both as a module to load the `Ctrain` class, or can be directly run, to train a model.

We describe the funcitoning of `train.py` through its command-line input flags.

The `train.py` can be used 

a) To either train a single model or a group of models.

b) To train a `binary ` , `versusall` or `multi` classifier

Also, using the parameters of `train.py` one can define the hyperparameters such as number of layers, learning rate, batch size etc. or one can do BayesianSearch for tuning all these hyperparameters.




##### I/O

`-datapaths`: is the path (or paths) leading to the directory (or directories) containing the data.

`-outpath`: is the path where we want the output to be written.

`-load_weights`: this flag allows us to initialize the system from a given weight configuration (`.hdf5` file). Loading is done through the `load_weights()` method of the model.

`-modelfile`: filename of the model we want to load (`.h5` file). Loading is done through the `keras.models.load_model()` method.

`-saveModelName`: Name of the model when it is saved.

`-save_data`: this allows to save the processed data including train, test and valid split so that it can be loaded directly during next runs 

`-saved_data`: this allows to use the saved data  for training the models and comparing different different since the same split of data can be used

`-compute_extrafeat`: this allows to compute additional morphological features

`-valid_set`: this allows to split the data either as train, test,valid or just train and test set

`-init_name`: this allows to set the name of the output directory for the current initial conditions

##### User Experience

`-verbose`: Print many messages on screen.

`-plot`: If activated, plots loss and accuracy during training once the run is over.

##### Data

`-L`: Images are resized to a square of (L x L) pixels.

`-testSplit`: Fraction of examples in the test set

`-classifier`: this allows to run either `binary`,`versusall` or `multi` classifier. The `binary` classifier should be used when you want to classify two classes. `versusall` classifier should be used, when one is interested in just one class compared to other classes. For example: `dinobryon` vs rest of the classes when you are interested in detecting `dinobryon` from the rest. `multi`classifier when one is interested in multi-class classification

`-class_select`: Classes to be looked at. If empty, all available classes are studied.

`-datakind`: Whether to load features (`'feat'`), images (`'image'`), or both (`'mixed'`) from the dataset.

`-ttkind`: Whether to load features (`'feat'`), images (`'image'`), or both (`'mixed'`) from the training and test set. This allows for example to load all the data with the `-datakind` flag, and then perform separate tests on subsets of the data without needing to reload the data.

`-balance_weight`: this allows to give higher weightage to the minority classes such that they are classified correctly. This should be used when there is a huge imbalance of classes in the data that you want to train the model on.

`-resize_images`: Images are resized to a square of LxL pixels by keeping the initial image proportions if resize=1. If resize=2, then the proportions are not kept but resized to match the user defined dimension

`-training_data`: This is to cope with the directory structures in the data.

##### Choosing models 

`-model_image`: For mixed data models, tells what model to use for the image branch. For image models, it is the whole model

`-model_feat`: For mixed data models, tells what model to use for the feature branch. For feat models, it is the whole model.

`-models_image`: Choose more than one models to train unlike `-model_image` parameter

`-stacking_ensemble`: For ensembling models using stacking option

`-Avg_ensemble`: For ensembling models using average option

`-only_ensemble`: this allows to run only ensemble models. This should be used when all the individual models are finished running

`-mixed_from_finetune`: this allows to train the mixed models using the trained finetuned image and feature models

`-mixed_from_notune`: this allows to train the mixed models using the trained image and feature models

`-mixed_from_scratch`: this allows to train the mixed models using the image and feature models from scratch.


##### Training time

`-totEpochs`: Total number of epochs for the training.

`-initial_epoch`: Initial epoch of the training.

`-earlyStopping`: If >0, we do early stopping, and this number is the patience (how many epochs without improving).

##### Setting hyperparameters manually

`-opt`: Optimizer. Currently, the two choices are `'sgd'` (with Nesterov momentum) and `'sgd'`.

`-bs`: Batch size.

`-lr`: Learning rate.

`-aug`: Boolean variable which decides whether or not to use augmentation. We are only augmenting for `'image'` models. The augmentation kind is currently hard-coded, and it amounts to 90 degrees rotations, vertical and horizontal flips, and 10% shears.

`-layers`: Layers for MLP models. 

##### Hypertuning using Bayesian search

`-hp_tuning`: this allows to do the hyperparameter tuning based on bayesian search

`-max_trials`: number of trials for running bayesian search hypertuning

`-executions_per_trial`: number of executions per trial. 

`-bayesian_epoch`: number of epochs for bayesian search. 

`-finetune`: this allows to finetune the models

`-epochs`: number of epochs to train the models using the best hyperparameters selected from bayesian search

`-finetune_epochs`: number of epochs for finetuning the model


#### An example of training a model
```python
python train.py -datapaths ./data/zooplankton_0p5x/ ./trained-models/Aquascope_Model/ -classifier multi -aug -datakind mixed -ttkind mixed -save_data yes -resize_images 2 -L 128 -finetune 1 -valid_set yes -training_data True -hp_tuning yes -models_image eff0 eff1 eff2 eff3 eff4 eff5 eff6 eff7 incepv3 res50 dense121 mobile -max_trials 10 -executions_per_trial 1  -compute_extrafeat yes -avg_ensemble yes -bayesian_epoch 100 -stacking_ensemble yes -epochs 200 -finetune_epochs 400 -balance_weight no -mixed_from_finetune 1 -init_name Initialization_01
```

---

Running above command would train all the models that we present in our publication.

If the reader wants to train and test the models on the exact same split of data as we used in our publication so as to reproduce the results then `train_from_saved.py` script should be run.

```python
python train_from_saved.py -datapaths ./data/zooplankton_0p5x/ ./trained-models/Aquascope_Model/ -classifier multi -aug -datakind mixed -ttkind mixed -saved_data yes -resize_images 2 -L 128 -finetune 1 -valid_set yes -training_data True -hp_tuning yes -models_image eff0 eff1 eff2 eff3 eff4 eff5 eff6 eff7 incepv3 res50 dense121 mobile -max_trials 10 -executions_per_trial 1  -compute_extrafeat yes -avg_ensemble yes -bayesian_epoch 100 -stacking_ensemble yes -epochs 200 -finetune_epochs 400 -balance_weight no -mixed_from_finetune 1 -init_name Initialization_01
```



### Main Classes

`Ctrain`: contained in `train.py`.
Class for training.

`Cdata`: contained in `src/helper_data.py`.
Class for data loading.

`CTrainTestSet`: contained in `src/helper_tts.py`.
Class for dataset curation.

`CModelWrapper`: contained in `src/helper_models.py`
Wrapper class for models.

`Cval`: contained in `validation.py`
Class for validation.

`Censemble`: contained in `predict.py`
Class for applying ensemble methods.

---

## Making predictions

1) Predictions using the **individual models** 

These are made through the `predict.py`, that loads one or more models, and makes predictions on the requested directories.

```python
Example usage of predict.py

python predict.py  -modelfullnames './trained-models/eff0/keras_model_finetune.h5' -weightnames 
'./trained-models/eff0/bestweights_finetune.hdf5' -testdirs './data/Centric_Diatoms/' -predname './out//'
```


```python
python predict.py   -modelfullnames './trained-models/conv2/keras_model.h5' \  # Names of the models to be loaded
                    -weightnames './trained-models/conv2/bestweights.hdf5' \   # For each model, weights to be loaded
                    -testdirs 'data/zooplankton_0p5x/validation/images/dinobryon/' \ # dirs with images to be checked
                    -thresholds 0.6 \ # Abstention thresholds
                    -ensMethods 'unanimity' \ # Ensembling thresholds. Useless in this example, since there is only one model
                    -predname './out/predictions/predict' # Send output to ./out/predictions/predict.txt
```




Explanation of the used parameters:

- `modelfullnames`: Names of the models to be loaded
- `weightnames`: For each model, weights to be loaded
- `testdirs`:dirs with images to be checked
- `thresholds`: Abstention thresholds (if confidence is lower than threshold, label as uncertain). Useless in this example, since there is only one model
- `ensMethods`: Ensembling thresholds. Useless in this example, since there is only one model
- `predname`: Send output to ./out/predictions/predict.txt



2) Predictions using the **ensemble models**

These are made through the `predict_ensemble.py` .

```python
python predict_ensemble.py  -testdir './data/zooplankton_0p5x_all/validation/counts/year_2018/15304*/0000000000_subset_static_html/' -model_path '/local/kyathasr/plankifier/trained-models/Aquascope_corrected_Data_with_Mixed/' -models_image eff2 mobile -init_names Iteration_01 Iteration_02 -path_to_save './out/Ensemble_prediction3' -ens_type 1
```

Explanation of the used parameters:

- `testdir`: dirs with images to be checked
- `modelpath`: main path for the trained models
- `models_image`: select the model names from which the ensemble to be formed and used for predicition
- `init_names`: Initializations names corresponding to the selected models. 
- `path_to_save`: path to save the predictions
- `ens_type`: to choose either average ensemble models, stacked ensemble models or both
