# plankifier
Code for plankton dataset creation and classification based on Keras.

End users (field scientists) should refer to the usage guide provided in the [code releases](https://github.com/mbaityje/plankifier/releases).

For the necessary installations and a quick guide to install a virtual environment, use the script `./scripts/installation.sh`.


---

## Quick Summary

To analyze the datasets, use `analyze_dataset.py`.

To train, use `train.py`, that trains a model on our plankton data.

To predict, use `predictions.py`, that classifies images with the option of using ensemble methods.

To validate, use `validation.py` and `validation-counts.py`, which perform two kinds of validation.

Since keras gives lots of annoying messages, when launching in interactive shell, you might want to send redirect the stderr by appending `2>/dev/null` to your execution.


### Examples

Examples of usage are contained in the script `master.sh` (launches all relevant the scripts), in the notebook `example.ipynb` (shows how to train a plankton model), and in the notebook `predict.py`.


---

## Repo Structure

The repo contains the following directories:

- `src`: contains auxiliary code.

- `out`: the output is stored here (no output is uploaded to GitHub, so it must be created).

- `data`: the input data is stored here.

- `images`: contains a handful of images, so that a user can see them without downloading the repo.

- `val`: folder for validation.

- `trained-models`: contains small trained models for the end users.

---

## Datasets

The datasets come from the Scripps Camera from the Pomati group at Eawag. We have two datasets, corresponding to two magnifications of the camera [0.5x (mostly zooplankton) and 5x (mostly phytoplankton)]. Everything we write here applies to both datasets, but was only tested on the 0.5x data.


### Importing data

Every time the the field scientists produce new labeled images, a new folder is created, and is labeled with the date of creation. 

We will assume that the data is stored in the directory
`data/1_zooplankton_0p5x`. In the future, phytoplankton data will have a slightly different path.

In order to import the data from the Eawag storage, run the script

```
cd data/1_zooplankton_0p5x
bash update.sh
cd -
```

This will only work if you have access rights to Eawag. 
The data directory will be copied (throug rsync) to `data/1_zooplankton_0p5x/training`.

### Structure of the data

At the moment of writing, we have two data directories
```
ls data/1_zooplankton_0p5x/training/

> zooplankton_trainingset_2020.04.28/         zooplankton_trainingset_2020.07.06/
```

Note that *different data directories may contain different classes*.
Each data directory will contain one subfolder per class. For example:

```
ls data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/cyclops/

> features.tsv  training_data/
```

The file `features.tsv` is a tab-separated-value file, with 67 features of the images obtained during preprocessing. It includes the image name (`url`), information on the size (*e.g.* `estimated_volume`, `area`, `eccentricity`, ...) and information on the colors (*e.g.* `intensity_red_moment_hu_3`).

The folder `training_data` contains the cleaned images

```
ls data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/cyclops/training_data/

> SPC-EAWAG-0P5X-1526948087602056-1089941170938-008779-020-3268-256-108-112.jpeg
> SPC-EAWAG-0P5X-1526948874674178-1090728236363-016649-023-1914-64-124-60.jpeg
> SPC-EAWAG-0P5X-1526948981674456-1090835245259-017719-000-1742-2482-112-117.jpeg
```



### Analyzing the data

In order to get some rough information  a single dataset, run 

```
python analyze_dataset.py -datapath data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28
```

A summary of the dataset will be created (`./out//dataset_info//zooplankton_trainingset_2020.04.28.txt`, the output location and name can be personalized through the flags `-outpath` and `-name`), along with a couple of figures in that same directory, containing class abundances and image size distributions.

The same script can be used to make the joint analysis of two separate datasets:

```
python analyze_dataset.py -datapath ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06
```

---

## Training models

A full example of how to preprocess the data and train models through the plankton classes is shown in `example.ipynb`.

In order to train a model, use `train.py`. 
An example on launching `train.py` for training a model is shown in `master.sh`.
The model runs by itsself with default parameters:

```
python train.py
```
There are lots of input commands that can be given to the script. To query them, use the `-h` flag (`python train.py -h`). 

### Description of `train.py`

The script can be used both as a module to load the `Ctrain` class, or can be directly run, to train a model.

We describe the funcitoning of `train.py` through its command-line input flags


##### I/O

`-datapaths`: is the path (or paths) leading to the directory (or directories) containing the data.

`-outpath`: is the path where we want the output to be written.

`-load_weights`: this flag allows us to initialize the system from a given weight configuration (`.hdf5` file). Loading is done through the `load_weights()` method of the model.

`-modelfile`: filename of the model we want to load (`.h5` file). Loading is done through the `keras.models.load_model()` method.

Examples of manual model loading are shown in `example.ipynb`.

`-saveModelName`: Name of the model when it is saved.

##### User Experience

`-verbose`: Print many messages on screen.

`-plot`: If activated, plots loss and accuracy during training once the run is over.

##### Hyperparameters

`-opt`: Optimizer. Currently, the two choices are `'sgd'` (with Nesterov momentum) and `'sgd'`.

`-bs`: Batch size.

`-lr`: Learning rate.

`-aug`: Boolean variable which decides whether or not to use augmentation. We are only augmenting for `'image'` models. The augmentation kind is currently hard-coded, and it amounts to 90 degrees rotations, vertical and horizontal flips, and 10% shears.

`-modelfile`: The name of the file where a model is stored (to be loaded with `keras.models.load_model()` ).

`-model_image`: For mixed data models, tells what model to use for the image branch. For image models, it is the whole model

`-model_feat`: For mixed data models, tells what model to use for the feature branch. For feat models, it is the whole model.

`-layers`: Layers for MLP models. 

`-dropout`: This is a dropout parameter which is passed to the model wrapper but is currently not used (August 2020) because dropouts are currently hardcoded.

##### Data

`-L`: Images are resized to a square of (L x L) pixels.

`-testSplit`: Fraction of examples in the test set

`-class_select`: Classes to be looked at. If empty, all available classes are studied.

`-datakind`: Whether to load features (`'feat'`), images (`'image'`), or both (`'mixed'`) from the dataset.

`-ttkind`: Whether to load features (`'feat'`), images (`'image'`), or both (`'mixed'`) from the training and test set. This allows for example to load all the data with the `-datakind` flag, and then perform separate tests on subsets of the data without needing to reload the data.

`-training_data`: This is to cope with the different directory structures that we were given. Sometimes the class folder has an extra folder inside, called training_data. For the moment, this happens in the training images they gave me, but not with the validation images.


##### Training time

`-totEpochs`: Total number of epochs for the training.

`-initial_epoch`: Initial epoch of the training.

`-earlyStopping`: If >0, we do early stopping, and this number is the patience (how many epochs without improving).


#### An example of training a model
```
python train.py -datapaths ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/ ./data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.07.06/ -outpath ./dummy_out -opt=adam -class_select bosmina hydra dirt -lr=0.001 -bs=32 -aug -model_image=conv2 -L 128 -datakind=image -ttkind=image -totEpochs=20 -earlyStopping=10
```

---


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

Predictions are made through the `predict.py`, that loads one or more models, and makes predictions on the requested directories.

```
python predict.py   -modelfullnames './trained-models/conv2/keras_model.h5' \  # Names of the models to be loaded
                    -weightnames './trained-models/conv2/bestweights.hdf5' \   # For each model, weights to be loaded
                    -testdirs 'data/1_zooplankton_0p5x/validation/tommy_validation/images/dinobryon/' \ # dirs with images to be checked
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



In the notebook `predict.py` we show how to manually manage predictions.


---

## Validating the results


### Confusion matrix validation

In the main directory you can find the launchable module `validation.py`.

To run a validation just do 
```
python validation.py 2>/dev/null
```

In the following case, we are also specifying abstention threshold, models and specific weights.
```
python validation.py -thresholds 0.0 0.9 -modelnames './trained-models/conv2/keras_model.h5' -weightnames './trained-models/conv2/bestweights.hdf5'
```
The test directories are currently hard-coded, in order not to have to write them each time, since they are always the same.
Abstention thresholds (`-thresholds`) and ensembling methods (`-ensMethods`) can be given as command-line arguments.
The program gives a command-line output with data on the confusion matrix, and a plot with the precision and recall
for all plankton classes (junk classes are excluded).


### Counts validation

The count folders are folders in which the taxonomists counted the number of each species, but without labeling the images one by one. We can only compare the total population measured by the taxonomists, with that of the classifier. This is the ultimate task we will give to the classifier, so it is an excellent validation check.


Input parameters are currently hard-coded, so the syntax is pretty simple:
```
cd ./val/
python validation-counts.py 2>/dev/null
cd ..
```
The script produces a histogram with the taxonomist and the classifier counts.

---



## To do:

- Checkpointing wrapper function
- Scanning of architecture and hyperparameters (e.g. using talos, or create class)
- Different losses and metrics
- Binary classifiers (use `binary_crossentropy loss` function)
- Have an explicit control of initial conditions (currently, we're using default, but for example orthogonal initial conditions could be helpful)
- Implement logging instead of print
- learning rate schedule
- Speed up the data reading
- Hybrid labels
- Merging lables (e.g. Carnivores, Herbivores, Plants)
- Cross Validation
- Weight averaging
- Semi-supervised and active learning
- Make unit tests
- **Write this same list as an *issue*, and remove it from this readme file**

--- 

##
