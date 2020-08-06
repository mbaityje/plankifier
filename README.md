# plankifier
Code for plankton dataset creation and classification.

End users (field scientists) should refer to the usage guide provided in the [code releases](https://github.com/mbaityje/plankifier/releases).

This readme file refers to the `master` branch.

---

## Quick Summary

To analyze the datasets, use:

To train, use `train.py`.

To predict, use:

To make ensemble predictions, use:

To validate, use:




### Examples

Examples of usage are contained in the script `master.sh` (launches all relevant the scripts) and in the notebook `example.ipynb` (shows how to train a plankton model).


---

## Datasets

The datasets come from the Scripps Camera from the Pomati group at Eawag. We have two datasets, corresponding to two magnifications of the camera [0.5x (mostly zooplankton) and 5x (mostly phytoplankton)]. Everything we write here applies to both datasets, but was only tested on the 0.5x data.


### Importing data

Every time the the field scientists produce new labeled images, a new folder is created, and is labeled with the date of creation. 
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

In order to train a model, use `train.py`. The model runs by itsself with default parameters:

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

`-load_weights`:

`-saveModelName`:

##### User Experience

##### Hyperparameters

##### Data

##### Training time










#### An example
```
python train.py -datapath 'data/1_zooplankton_0p5x/training/zooplankton_trainingset_2020.04.28/' -outpath 'out/dummy_output' -opt 'adam' -bs 8 -lr 1e-3 -L=128 -model='conv2' -datakind='image' -testSplit=0.2 -totEpochs=5 -dropout=0.5 -aug
```

#### Classes

`Ctrain`: contained in `train.py`.


`Cdata`: contained in `helper_data.py`.


`CTrainTestSet`: contained in `helper_data.py`.

`Cval`

`Censemble`





#### To do:

- Scanning of architecture and hyperparameters (e.g. using talos, or create class)
- Binary classifiers (use binary_crossentropy loss function)
- Have an explicit control of initial conditions (currently, we're using default, but for example orthogonal initial conditions could be helpful)
- Implement logging instead of print
- learning rate schedule
- Speed up the data reading
- Hybrid labels
- Cross Validation
- Write this same list as an *issue*, and remove it from this readme file

--- 

##
