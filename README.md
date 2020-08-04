# plankifier
Code for plankton dataset creation and classification.

This wiki refers to the `master` branch.

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

## Examples

Examples of usage are contained in the script `master.sh` and in the notebook `example.ipynb`.




