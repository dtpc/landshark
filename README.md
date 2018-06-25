# landshark

Large-scale spatial inference with Tensorflow.

Designed to work with Aboleth (github.com/data61/aboleth).


## Introduction

Landshark is a set of python command line tools that for supervised learning
problems on large spatial raster datasets (with sparse targets).  It solves
problems in which the user has a set of target point measurements (such as
geochemistry, soil classification, or depth to basement) and wants to relate
those to a number of raster covariates (like satellite imagery or geophysics)
to predict the targets on the raster grid.

Many such tools exist already for this problem. Landshark fills a particular
niche: where we want to efficiently learn models with very large numbers of
training points and/or very large covariate images using tensorflow. Landshark
is particularly useful for the case when the training data itself will not fit
in memory, and must be streamed to a minibatch stochastic gradient descent
algorithm for model learning.

There are no actual models in landshark. It is really just a training and query
data conversion system to get shapefiles and geotiffs to and from 'tfrecord'
files which can be read efficiently by tensorflow.

The choice of models is up to the user: arbitary tensorflow models are fine, as
is anything built on top of tensorflow like Keras or Aboleth. There's also
a scikit-learn interface for problems that do fit in memory (although this is
mainly for validation purposes).


## Data Prerequisites

Landshark has not tried to replicate features that exist in other tools,
especially image processing and GIS. Therefore, it has quite strict
requirements for data input:

1. Targets are stored as points in a shapefile
2. Covariates are stored as a set of geotiffs (1 or more)
3. All geotiffs have the same projection, resolution and bounding box
4. The projection of the targets is the same as that of the geotiffs
4. All target points are inside the bounding box of the geotiffs
5. The covariate images are also those used for prediction (i.e, the prediction
   image will come out with the same resultion and bounding box as the
   covariate images)
6. Geotiffs may have multiple bands, and different geotiffs may have different
   datatypes (eg uint8, float32), but within a single geotiff all bands must
   have the same datatype
7. Geotiffs have been categorized into continuous data (referred to as 'ordinal'), 
   and categorical data. These two sets of geotiffs are stored in separate
   folders.

## Design choices

### Intermediate HDF5 Format

Reading a few pixels from 100 (often compressed) geotiffs is slow. Reading
the same data from an HDF5 file where the 100 bands are contiguous is fast.
Therefore, if we need to do more than a couple of reads from that big geotiff
stack, it saves a lot of time to first re-order all that data to
'band-interleaved by pixel'. We could do this back to geotiff, but HDF5 is
convenient and allows us to store some extra data (and reading is higher
performance too). 


### Configuration as Code

The model specification files in landshark are python functions (or classes in
the case of the sklearn interface). Whilst this presents a larger barrier for
new users if they're not familiar with Python, it allows users to customise
models completely to their particular use-case. A classic example the authors
often come across are custom likelihood functions in Bayesian algorithms.


## Usage Example

We have say a dozen `.tif` covariates stored between our `./ord_images` and
`./cat_images` folder. We have a target shapefile in our `./targets` folder. 
We're going to use the `landshark-import` command to get our data into a format
useable by landshark.

### Import the data

We start by creating a tiff stack called "murray".

```bash
$ landshark-import tifs --ordinal ord_images/ --categorical cat_images --name murray
```

The result of this command will be a file in the current directory called 
`features_murray.hdf5`. Similarly, we import some "Sodium" targets from
a shapefile. Note we can import as many records as we like using multiple
`--record` flags:

```bash
$ landshark-import targets --name sodium --shapefile ./targets/Na.shp --dtype ordinal --record Na_conc --record meas_error
```

We've also specified that the type of the records is ordinal (ie continuous).
From this command, landshark will output `targets_sodium.hdf5`.

### Extract Train/test and Query Data

Let's try putting together a regression problem from the data we just imported.
We're going to use the `landshark-extract` command  for this. 
Starting with a train/test set, we use the 'traintest' sub-command. Note that
`landshark-extract` takes some options, and so does the subcommand:

```bash
$ landshark-extract --features features_murray.hdf5 traintest --targets targets_sodium.hdf5 --name myproblem
```

This command will create a folder in the new directory called
`traintest_myproblem_fold1of10`. The 'fold1of10' part of that folder name is
indicating how the test data were selected. For more information see the
landshark-extract options, but the default is fine for now.

Similarly, we can extract the query data:

```bash
$ landshark-extract --features features_murray.hdf5 query --name myproblem
```

This command will create a folder in a new directory called
`query_myproblem_strip1of1`. The 'strip1of1' indicates the whole image has been
extracted. For large images it is possible to extract only a small (horizontal)
window of the image for iterative processing. See the landshark-extract docs
for details on this.


### Train a Model
We're finally ready to actually train a model. We've set up our model as per
the documentation, in a file called `dnn.py`. There are a couple of model file
examples in the `config_files` folder in this repo.

```bash
$ landshark train dnn.py traintest_myproblem_fold1of10
```
This will start the training off, first creating a folder to store the model
checkpoints called `model_dnn`




