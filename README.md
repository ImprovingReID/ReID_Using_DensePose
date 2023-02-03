# ReID_Using_DensePose

This is a master thesis by Anton Fredriksson and Björn Elwin at Axis Communications.

We want to improve Re-ID algorithms using Facebook AI's DensePose

## Setup

Detectron2 and DensePose needs to be downloaded to use the functions for creating Texture maps.

The git to detectron2 is found [here](https://github.com/facebookresearch/detectron2)  and the installationguide is [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) 

Denspose is found within the Detectron2 repository at Detectron2/projects/Densepose and an installation guide for the package is foun at the bottom of [this page](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/GETTING_STARTED.md)

Denspose needs a config file which we have choosen to include locally in this repository, you can find these files at [Densepose/configs](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs)


### Install commands in short
```console
foo@bar:~$ python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
foo@bar:~$ python3 -m pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```

### Config files
Config 

## How to use the functions



## Accessing data