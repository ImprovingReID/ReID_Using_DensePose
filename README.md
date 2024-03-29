# ReID_Using_DensePose

This is a master thesis by Anton Fredriksson and Björn Elwin at Axis Communications.

We want to improve Re-ID algorithms using Facebook AI's DensePose


## Setup
Clone this alongside the detectron2 repository found [here](https://github.com/facebookresearch/detectron2) so the files are repositories are located next to eachother.

The code is easiest run by creating a virtual environment containing all the required pip installations 


### Setup Environment
Create virtual environment
```console
python3 -m venv env
```
Enter virtual environment 
```console
source env/bin/avtivate
```
Install requirements
```console
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html matplotlib tqdm tk opencv-python logger pytorch_metric_learning
```
Add path to densepose and detectron files
```console
pip install -e ../detectron2/projects/DensePose  
pip install -e ../detectron2  
```

kanske torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0

## Setup with docker
This repo is easy to run with running the script docker_run.sh in "scripts" which builds and runs a docker image.
Make sure to change the relatives path to "ReID_Using_DensePose" and "Data" 

``` console 
foo@bar:~$ ./docker_run.sh
```


## Setup without docker

Detectron2 and DensePose needs to be downloaded to use the functions for creating Texture maps.

* The git to detectron2 is found [here](https://github.com/facebookresearch/detectron2)  and the installationguide is [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) 

* Denspose is found within the Detectron2 repository at Detectron2/projects/Densepose and an installation guide for the package is foun at the bottom of [this page](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/GETTING_STARTED.md)

* Denspose needs a config file which we have choosen to include locally in this repository, you can find these files at [Densepose/configs](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose/configs)



### Requirements

* Linux or macOS with Python ≥ 3.7

* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at [pytorch.org](pytorch.org) to make sure of this

* OpenCV is optional but needed by demo and visualization


### Install commands in short
```console
foo@bar:~$ pip install opencv-python  
foo@bar:~$ pip3 install torch torchvision torchaudio
foo@bar:~$ python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
foo@bar:~$ python3 -m pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
```

## How to use the functions




## Accessing data