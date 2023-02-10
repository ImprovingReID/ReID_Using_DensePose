#!/bin/bash
# Source python
eval "$(pyenv init -)"

# Install dependencies for Densepose and resnet
PIP_CONSTRAINT=/ReID_Using_DensePose/scripts/constraints.txt
export PIP_CONSTRAINT
python3 -m pip install pip
python3 -m pip install \
  -r /ReID_Using_DensePose/Docker/requirements.in

# Step into working directory
# cd /kpi_workspace

# Display prompt at startup
echo ""
echo "Welcome to the DensePose re-ID docker image"

/bin/bash