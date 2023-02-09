FROM docker-prod.se.axis.com/axis/debian-jenkins-slave:buster

# Work directory
ENV HOME=/ReID_Using_DensePose

# Set environment variables for python
ARG PYTHON_VERSION=3.8
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$HOME/.local/bin:$PATH"
ENV PYTHON_VERSION=${PYTHON_VERSION}

WORKDIR $HOME

RUN pip install opencv-python
RUN pip3 install torch torchvision torchaudio
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN python3 -m pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose