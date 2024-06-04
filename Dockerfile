FROM ghcr.io/walkerlab/docker-pytorch-jupyter-cuda:cuda-11.8.0-pytorch-1.13.0-torchvision-0.14.0-torchaudio-0.13.0-ubuntu-20.04
LABEL maintainer='plume_walkerlab_base_hyak'
RUN apt-get update \
    && apt-get install -y libopenmpi-dev \
    && apt-get install -y tmux \
    && apt-get install -y htop 
RUN pip install --upgrade pip
RUN pip install scikit-learn==1.1.2 yapf==0.33.0 gym==0.21.0 h5py spyder stable_baselines3==1.7.0 moviepy mpi4py \
tqdm urllib3 virtualenv joblib natsort mpl_scatter_density setproctitle \
statsmodels "imageio==2.6.0" "imageio-ffmpeg==0.4.2" array2gif datajoint
RUN pip install git+https://github.com/webermarcolivier/statannot.git
# git cloRUN git clone https://github.com/example/example.git && cd example && git checkout 0123abcdef
