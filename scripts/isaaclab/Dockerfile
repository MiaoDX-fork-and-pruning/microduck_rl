FROM nvcr.io/nvidia/isaac-sim:6.0.1

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    cloudpickle==3.1.2 \
    farama-notifications==0.0.6 \
    flatdict==4.1.0 \
    gymnasium==1.2.0 \
    h5py==3.16.0 \
    lazy-loader==0.4 \
    prettytable==3.3.0 \
    warp-lang==1.13.0 \
    wcwidth==0.2.13

RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0

# Isaac Sim keeps a deprecated ML archive with its own older Torch/NCCL tree.
# Remove that archive so Kit cannot load a conflicting libtorch_cuda.so after
# the IsaacLab-supported Torch installation above.
RUN rm -rf /isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/torch \
    /isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia
