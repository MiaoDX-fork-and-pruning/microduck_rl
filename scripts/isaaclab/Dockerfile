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

# RSL-RL imports these modules at trainer startup. Install them without
# dependency resolution so pip cannot replace the Isaac Sim Torch/CUDA stack.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
    tensordict==0.10.0 \
    pyvers==0.1.0 \
    importlib-metadata==8.7.0 \
    orjson==3.11.3 \
    GitPython==3.1.45 \
    gitdb==4.0.12 \
    smmap==5.0.2 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    antlr4-python3-runtime==4.9.3 \
    tensorboard==2.20.0 \
    absl-py==2.3.1 \
    grpcio==1.74.0 \
    markdown==3.8.2 \
    protobuf==6.32.0 \
    werkzeug==3.1.3

# Keep Isaac Sim's validated Torch/CUDA 12.8 stack.  RSL-RL declares a broad
# torch>=2.6 dependency and pip would otherwise replace it with Torch 2.14 and
# CUDA 13. Install the trainer without dependency resolution; its runtime
# dependencies that are not already supplied by Isaac Sim are listed above or
# can be diagnosed explicitly by the PPO smoke.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps \
    rsl-rl-lib==5.0.1

# Isaac Sim keeps a deprecated ML archive with its own older Torch/NCCL tree.
# Remove that archive so Kit cannot load a conflicting libtorch_cuda.so after
# the IsaacLab-supported Torch installation above.
RUN rm -rf /isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/torch \
    /isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia
