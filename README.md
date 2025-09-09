# straight_flow_celeba_reproduce
Follow up for loss diverge issue reported in https://x.com/LodestoneE621/status/1954757495072882693

Code was copied & modified from the author's gist:

https://gist.github.com/lodestone-rock/97d442ff3a92d53e4256959ead3e969e 

# Import data 
We first import the wandb result from https://wandb.ai/lodestone-rock/straight_flow_celeba
```
wandb login
python ./import.py
```
# Run 1 with public release
Remember to modify Line 682 preview_path in training.py
```
export WANDBKEY=ADD_YOUR_KEY
docker run --rm -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name test rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.6.0 bash -c "git clone https://github.com/lorri-rao/straight_flow_celeba_reproduce.git && pip install wandb einops gdown && wandb login $WANDBKEY && python ./straight_flow_celeba_reproduce/training.py"
```

# Run 2 with rocm nightly release candidate
Remember to modify Line 682 preview_path in training.py
```
docker run --rm -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v  $HOME/.ssh:/root/.ssh --shm-size 64G --name test rocm/pyt-megatron-lm-jax-nightly-private:pytorch_20250908 bash -c "git clone https://github.com/lorri-rao/straight_flow_celeba_reproduce.git && pip install wandb einops gdown && wandb login $WANDBKEY && python ./straight_flow_celeba_reproduce/training.py"
```