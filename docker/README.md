# Setting up a Flax-ready container

We proceed as follows, starting on *your local computer* (gwkilab does not support the first two steps).

1. we initialize a git submodule for building standard Docker images at Lamarr.
2. we call our own build script, which uses parts of the standard build process.
3. we start a Slurm job in the way that is usual at Lamarr.

Before we start, you need to install and generate an API token for NVidia's NGC system, install their `ngc` client locally, and have a working `docker` installation. To meet these prerequisites, follow the general instructions from the [Lamarr cluster documentation](https://gitlab.tu-dortmund.de/lamarr/lamarr-public/cluster#custom-docker-images).

Now, initialize the git submodule at `docker/custom-container-example/`:

```
git submodule init
git submodule update
```

Then, call our build script. This process takes quite some time.

```
./build.sh
```

**Note:** You can always pull the git submodule to receive the latest version of the standard build process. Change the `docker/build.sh` script (or set up an entirely new Dockerfile) to customize the Docker image that is being built.

You can now start a Slurm job from this image, from gwkilab, as usual. Consider using the `docker/srun.sh` script for this purpose.






# Legacy notes on earlier and unsuccessful attempts of setting up Flax containers

To update the container's Python to version 3.11, you need to stop your current job and start a spin-off of your job with root privileges. **Caution:** This will terminate all SSH sessions and all computations in your job.

```
srun -c 1 --container-remap-root --container-name=<my container name> --pty /bin/bash
```

Inside the privileged session, build Python 3.11 from the source files:

```
apt update
mv /etc/environment /etc/environment.bck
apt upgrade
mv /etc/environment.bck /etc/environment
apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.11.6/Python-3.11.6.tgz
tar -xf Python-3.11.6.tgz
cd Python-3.11.6
./configure -enable-optimizations
make -j$(nproc)
make altinstall
cd ..
rm -rf Python-3.11.6
rm Python-3.11.6.tgz

apt install -y cuda-compat-12-3
```

Next, install JAX with a CUDA backend:

```
python3.11 -m pip install --upgrade pip setuptools wheel
python3.11 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Finally, clone *qunfold* and install additional flax dependencies:

```
git clone git@github.com:mirkobunse/qunfold.git
cd qunfold
python3.11 -m pip install .[experiments]
python3.11 -m pip install flax>=0.7.5 clu
```
