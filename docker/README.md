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
