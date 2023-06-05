#!/bin/bash
set -e
USER=`id --user --name` # cluster user
NAME="qunfold" # default container name
RESOURCES="-c 2 -m 64g" # default resources allocated by each container

# find the name of the image (with or without prefix)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -f "${SCRIPT_DIR}/.PUSH" ]; then
    IMAGE="$(cat "${SCRIPT_DIR}/.PUSH")"
elif [ -f "${SCRIPT_DIR}/.IMAGE" ]; then
    IMAGE="$(cat "${SCRIPT_DIR}/.IMAGE")"
else
    echo "ERROR: Could not find any Docker image. Run 'make' or 'make push' first!"
    exit 1
fi

# always print usage information
echo "Found the Docker image ${IMAGE}"
echo "| Usage: $0 [-r <resources>] [-n <name>] [args...]"

# runtime arguments
ARGS=
while [ "$1" != "" ]; do
case "$1" in
    -r|--resources) # configure resources
        RESOURCES="$2"
        shift 2
        ;;
    -n|--name) # configure the container name
        NAME="$2"
        shift 2
        ;;
    *) break ;;
esac
done

# start a single container (only available from gateway machine)
args="${@:1}"
echo "| Resources: '$RESOURCES'"
echo "| Name: ${USER}-${NAME}"
echo "| Args: $args"
read -p "Run this INTERACTIVE container? [y|N] " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker create \
        --tty --interactive \
        --volume /home/${USER}:/mnt/home \
        --name "${USER}-${NAME}" \
        $RESOURCES \
        $IMAGE \
        $args # pass additional arguments to the container entrypoint
    docker start "${USER}-${NAME}"
    docker attach "${USER}-${NAME}"
fi
