#!/bin/bash
set -e

# runtime arguments
IMAGE="nvcr.io/ml2r/lamarr-dortmund/interactive_jax:23.10-py3"
RESOURCES="-c 8 --mem=64GB -p GPU1 --gres=gpu:1 --container-image=${IMAGE}"
NAME="flax-qunfold"
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
ARGS="${@:1}" # remaining arguments

# print configuration
echo "| Usage: ./srun.sh -n <name> -r \"<resources>\""
echo "| Resources: \"${RESOURCES}\""
echo "| Name: ${NAME}"
echo "Omit the --container-image if you want to reuse a container of the given name."

# check if a session with the given NAME exists
if ! tmux has-session -t ${NAME} 2>/dev/null; then
  read -p "Call srun within the new tmux session \"${NAME}\"? [y|N] " -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux new -s ${NAME} -- "\
      echo '|' Detach: Ctrl+b followed by d; \
      echo '|' Attach: tmux attach -t ${NAME}; \
      srun \
        --export ALL \
        --container-name=${NAME} \
        --job-name=${NAME} \
        ${RESOURCES} \
        --pty /bin/bash; \
      "
  fi
else
  echo "The tmux session \"${NAME}\" already exists; attach with: tmux attach -t ${NAME}"
fi
