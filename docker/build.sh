#!/bin/bash

function create_image { # this function mimics custom-container-example/build.sh
  local IMPORT_NAME=$1
  local EXPORT_NAME=nvcr.io/ml2r/lamarr-dortmund/$2
  echo "Creating ${EXPORT_NAME} from ${IMPORT_NAME}"
  cd custom-container-example/

  # create Dockerfile
  echo "FROM ${IMPORT_NAME}" > Dockerfile
  cat Dockerfile.bak >> Dockerfile

  # remove old image
  echo "Removing the image if it exists"
  docker rmi ${EXPORT_NAME}

  # build
  echo "Building the new image"
  docker build -f Dockerfile --network=host -t ${EXPORT_NAME} .
  docker push ${EXPORT_NAME}

  # clean up
  rm Dockerfile
}

# maintain the "interactive_<name>" convention of the custom-container-example/
create_image \
  "nvcr.io/nvidia/jax:23.10-py3" \
  "interactive_jax:23.10-py3" # 23.10 is the correct version because it uses CUDA 12.2.0
