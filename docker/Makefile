# constrain the build node: make BUILD_ARGS="--build-arg constraint:node==XXX"
BUILD_ARGS=

GROUP=$(shell id --group --name)
GID=$(shell id --group)
USER=$(shell id --user --name)
UID=$(shell id --user)
CONTAINER_NAME=qunfold

image: .IMAGE
.IMAGE: Dockerfile
	- docker rmi -f $(USER)/$(CONTAINER_NAME)
	docker build \
	    --build-arg group=$(GROUP) \
	    --build-arg gid=$(GID) \
	    --build-arg user=$(USER)-$(CONTAINER_NAME) \
	    --build-arg uid=$(UID) \
	    --tag $(USER)/$(CONTAINER_NAME) \
	    $(BUILD_ARGS) .
	echo "$(USER)/$(CONTAINER_NAME)" > $@

push: .PUSH
.PUSH: .IMAGE
	@ echo "The image '$(USER)/$(CONTAINER_NAME)' is built for the Docker image repository '${DOCKER_REPOSITORY}'."
	@ echo "You can set the name of this repository with the environment variable DOCKER_REPOSITORY.\n"
	- docker rmi -f $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	docker tag $(USER)/$(CONTAINER_NAME) $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	docker push $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	docker pull $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	echo "$(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)" > $@

clean:
	- docker rmi -f $(USER)/$(CONTAINER_NAME)
	- docker rmi -f $(DOCKER_REPOSITORY)/$(USER)/$(CONTAINER_NAME)
	rm -f .IMAGE .PUSH

.PHONY: image push clean
