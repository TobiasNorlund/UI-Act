#!/bin/bash
set -e

# Path to repo/project root dir (independent of pwd)
PROJECT_ROOT=$( cd $(dirname $(readlink -f $0) ); pwd )

# Load environment variables from (non-tracked) .env file
if [ -f "$PROJECT_ROOT/.env" ]
then
    export $(cat $PROJECT_ROOT/.env | xargs)
fi

# Docker image name for this project
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-tobiasnorlund/ui-act}"

# Path to where in the docker container the project root will be mounted
export DOCKER_WORKSPACE_PATH="${DOCKER_WORKSPACE_PATH:-/workspace}"

# Docker user id and group id
export DOCKER_UID=`id -u`
export DOCKER_GID=`id -g`

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --gpu)
    RUNTIME_ARGS="--gpus all"
    shift # past argument
    ;;
    -v|--mount)
    MOUNT="-v $2"
    shift # past argument
    shift # past value
    ;;
    -d|--detach)
    DETACH="--detach"
    shift # past argument
    ;;
    *)    # unknown option
    echo "Unrecognized argument '$1'"
    exit 1
    ;;
esac
done

CONTAINER_NAME="${CONTAINER_NAME:-${DOCKER_IMAGE_NAME//\//-}}"
HOME_VOLUME_NAME="${HOME_VOLUME_NAME:-$CONTAINER_NAME-home}"

# Stop any potentially running container with the same name
docker stop $CONTAINER_NAME 2> /dev/null || true

# Check if there is a running ssh agent, if so we want to forward it
if [[ ! -z "${SSH_AUTH_SOCK}" ]]; then
  SSH_AGENT_FORWARD="-v $SSH_AUTH_SOCK:/ssh-agent -e SSH_AUTH_SOCK=/ssh-agent"
fi

# Optionally set a data dir mount if DATA_DIR env is set
if [[ ! -z "${DATA_DIR}" ]]; then
  DATA_MOUNT="-v $DATA_DIR:$DOCKER_WORKSPACE_PATH/data"
fi


set -x
docker build --rm --build-arg DOCKER_WORKSPACE_PATH --build-arg DOCKER_UID --build-arg DOCKER_GID -t $DOCKER_IMAGE_NAME $PROJECT_ROOT
docker run --rm -it \
  --name $CONTAINER_NAME \
  -v $PROJECT_ROOT:$DOCKER_WORKSPACE_PATH \
  -v $HOME_VOLUME_NAME:/home/docker-user \
  --ipc host \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY \
  $DATA_MOUNT \
  $SSH_AGENT_FORWARD \
  $MOUNT \
  $RUNTIME_ARGS \
  $DETACH \
  $DOCKER_IMAGE_NAME bash
