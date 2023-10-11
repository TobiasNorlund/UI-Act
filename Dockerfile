FROM python:3.10

ARG DOCKER_UID
ARG DOCKER_GID
RUN groupadd -r docker-user -g $DOCKER_GID && useradd -r -u $DOCKER_UID -g $DOCKER_GID -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y \
    less nano jq git xdotool

COPY --chmod=755 bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
WORKDIR $DOCKER_WORKSPACE_PATH

RUN pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install numpy \
                pytorch_lightning \
                wandb \
                pydantic \
                einops \
                torchmetrics  \
                matplotlib \
                pyautogui \
                decord \
                av \
                pytest

USER $DOCKER_UID:$DOCKER_GID
