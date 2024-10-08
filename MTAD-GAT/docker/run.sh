#!/bin/sh

xhost +local:root

docker run -it \
    --detach \
    --rm  \
    --gpus all \
    -p 8885:8885 \
    --privileged \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${PWD}/..:/mtad-gat" \
    --ipc=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    insujin/anomaly/mtad-gat
    
    bash
