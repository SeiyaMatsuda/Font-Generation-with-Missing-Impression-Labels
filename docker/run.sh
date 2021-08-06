#!/bin/bash
docker run \
    -d \
    --init \
    -p6006:6006 -p5000:5000 -p8888:8888 -p8501:8501 \
    --rm \
    -it \
    --gpus=all \
    --ipc=host \
    --name=matsuda_project \
    --volume=$PWD:/workspace \
    pytorch/matsuda:latest \
    ${@-fish}
