#!/bin/bash
docker exec -itd matsuda_project tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
