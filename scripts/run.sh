#! /bin/bash

SEED=$1
ITER=$2

echo "Running iterion '${ITER}' with seed '${SEED}'"

python truck_trailer_multi_stage_loop_operating_script.py "${SEED}" "${ITER}"
