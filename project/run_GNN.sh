#!/bin/bash
#BSUB -J supply_stgnn
#BSUB -q gpuv100
#BSUB -W 0:30
#BSUB -B
#BSUB -N
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o supply_%J.out
#BSUB -e supply_%J.err



module load cuda/12.1

python -u project/Supply_wind.py