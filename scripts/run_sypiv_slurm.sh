#!/bin/bash
# Example Slurm batch script for running syPIV on an HPC cluster

#SBATCH --job-name=syPIV
#SBATCH --output=syPIV_%j.out
#SBATCH --error=syPIV_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=compute

module load python  # adjust for your cluster

# Activate your environment here, for example:
# source ~/envs/sypiv/bin/activate

DATA_DIR=${DATA_DIR:-"$PWD/data/cylinder_les"}
GRID_FILE=${GRID_FILE:-"$DATA_DIR/cylinder.sp.x"}
FLOW_FILE=${FLOW_FILE:-"$DATA_DIR/sol-0000010.q"}

python -m sypivlib.scripts.batch_sypiv \
  --grid "$GRID_FILE" \
  --flow "$FLOW_FILE" \
  --x-min 0.0 --x-max 0.003 \
  --y-min 0.0 --y-max 0.001 \
  --snapshots 3 \
  --out-dir "$PWD/sypiv_output"

