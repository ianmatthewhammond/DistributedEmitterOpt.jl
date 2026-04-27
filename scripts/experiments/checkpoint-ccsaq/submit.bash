#!/bin/bash
#SBATCH -n 40
#SBATCH -N 1
#SBATCH --output=logs/output-%j.txt
#SBATCH --time=96:00:00
#SBATCH --mem=186G
#SBATCH -p mit_normal
#SBATCH -J "ccsaq-checkpoint"

echo "Job number: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

mkdir -p logs

# NLopt-MMA-CCSA shared library must be built at ~/github/NLopt-MMA-CCSA/build/
export JULIA_NLOPT_MMACCSA_LIB="$HOME/github/NLopt-MMA-CCSA/build/libmmaccsa.so"

OMP_NUM_THREADS=40 JULIA_NUM_THREADS=40 \
    julia --project="$HOME/github/DistributedEmitterOpt.jl-mma-backend" \
    constrained_metal_nominal.jl

echo "End: $(date)"
