#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH -J fe-arm-new-ccsaq
#SBATCH -o /home/gridsan/ihammond/firstepoch-compare-20260629/arm-new-ccsaq/slurm-%j.out

set -u
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
module load julia/1.11.3

export FIRSTEPOCH_OLD_ROOT=/home/gridsan/ihammond/GitHub/Emitter3DTopOpt
export FIRSTEPOCH_NEW_ROOT=/home/gridsan/ihammond/GitHub/worktrees/DEO-firstepoch-20260629
export FIRSTEPOCH_RUNROOT=/home/gridsan/ihammond/firstepoch-compare-20260629
SCRIPTS="$FIRSTEPOCH_NEW_ROOT/scripts/firstepoch-20260629"
export JULIA_NUM_THREADS="${SLURM_CPUS_ON_NODE:-40}"

julia --project="$FIRSTEPOCH_NEW_ROOT" "$SCRIPTS/arm_new_ccsaq.jl"
rc=$?
echo "exit $rc at $(date)"
exit $rc
